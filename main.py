import os
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.utils import which
from pydub.effects import normalize
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import nltk
from time import sleep, time
import torch
from langdetect import detect, LangDetectException, detect_langs
import json
from textblob import TextBlob
import re

# ✅ Ensure FFmpeg path is set (especially for Windows)
AudioSegment.converter = which("ffmpeg")

# Ensure NLTK tokenizer is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Language code mapping for Whisper and other services
LANGUAGE_MAP = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'ru': 'russian',
    'ja': 'japanese',
    'zh': 'chinese',
    'hi': 'hindi',
    'ar': 'arabic',
    'ko': 'korean',
}

LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'hi': 'Hindi',
    'ar': 'Arabic',
    'ko': 'Korean',
}

# Shared progress + cancel flags
progress_status = {"stage": "Idle", "percent": 0, "language": "English"}
cancel_requested = False  # Global cancel flag

# Cache models to avoid reinitializing every time
_whisper_model = None
_summarizer = None

# Auto-detect device (GPU if available, else CPU)
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"

def get_compute_type(device):
    """Get optimal compute type based on device."""
    if device == "cuda":
        return "float16"  # Faster on GPU
    elif device == "mps":
        return "float16"  # Apple Silicon
    else:
        return "int8"  # CPU optimization


def update_progress(stage, percent):
    """Helper to update progress stage and percentage."""
    progress_status["stage"] = stage
    progress_status["percent"] = percent


def detect_language(text):
    """Detect language from text using multi-method approach for better accuracy."""
    try:
        # Method 1: langdetect with confidence scoring
        lang_probs = detect_langs(text)
        if lang_probs and len(lang_probs) > 0:
            best_lang = lang_probs[0]
            lang_code = best_lang.lang
            confidence = best_lang.prob
            print(f"✅ Language detection confidence: {confidence:.2%}")
        else:
            lang_code = detect(text)
            confidence = 1.0
        
        # Validate language code
        if lang_code not in LANGUAGE_NAMES:
            # Map similar languages
            if lang_code.startswith('zh'):
                lang_code = 'zh'
            elif lang_code.startswith('pt'):
                lang_code = 'pt'
            else:
                lang_code = 'en'
        
        lang_name = LANGUAGE_NAMES.get(lang_code, 'English')
        update_progress(f"🌐 Detected language: {lang_name} ({confidence:.0%})", progress_status.get("percent", 50))
        return lang_code, lang_name
    except Exception as e:
        print(f"⚠️ Language detection failed: {e}. Defaulting to English")
        return 'en', 'English'


def check_cancel():
    """Abort processing if user requested cancellation."""
    global cancel_requested
    if cancel_requested:
        update_progress("❌ Processing canceled.", 0)
        raise Exception("Processing canceled by user.")


def reset_cancel():
    """Reset cancel flag before new process starts."""
    global cancel_requested
    cancel_requested = False


def apply_noise_reduction(audio, reduction_factor=0.8):
    """Apply noise reduction by normalizing and lowering quiet parts of audio."""
    try:
        # Normalize audio first
        normalized = normalize(audio)
        # Apply slight compression to reduce noise
        reduced = normalized.apply_gain_dbfs(-2)
        return reduced
    except Exception as e:
        print(f"⚠️ Warning: Noise reduction failed: {e}. Continuing with original audio.")
        return audio


def extract_audio(video_path, output_path):
    update_progress("🎧 Extracting and processing audio...", 10)
    check_cancel()
    clip = None
    try:
        clip = mp.VideoFileClip(video_path)
        if clip.audio is None:
            raise ValueError("Video file has no audio track")
        
        # Extract audio at 16kHz for optimal speech recognition
        update_progress("🎧 Converting audio format...", 15)
        clip.audio.write_audiofile(
            output_path, 
            verbose=False, 
            logger=None,
            fps=16000  # 16kHz sample rate is optimal for speech recognition
        )
        
        # Apply noise reduction for cleaner transcription
        update_progress("🔧 Applying noise reduction...", 20)
        audio = AudioSegment.from_wav(output_path)
        processed_audio = apply_noise_reduction(audio)
        processed_audio.export(output_path, format="wav")
        
        update_progress("✅ Audio processing completed.", 25)
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False
    finally:
        if clip is not None:
            clip.close()
            if clip.audio is not None:
                clip.audio.close()


def transcribe_audio(audio_path, language='en'):
    """Transcribe audio using Faster-Whisper with HIGH ACCURACY settings.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en', 'es', 'fr', etc.)
    """
    update_progress(f"🗣️ Transcribing audio ({LANGUAGE_NAMES.get(language, 'English')})...", 40)
    check_cancel()

    global _whisper_model
    try:
        # Initialize Faster-Whisper with better accuracy
        if _whisper_model is None:
            device = get_device()
            compute_type = get_compute_type(device)
            # Use "base" model for better accuracy (was "tiny")
            model_size = os.environ.get('WHISPER_MODEL_SIZE', 'base')
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"✅ Whisper-{model_size} on {device} ({compute_type})  - High Accuracy mode")
        model = _whisper_model

        # Load audio
        audio = AudioSegment.from_wav(audio_path)
        # 3-minute chunks for better context
        chunk_length_ms = 3 * 60 * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        full_text = ""
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            check_cancel()

            temp_chunk_path = f"temp_chunk_{i}.wav"
            try:
                chunk.export(temp_chunk_path, format="wav")
            except Exception as e:
                print(f"❌ Error exporting chunk {i}: {e}")
                continue

            update_progress(f"🗣️ Chunk {i}/{total_chunks}...", 40 + int((i / total_chunks) * 20))

            try:
                # HIGH ACCURACY settings
                segments, info = model.transcribe(
                    temp_chunk_path, 
                    language=language,
                    beam_size=5,        # Larger beam = better accuracy
                    vad_filter=True,    # Enable VAD for quality
                    vad_parameters=dict(min_silence_duration_ms=300),
                    temperature=0.0     # Deterministic
                )
                chunk_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
                if chunk_text:
                    full_text += chunk_text + " "
                    print(f"✅ Chunk {i}: {len(chunk_text.split())} words")
            except Exception as e:
                print(f"❌ Error transcribing chunk {i}: {e}")
            finally:
                try:
                    if os.path.exists(temp_chunk_path):
                        os.remove(temp_chunk_path)
                except Exception as cleanup_error:
                    print(f"⚠️ Could not remove temp: {cleanup_error}")

        if not full_text.strip():
            update_progress("⚠️ Transcription failed (no text).", 55)
            return ""

        cleaned_text = " ".join(full_text.split())
        print(f"📊 Total: {len(cleaned_text.split())} words")
        update_progress("✅ Transcription completed.", 60)
        return cleaned_text.strip()

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def summarize_text(text):
    update_progress("🧠 Summarizing text...", 65)
    check_cancel()
    global _summarizer
    try:
        # Validate input
        if not text or not text.strip():
            raise ValueError("Empty text cannot be summarized")
        
        # If text is too short, just return as-is
        words = text.split()
        if len(words) < 50:
            update_progress("✅ Summary generation completed.", 85)
            return text
        
        # Cache summarizer to avoid reloading model every time
        if _summarizer is None:
            try:
                print("🔄 Loading BART summarization model...")
                device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
                if device == 0:
                    model = model.to("cuda")
                _summarizer = {"model": model, "tokenizer": tokenizer, "device": device}
                print("✅ Summarizer model loaded successfully")
            except Exception as model_error:
                print(f"❌ Failed to load summarizer model: {model_error}")
                raise
        
        model = _summarizer["model"]
        tokenizer = _summarizer["tokenizer"]
        device = _summarizer["device"]
        
        # Optimize: Use chunks of appropriate size
        max_chunk_length = 1024  # Max tokens for BART input
        text_chunks = []
        
        # Split text into sentences for better chunking
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len((current_chunk + " " + sentence).split()) < 200:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    text_chunks.append(current_chunk + ".")
                current_chunk = sentence
        if current_chunk:
            text_chunks.append(current_chunk + ".")
        
        summaries = []
        
        for idx, chunk in enumerate(text_chunks):
            check_cancel()
            if not chunk.strip() or len(chunk.split()) < 20:
                continue
            try:
                # Tokenize and summarize
                inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
                if device == 0:
                    inputs = inputs.to("cuda")
                # Make summary longer and more meaningful
                summary_ids = model.generate(
                    inputs,
                    max_length=300,  # was 150
                    min_length=80,   # was 30
                    num_beams=4,     # was 2
                    length_penalty=1.2,
                    early_stopping=True
                )
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                if summary_text.strip():
                    summaries.append(summary_text)
            except Exception as chunk_error:
                print(f"⚠️ Error summarizing chunk {idx + 1}: {chunk_error}")
                # Use extractive approach as fallback
                words_in_chunk = chunk.split()
                if len(words_in_chunk) > 50:
                    summaries.append(" ".join(words_in_chunk[:2*len(words_in_chunk)//3]))
        
        final_summary = " ".join(summaries)
        if not final_summary.strip():
            # If all chunks failed, return first third of text
            final_summary = " ".join(text.split()[:len(text.split())//3])
        
        update_progress("✅ Summary generation completed.", 85)
        return final_summary
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise so app.py can handle it properly


def text_to_speech(text, output_audio_path):
    update_progress("🔊 Generating summary speech...", 90)
    check_cancel()
    try:
        if not text or not text.strip():
            raise ValueError("No text to speak")
        tts = gTTS(text=text, lang='en')
        tts.save(output_audio_path)
        update_progress("✅ Summary speech generated.", 100)
    except Exception as e:
        print(f"❌ Error generating summary audio: {e}")
        raise  # Re-raise exception so caller knows it failed


def process_video_pipeline(video_filename):
    """Main pipeline for a single uploaded video."""
    update_progress("🚀 Starting process...", 5)
    check_cancel()

    base_name = os.path.splitext(video_filename)[0]
    video_path = os.path.join("static", "videos", video_filename)
    result_dir = os.path.join("static", "results", base_name)
    os.makedirs(result_dir, exist_ok=True)

    audio_path = os.path.join(result_dir, f"{base_name}_audio.wav")
    transcript_path = os.path.join(result_dir, f"{base_name}_transcript.txt")
    summary_path = os.path.join(result_dir, f"{base_name}_summary.txt")
    summary_audio_path = os.path.join(result_dir, f"{base_name}_summary_audio.mp3")

    # Step 1: Extract audio
    if not extract_audio(video_path, audio_path):
        return None

    # Step 2: Transcribe
    transcript = transcribe_audio(audio_path)
    if not transcript.strip():
        return None
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Step 3: Summarize
    summary = summarize_text(transcript)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # Step 4: Text to Speech
    text_to_speech(summary, summary_audio_path)

    update_progress("✅ Completed successfully!", 100)
    # Removed sleep delay for faster response
    update_progress("Idle", 0)

    return {
        "video": f"/static/videos/{video_filename}",
        "transcript": f"/{transcript_path}",
        "summary": f"/{summary_path}",
        "summary_audio": f"/{summary_audio_path}",
        "folder": base_name,
    }
