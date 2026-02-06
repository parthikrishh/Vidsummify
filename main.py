import os
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.utils import which
from faster_whisper import WhisperModel
from transformers import pipeline
from gtts import gTTS
import nltk
from time import sleep

# ✅ Ensure FFmpeg path is set (especially for Windows)
AudioSegment.converter = which("ffmpeg")

# Ensure NLTK tokenizer is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Shared progress + cancel flags
progress_status = {"stage": "Idle", "percent": 0}
cancel_requested = False  # Global cancel flag


def update_progress(stage, percent):
    """Helper to update progress stage and percentage."""
    progress_status["stage"] = stage
    progress_status["percent"] = percent


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


def extract_audio(video_path, output_path):
    update_progress("🎧 Extracting audio from video...", 10)
    check_cancel()
    try:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path, verbose=False, logger=None)
        update_progress("✅ Audio extraction completed.", 25)
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False


def transcribe_audio(audio_path):
    """Transcribe audio using Faster-Whisper with chunking for speed."""
    update_progress("🗣️ Transcribing audio to text...", 40)
    check_cancel()

    try:
        # Initialize Faster-Whisper model (base or small for speed)
        model_size = "base"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Load audio using pydub
        audio = AudioSegment.from_wav(audio_path)
        chunk_length_ms = 5 * 60 * 1000  # 5 minutes
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

            update_progress(f"🗣️ Processing chunk {i}/{total_chunks}...", 40 + int((i / total_chunks) * 20))

            try:
                segments, _ = model.transcribe(temp_chunk_path, language="en")
                chunk_text = " ".join([seg.text for seg in segments])
                full_text += chunk_text + " "
            except Exception as e:
                print(f"❌ Error transcribing chunk {i}: {e}")
            finally:
                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)

        if not full_text.strip():
            update_progress("⚠️ Transcription failed (no text).", 55)
            return ""

        update_progress("✅ Transcription completed.", 60)
        return full_text.strip()

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return ""


def summarize_text(text):
    update_progress("🧠 Summarizing text...", 65)
    check_cancel()
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        max_chunk_size = 1000
        text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        summaries = []
        for chunk in text_chunks:
            check_cancel()
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        update_progress("✅ Summary generation completed.", 85)
        return " ".join(summaries)
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return "Summary generation failed."


def text_to_speech(text, output_audio_path):
    update_progress("🔊 Generating summary speech...", 90)
    check_cancel()
    try:
        if not text.strip():
            raise ValueError("No text to speak")
        tts = gTTS(text=text, lang='en')
        tts.save(output_audio_path)
        update_progress("✅ Summary speech generated.", 100)
    except Exception as e:
        print(f"❌ Error generating summary audio: {e}")


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
    sleep(1)
    update_progress("Idle", 0)

    return {
        "video": f"/static/videos/{video_filename}",
        "transcript": f"/{transcript_path}",
        "summary": f"/{summary_path}",
        "summary_audio": f"/{summary_audio_path}",
        "folder": base_name,
    }
