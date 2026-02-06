import os
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.utils import which
from faster_whisper import WhisperModel
from transformers import pipeline
from gtts import gTTS
import nltk

AudioSegment.converter = which("ffmpeg")

nltk.download("punkt", quiet=True)

progress_status = {"stage": "Idle", "percent": 0}
cancel_requested = False


def update_progress(stage, percent):
    progress_status["stage"] = stage
    progress_status["percent"] = percent


def check_cancel():
    global cancel_requested
    if cancel_requested:
        update_progress("Canceled", 0)
        raise Exception("Canceled")


def reset_cancel():
    global cancel_requested
    cancel_requested = False


# 🔥 Load Whisper ONCE
WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")

# 🔥 Load summarizer ONCE
SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def extract_audio(video_path, output_path):
    update_progress("Extracting audio", 15)
    check_cancel()

    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path, verbose=False, logger=None)

    update_progress("Audio done", 30)


def transcribe_audio(audio_path):

    update_progress("Transcribing", 45)
    check_cancel()

    segments, _ = WHISPER_MODEL.transcribe(audio_path)

    text = " ".join([s.text for s in segments])

    update_progress("Transcription done", 65)

    return text


def summarize_text(text):

    update_progress("Summarizing", 75)
    check_cancel()

    result = SUMMARIZER(text[:3000])[0]["summary_text"]

    update_progress("Summary done", 90)

    return result


def text_to_speech(text, output_audio_path):

    update_progress("Text to speech", 95)
    check_cancel()

    gTTS(text=text, lang="en").save(output_audio_path)

    update_progress("Finished", 100)
