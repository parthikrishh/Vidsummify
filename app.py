from flask import Flask, render_template, request, url_for, jsonify
import os
from main import (
    extract_audio,
    transcribe_audio,
    summarize_text,
    text_to_speech,
    progress_status,
    reset_cancel,
)
import time

app = Flask(__name__)

# Directories
VIDEOS_DIR = "static/videos"
RESULTS_DIR = "static/results"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    from main import cancel_requested, check_cancel

    if 'video' not in request.files:
        return "❌ No video uploaded."

    video = request.files['video']
    if video.filename == '':
        return "❌ No file selected."

    base_name = os.path.splitext(video.filename)[0]
    video_path = os.path.join(VIDEOS_DIR, video.filename)
    video.save(video_path)

    # Create a folder for each video's results
    video_result_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(video_result_dir, exist_ok=True)

    # Reset cancel flag for new upload
    reset_cancel()

    # Define output paths
    audio_path = os.path.join(video_result_dir, f"{base_name}_audio.wav")
    transcript_path = os.path.join(video_result_dir, f"{base_name}_transcript.txt")
    summary_path = os.path.join(video_result_dir, f"{base_name}_summary.txt")
    summary_audio_path = os.path.join(video_result_dir, f"{base_name}_summary_audio.mp3")

    try:
        time.sleep(1)
        extract_audio(video_path, audio_path)
        check_cancel()

        transcript = transcribe_audio(audio_path)
        check_cancel()
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        summary = summarize_text(transcript)
        check_cancel()
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        text_to_speech(summary, summary_audio_path)
        check_cancel()

    except Exception as e:
        if "canceled" in str(e).lower():
            return "❌ Processing canceled by user."
        else:
            return f"❌ Error during processing: {e}"

    return render_template(
        "result.html",
        video_file=f"videos/{video.filename}",
        transcript=transcript,
        summary=summary,
        summary_audio=f"results/{base_name}/{base_name}_summary_audio.mp3",
        video_name=base_name
    )


# ✅ Progress route for live frontend updates
@app.route('/progress')
def get_progress():
    return jsonify(progress_status)


# ✅ Cancel route to abort processing
@app.route('/cancel', methods=['POST'])
def cancel_processing():
    from main import cancel_requested
    globals()['cancel_requested'] = True
    return jsonify({"status": "canceled"})


if __name__ == "__main__":
    app.run(debug=True)
