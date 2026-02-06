from flask import Flask, render_template, request, jsonify
import os
import time
import main

app = Flask(__name__)

VIDEOS_DIR = "static/videos"
RESULTS_DIR = "static/results"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():

    if "video" not in request.files:
        return "No video uploaded."

    video = request.files["video"]

    if video.filename == "":
        return "No file selected."

    base_name = os.path.splitext(video.filename)[0]
    video_path = os.path.join(VIDEOS_DIR, video.filename)
    video.save(video_path)

    result_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(result_dir, exist_ok=True)

    main.reset_cancel()

    audio_path = os.path.join(result_dir, f"{base_name}_audio.wav")
    transcript_path = os.path.join(result_dir, f"{base_name}_transcript.txt")
    summary_path = os.path.join(result_dir, f"{base_name}_summary.txt")
    summary_audio_path = os.path.join(result_dir, f"{base_name}_summary_audio.mp3")

    try:
        main.extract_audio(video_path, audio_path)

        transcript = main.transcribe_audio(audio_path)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        summary = main.summarize_text(transcript)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        main.text_to_speech(summary, summary_audio_path)

    except Exception as e:
        return f"Processing failed: {e}"

    return render_template(
        "result.html",
        video_file=f"videos/{video.filename}",
        transcript=transcript,
        summary=summary,
        summary_audio=f"results/{base_name}/{base_name}_summary_audio.mp3",
    )


@app.route("/progress")
def progress():
    return jsonify(main.progress_status)


@app.route("/cancel", methods=["POST"])
def cancel():
    main.cancel_requested = True
    return jsonify({"status": "canceled"})


if __name__ == "__main__":
    app.run(debug=True)
