from flask import Flask, render_template, request, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from main import (
    extract_audio,
    transcribe_audio,
    summarize_text,
    text_to_speech,
    progress_status,
    reset_cancel,
)
import time

# Configuration
ALLOWED_EXTENSIONS = {'mp4'}
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 500 * 1024 * 1024))  # 500MB default

app = Flask(__name__)
# Use environment variable for secret key, fallback to default (change in production!)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Directories
VIDEOS_DIR = "static/videos"
RESULTS_DIR = "static/results"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_file_size(file):
    """Check if file size is within limits."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    return file_size <= MAX_FILE_SIZE


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    from main import check_cancel

    if 'video' not in request.files:
        return render_template('error.html', error_message="No video file uploaded. Please select a file."), 400

    video = request.files['video']
    if video.filename == '':
        return render_template('error.html', error_message="No file selected. Please choose a video file."), 400

    # Validate file extension
    if not allowed_file(video.filename):
        return render_template('error.html', error_message="Invalid file type. Only MP4 files are allowed."), 400

    # Validate file size
    if not check_file_size(video):
        return render_template('error.html', error_message=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB."), 400

    # Secure filename to prevent directory traversal
    filename = secure_filename(video.filename)
    base_name = os.path.splitext(filename)[0]
    video_path = os.path.join(VIDEOS_DIR, filename)
    
    try:
        video.save(video_path)
    except Exception as e:
        return render_template('error.html', error_message=f"Error saving file: {str(e)}"), 500

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
        # Removed delay for faster processing
        extract_audio(video_path, audio_path)
        check_cancel()

        transcript = transcribe_audio(audio_path)
        check_cancel()
        
        if not transcript or not transcript.strip():
            return render_template('error.html', error_message="Failed to transcribe audio. The video may not contain speech or audio is unclear."), 500
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        summary = summarize_text(transcript)
        check_cancel()
        
        if not summary or not summary.strip() or "failed" in summary.lower():
            return render_template('error.html', error_message="Failed to generate summary. Please try again with a different video."), 500
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        text_to_speech(summary, summary_audio_path)
        check_cancel()

    except ValueError as ve:
        # Handle validation errors
        return render_template('error.html', error_message=f"Validation error: {str(ve)}"), 400
    except Exception as e:
        if "canceled" in str(e).lower():
            return render_template('error.html', error_message="Processing was canceled by user."), 200
        else:
            return render_template('error.html', error_message=f"Error during processing: {str(e)}"), 500

    return render_template(
        "result.html",
        video_file=f"videos/{filename}",
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
    import main
    main.cancel_requested = True
    return jsonify({"status": "canceled"})


if __name__ == "__main__":
    # Use environment variable for debug mode (default: True for development)
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
