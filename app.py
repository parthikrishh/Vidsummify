from flask import Flask, render_template, request, url_for, jsonify, redirect
from flask_login import LoginManager, login_required, current_user
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import time

from models import db, User, ProcessingHistory, LanguageSupport
from auth import auth_bp
from main import (
    extract_audio,
    transcribe_audio,
    summarize_text,
    text_to_speech,
    progress_status,
    reset_cancel,
    detect_language,
    LANGUAGE_NAMES,
    LANGUAGE_MAP,
)

# Configuration
ALLOWED_EXTENSIONS = {'mp4'}
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 500 * 1024 * 1024))  # 500MB default
SQLALCHEMY_DATABASE_URI = 'sqlite:///vidsummify.db'

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Register blueprints
app.register_blueprint(auth_bp)

# Routes
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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


@app.route('/', methods=['GET'])
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return render_template('index.html', languages=LanguageSupport.get_supported_languages())


@app.route('/upload', methods=['POST'])
@login_required
def upload_video():
    from main import check_cancel

    # Debug logging
    print(f"📨 Upload request received from user: {current_user.username}")
    print(f"📨 Request files: {request.files.keys()}")
    print(f"📨 Request form: {request.form.keys()}")

    if 'video' not in request.files:
        error_msg = "No video file uploaded. Please select a file."
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400

    video = request.files['video']
    if video.filename == '':
        error_msg = "No file selected. Please choose a video file."
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400

    # Validate file extension
    if not allowed_file(video.filename):
        error_msg = "Invalid file type. Only MP4 files are allowed."
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400

    # Validate file size
    if not check_file_size(video):
        error_msg = f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB."
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400

    # Get selected transcription and output language from form
    selected_language = request.form.get('language', 'en').lower()
    if selected_language not in LANGUAGE_NAMES:
        selected_language = 'en'

    output_language = request.form.get('output_language', 'en').lower()
    if output_language not in LANGUAGE_NAMES:
        output_language = 'en'

    # Secure filename to prevent directory traversal
    filename = secure_filename(video.filename)
    if not filename:
        error_msg = "Invalid filename."
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    base_name = os.path.splitext(filename)[0]
    video_path = os.path.join(VIDEOS_DIR, filename)
    
    try:
        video.save(video_path)
        print(f"✅ File saved: {video_path}")
    except Exception as e:
        error_msg = f"Error saving file: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

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

    # Timing
    start_time = time.time()
    extraction_time = 0
    transcription_time = 0
    summarization_time = 0
    tts_time = 0
    detected_language = 'en'

    try:
        # Step 1: Extract audio
        step_start = time.time()
        print(f"📊 Step 1: Extracting audio...")
        extract_audio(video_path, audio_path)
        check_cancel()
        extraction_time = time.time() - step_start
        print(f"✅ Audio extracted in {extraction_time:.2f}s")

        # Step 2: Transcribe audio with selected language
        step_start = time.time()
        print(f"📊 Step 2: Transcribing audio...")
        transcript = transcribe_audio(audio_path, language=selected_language)
        check_cancel()
        transcription_time = time.time() - step_start
        print(f"✅ Transcription completed in {transcription_time:.2f}s")
        
        if not transcript or not transcript.strip():
            error_msg = "Failed to transcribe audio. The video may not contain speech or audio is unclear."
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Detect actual language from transcript
        try:
            detected_language, _ = detect_language(transcript[:500])
        except:
            detected_language = selected_language
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Step 3: Summarize text
        step_start = time.time()
        print(f"📊 Step 3: Summarizing text...")
        summary = summarize_text(transcript)
        check_cancel()
        summarization_time = time.time() - step_start
        print(f"✅ Summarization completed in {summarization_time:.2f}s")

        # Translate transcript and summary if needed
        if output_language != detected_language:
            print(f"🌐 Translating transcript and summary from {detected_language} to {output_language}")
            try:
                from textblob import TextBlob
                transcript_blob = TextBlob(transcript)
                transcript = str(transcript_blob.translate(to=output_language))
                summary_blob = TextBlob(summary)
                summary = str(summary_blob.translate(to=output_language))
            except Exception as translate_error:
                print(f"⚠️ Translation failed: {translate_error}")

        if not summary or not summary.strip():
            error_msg = "Failed to generate summary. The transcript may be too short or unclear."
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        # Step 4: Generate speech
        step_start = time.time()
        print(f"📊 Step 4: Generating speech...")
        text_to_speech(summary, summary_audio_path)
        check_cancel()
        tts_time = time.time() - step_start
        print(f"✅ Speech generated in {tts_time:.2f}s")

    except ValueError as ve:
        error_msg = f"Validation error: {str(ve)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = str(e)
        if "canceled" in error_msg.lower():
            print(f"⚠️ Processing canceled")
            return jsonify({'error': 'Processing was canceled by user.'}), 200
        else:
            print(f"❌ Processing error: {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f"Error during processing: {error_msg}"}), 500

    # Calculate metrics
    total_time = time.time() - start_time
    transcript_words = len(transcript.split())
    summary_words = len(summary.split())
    compression_ratio = summary_words / transcript_words if transcript_words > 0 else 0
    original_size = os.path.getsize(video_path) / (1024 * 1024)  # MB

    # Save processing history
    try:
        history = ProcessingHistory(
            user_id=current_user.id,
            video_filename=filename,
            original_video_size=original_size,
            language_detected=LANGUAGE_NAMES.get(detected_language, 'English'),
            language=LANGUAGE_NAMES.get(selected_language, 'English'),
            audio_path=audio_path,
            transcript_path=transcript_path,
            summary_path=summary_path,
            summary_audio_path=summary_audio_path,
            transcript_length=transcript_words,
            summary_length=summary_words,
            compression_ratio=compression_ratio,
            processing_time=total_time,
            extraction_time=extraction_time,
            transcription_time=transcription_time,
            summarization_time=summarization_time,
            tts_time=tts_time,
            status='completed'
        )
        db.session.add(history)
        db.session.commit()
    except Exception as db_error:
        print(f"⚠️ Warning: Could not save processing history: {db_error}")

    # Success response
    print(f"✅ Processing completed in {total_time:.2f}s")
    return jsonify({
        'success': True,
        'video_file': f"videos/{filename}",
        'transcript': transcript,
        'summary': summary,
        'summary_audio': f"results/{base_name}/{base_name}_summary_audio.mp3",
        'video_name': base_name,
        'compression_ratio': round(compression_ratio, 2),
        'transcript_words': transcript_words,
        'summary_words': summary_words,
        'processing_time': round(total_time, 2)
    }), 200


@app.route('/result', methods=['GET'])
@login_required
def result():
    """Display processing results - gets data from last successful processing"""
    latest = ProcessingHistory.query.filter_by(user_id=current_user.id).order_by(
        ProcessingHistory.created_at.desc()
    ).first()
    
    if not latest:
        return redirect(url_for('home'))
    
    return render_template('result.html',
        video_file=f"videos/{latest.video_filename}",
        transcript=open(latest.transcript_path, 'r', encoding='utf-8').read() if os.path.exists(latest.transcript_path) else "Transcript not found",
        summary=open(latest.summary_path, 'r', encoding='utf-8').read() if os.path.exists(latest.summary_path) else "Summary not found",
        summary_audio=latest.summary_audio_path,
        video_name=os.path.splitext(latest.video_filename)[0],
        compression_ratio=round(latest.compression_ratio, 2) if latest.compression_ratio else 0,
        transcript_words=latest.transcript_length if latest.transcript_length else 0,
        summary_words=latest.summary_length if latest.summary_length else 0,
        processing_time=round(latest.processing_time, 2) if latest.processing_time else 0
    )


@app.route('/history', methods=['GET'])
@login_required
def history():
    """Show user's processing history"""
    page = request.args.get('page', 1, type=int)
    history_items = current_user.processing_history.order_by(
        db.desc(ProcessingHistory.created_at)
    ).paginate(page=page, per_page=10)
    
    return render_template('history.html', history=history_items)


@app.route('/api/history/<int:history_id>', methods=['GET'])
@login_required
def get_history_item(history_id):
    """Get history item details as JSON"""
    item = ProcessingHistory.query.get_or_404(history_id)
    if item.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(item.to_dict())


@app.route('/api/statistics', methods=['GET'])
@login_required
def get_statistics():
    """Get user statistics"""
    stats = current_user.get_stats()
    return jsonify(stats)


# ✅ Progress route for live frontend updates
@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(progress_status)


# ✅ Cancel route to abort processing
@app.route('/cancel', methods=['POST'])
def cancel_processing():
    import main
    main.cancel_requested = True
    return jsonify({"status": "canceled"})


@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_message="Page not found (404)"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error_message="Server error (500)"), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
