import sys

# Fix Windows console encoding for emoji/unicode support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from flask import Flask, render_template, request, url_for, jsonify, redirect, send_file, session
from flask_login import LoginManager, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
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
    get_progress_status,
    reset_cancel,
    request_cancel,
    set_current_session,
    clear_session,
    detect_language,
    LANGUAGE_NAMES,
    LANGUAGE_MAP,
)

# Load environment variables from .env file
load_dotenv()

# Configuration
ALLOWED_EXTENSIONS = {'mp4'}
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 500 * 1024 * 1024))  # 500MB default
SQLALCHEMY_DATABASE_URI = 'sqlite:///vidsummify.db'

app = Flask(__name__)

# Security: Load SECRET_KEY from .env (required for production)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("SECRET_KEY must be set in .env environment variable!")

app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour CSRF token validity

# Initialize extensions
db.init_app(app)
csrf = CSRFProtect(app)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://"
)

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
RESULTS_DIR = "static/results"

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
@limiter.limit("10 per hour")  # Rate limit uploads
def upload_video():
    from main import check_cancel
    import traceback

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

    # Get selected language from form
    selected_language = request.form.get('language', 'en').lower()
    if selected_language not in LANGUAGE_NAMES:
        selected_language = 'en'

    # Secure filename to prevent directory traversal
    filename = secure_filename(video.filename)
    if not filename:
        error_msg = "Invalid filename."
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    base_name = os.path.splitext(filename)[0]
    
    # Add unique timestamp to prevent filename collisions
    unique_id = int(time.time() * 1000)
    unique_base_name = f"{base_name}_{unique_id}"
    unique_filename = f"{unique_base_name}.mp4"
    

    # Save video to temp directory (will be deleted after processing)
    import tempfile
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, unique_filename)

    try:
        video.save(video_path)
        original_size = os.path.getsize(video_path) / (1024 * 1024)  # MB - get size before processing
        print(f"✅ File saved to temp: {video_path}")
    except Exception as e:
        error_msg = f"Error saving file: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

    # Create a unique folder for each video's results
    video_result_dir = os.path.join(RESULTS_DIR, unique_base_name)
    os.makedirs(video_result_dir, exist_ok=True)

    # Copy video to static results directory for frontend access
    static_video_path = os.path.join(video_result_dir, f"{unique_base_name}.mp4")
    try:
        import shutil
        shutil.copy2(video_path, static_video_path)
        print(f"✅ Video copied to static: {static_video_path}")
    except Exception as e:
        print(f"⚠️ Could not copy video to static: {e}")
        static_video_path = None

    # Create unique session ID for this processing job
    processing_session_id = f"{current_user.id}_{int(time.time() * 1000)}"
    session['processing_session_id'] = processing_session_id
    
    # Initialize session for processing
    reset_cancel(processing_session_id)
    set_current_session(processing_session_id)


    # Define output paths (using unique_base_name for uniqueness)
    audio_path = os.path.join(video_result_dir, f"{unique_base_name}_audio.wav")
    transcript_path = os.path.join(video_result_dir, f"{unique_base_name}_transcript.txt")
    summary_path = os.path.join(video_result_dir, f"{unique_base_name}_summary.txt")
    summary_audio_path = os.path.join(video_result_dir, f"{unique_base_name}_summary_audio.mp3")

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
            traceback.print_exc()
            return jsonify({'error': f"Error during processing: {error_msg}"}), 500

    # Clean up temp video file
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"✅ Temp video deleted: {video_path}")
    except Exception as cleanup_error:
        print(f"⚠️ Could not delete temp video: {cleanup_error}")
    
    # Calculate metrics
    total_time = time.time() - start_time
    transcript_words = len(transcript.split())
    summary_words = len(summary.split())
    compression_ratio = summary_words / transcript_words if transcript_words > 0 else 0

    # Save processing history

    try:
        # Store the static video path relative to 'static/' for frontend use
        video_rel_path = None
        if static_video_path and ("static/" in static_video_path or "static\\" in static_video_path):
            video_rel_path = static_video_path.replace("\\", "/").split("static/")[-1]
        else:
            video_rel_path = static_video_path.replace("\\", "/") if static_video_path else ""

        history = ProcessingHistory(
            user_id=current_user.id,
            video_filename=unique_filename,
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
            status='completed',
            # Store the static video path for result page
            tags=video_rel_path or ""
        )
        db.session.add(history)
        db.session.commit()
    except Exception as db_error:
        print(f"⚠️ Warning: Could not save processing history: {db_error}")

    # Success response
    print(f"✅ Processing completed in {total_time:.2f}s")
    return jsonify({
        'success': True,
        'transcript': transcript,
        'summary': summary,
        'summary_audio': f"results/{unique_base_name}/{unique_base_name}_summary_audio.mp3",
        'video_name': base_name,
        'compression_ratio': round(compression_ratio, 2),
        'transcript_words': transcript_words,
        'summary_words': summary_words,
        'processing_time': round(total_time, 2),
        'video_url': video_rel_path or ""
    }), 200



@app.route('/result', methods=['GET'])
@app.route('/result/<int:history_id>', methods=['GET'])
@login_required
def result(history_id=None):
    """Display processing results - gets data from specific history record or last successful processing"""
    if history_id:
        # Get specific history record
        history = ProcessingHistory.query.filter_by(
            id=history_id,
            user_id=current_user.id
        ).first()
    else:
        # Get latest
        history = ProcessingHistory.query.filter_by(user_id=current_user.id).order_by(
            ProcessingHistory.created_at.desc()
        ).first()
    
    if not history:
        return redirect(url_for('home'))
    
    # Get the relative audio path from stored path (after 'static/')
    summary_audio_file = None
    if history.summary_audio_path and os.path.exists(history.summary_audio_path):
        # Convert to relative path for static folder
        if 'static/' in history.summary_audio_path or 'static\\' in history.summary_audio_path:
            summary_audio_file = history.summary_audio_path.replace('\\', '/').split('static/')[-1]
        else:
            summary_audio_file = history.summary_audio_path.replace('\\', '/')
    
    if not summary_audio_file:
        print(f"⚠️ Audio file not found: {history.summary_audio_path}")
        summary_audio_file = ""

    # Get the video URL from tags field (used for video path)
    video_url = history.tags if history.tags else ""

    return render_template('result.html',
        transcript=open(history.transcript_path, 'r', encoding='utf-8').read() if os.path.exists(history.transcript_path) else "Transcript not found",
        summary=open(history.summary_path, 'r', encoding='utf-8').read() if os.path.exists(history.summary_path) else "Summary not found",
        summary_audio=summary_audio_file,
        video_name=os.path.splitext(history.video_filename)[0],
        compression_ratio=round(history.compression_ratio, 2) if history.compression_ratio else 0,
        transcript_words=history.transcript_length if history.transcript_length else 0,
        summary_words=history.summary_length if history.summary_length else 0,
        processing_time=round(history.processing_time, 2) if history.processing_time else 0,
        video_url=video_url
    )


@app.route('/api/translate', methods=['POST'])
@login_required
def translate_text_endpoint():
    """Translate transcript or summary to selected language"""
    from main import translate_text
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('language', 'es')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text to translate'}), 400
        
        # Validate language code
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'hi', 'ar', 'ko', 'ta']
        if target_language not in valid_languages:
            return jsonify({'error': 'Invalid language code'}), 400
        
        print(f"📝 Translating to {target_language}...")
        translated = translate_text(text, target_language)
        
        return jsonify({
            'success': True,
            'translated_text': translated,
            'language': target_language
        }), 200
    
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500


@app.route('/api/generate-audio', methods=['POST'])
@login_required
def generate_audio_endpoint():
    """Generate audio from text in user's preferred language"""
    from main import text_to_speech, translate_text
    import uuid
    import os
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('language', 'en')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Validate language code
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'hi', 'ar', 'ko', 'ta']
        if target_language not in valid_languages:
            return jsonify({'error': 'Invalid language code'}), 400
        
        # If language is not English, translate first
        if target_language != 'en':
            print(f"🌐 Translating text to {target_language} for audio...")
            text = translate_text(text, target_language)
        
        # Generate unique filename for the audio
        audio_filename = f"audio_{uuid.uuid4().hex[:8]}_{target_language}.mp3"
        audio_dir = os.path.join('static', 'audio_cache')
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, audio_filename)
        
        print(f"🔊 Generating audio in {target_language}...")
        
        # Temporarily disable progress updates for this call
        import main
        original_update = main.update_progress
        original_check = main.check_cancel
        main.update_progress = lambda *args, **kwargs: None
        main.check_cancel = lambda: None
        
        try:
            text_to_speech(text, audio_path, language=target_language, speed_factor=1.15)
        finally:
            main.update_progress = original_update
            main.check_cancel = original_check
        
        # Return the audio file URL
        audio_url = f"/static/audio_cache/{audio_filename}"
        
        return jsonify({
            'success': True,
            'audio_url': audio_url,
            'language': target_language
        }), 200
    
    except Exception as e:
        print(f"❌ Audio generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Audio generation failed: {str(e)}'}), 500


@app.route('/settings', methods=['GET'])
@login_required
def settings():
    """Display advanced settings page"""
    return render_template('settings.html')


@app.route('/export/<format>', methods=['POST'])
@login_required
def export_results(format):
    """Export processing results in specified format - supports both original and translated content"""
    from main import export_to_pdf, export_to_docx, export_to_json, export_to_srt, translate_text
    import os
    
    try:
        # Get JSON data from request
        data = request.get_json() if request.is_json else {}
        
        # Get content and language
        transcript = data.get('transcript')
        summary = data.get('summary')
        export_language = data.get('language', 'en')  # Get language from request
        
        # Get latest processing history for metadata
        latest = ProcessingHistory.query.filter_by(user_id=current_user.id).order_by(
            ProcessingHistory.created_at.desc()
        ).first()
        
        # If no content provided, read from files
        if not transcript or not summary:
            if not latest:
                return jsonify({'error': 'No processing history found'}), 404
            
            # Read transcript and summary from files
            transcript = open(latest.transcript_path, 'r', encoding='utf-8').read() if os.path.exists(latest.transcript_path) else ""
            summary = open(latest.summary_path, 'r', encoding='utf-8').read() if os.path.exists(latest.summary_path) else ""
        
        # ===== TRANSLATE CONTENT IF NOT ENGLISH =====
        # Use LANGUAGE_NAMES imported from main.py
        language_name = LANGUAGE_NAMES.get(export_language, 'English')
        
        # Translate content if language is not English
        if export_language and export_language != 'en':
            print(f"📝 Translating content to {language_name} ({export_language}) for PDF export...")
            try:
                transcript = translate_text(transcript, export_language)
                summary = translate_text(summary, export_language)
                print(f"✅ Translation to {language_name} completed")
            except Exception as trans_error:
                print(f"⚠️ Translation failed: {trans_error}. Using original content.")
        
        # Get base name from latest history
        base_name = os.path.splitext(latest.video_filename)[0] if latest else "export"
        
        # Metadata - pass both language code and name for PDF generation
        metadata = {
            'duration': f"{latest.processing_time:.2f}s" if latest and latest.processing_time else "N/A",
            'language': export_language,  # Pass language code (ta, hi, etc.) for font selection
            'language_name': language_name,  # Pass display name for metadata
            'compression_ratio': latest.compression_ratio if latest else 0
        }
        
        # Create exports directory if not exists
        os.makedirs('static/exports', exist_ok=True)
        
        # Export based on format
        if format.lower() == 'pdf':
            file_path = export_to_pdf(base_name, transcript, summary, metadata)
        elif format.lower() == 'docx':
            file_path = export_to_docx(base_name, transcript, summary, metadata)
        elif format.lower() == 'json':
            file_path = export_to_json(base_name, transcript, summary, metadata)
        elif format.lower() == 'srt':
            file_path = export_to_srt(base_name, transcript, summary)
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        # Return the file
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/favorite/<int:history_id>', methods=['POST'])
@login_required
def toggle_favorite(history_id):
    """Toggle favorite status for a video"""
    try:
        history = ProcessingHistory.query.get(history_id)
        
        if not history or history.user_id != current_user.id:
            return jsonify({'error': 'Not found'}), 404
        
        history.is_favorite = not history.is_favorite
        db.session.commit()
        
        return jsonify({
            'success': True,
            'is_favorite': history.is_favorite,
            'message': f"{'Added to' if history.is_favorite else 'Removed from'} favorites"
        }), 200
    except Exception as e:
        print(f"❌ Favorite error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tags/<int:history_id>', methods=['POST'])
@login_required
def update_tags(history_id):
    """Update tags for a video"""
    try:
        data = request.get_json()
        tags = data.get('tags', '')
        
        history = ProcessingHistory.query.get(history_id)
        
        if not history or history.user_id != current_user.id:
            return jsonify({'error': 'Not found'}), 404
        
        # Sanitize and validate tags
        tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        tags_list = tags_list[:10]  # Max 10 tags
        history.tags = ','.join(tags_list)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'tags': tags_list,
            'message': 'Tags updated successfully'
        }), 200
    except Exception as e:
        print(f"❌ Tags error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
@login_required
def search_history():
    """Search processing history by tags or filename"""
    try:
        query = request.args.get('q', '').lower()
        tag = request.args.get('tag', '').lower()
        favorites_only = request.args.get('favorites', 'false').lower() == 'true'
        
        results = ProcessingHistory.query.filter_by(user_id=current_user.id)
        
        if favorites_only:
            results = results.filter_by(is_favorite=True)
        
        if tag:
            results = results.filter(ProcessingHistory.tags.ilike(f'%{tag}%'))
        elif query:
            results = results.filter(ProcessingHistory.video_filename.ilike(f'%{query}%'))
        
        results = results.order_by(ProcessingHistory.created_at.desc()).limit(20).all()
        
        return jsonify({
            'success': True,
            'results': [r.to_dict() for r in results]
        }), 200
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({'error': str(e)}), 500

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


# ✅ Progress route for live frontend updates (requires login)
@app.route('/progress', methods=['GET'])
@login_required
@limiter.exempt  # Exempt from rate limiting - polled frequently during processing
def get_progress():
    """Get progress status for current user's session"""
    session_id = session.get('processing_session_id')
    progress = get_progress_status(session_id)
    return jsonify(progress)


# ✅ Cancel route to abort processing (requires login + CSRF)
@app.route('/cancel', methods=['POST'])
@login_required
@limiter.exempt  # Exempt from rate limiting
def cancel_processing():
    """Cancel processing for current user's session only"""
    session_id = session.get('processing_session_id')
    if session_id:
        request_cancel(session_id)
        return jsonify({"status": "canceled", "session_id": session_id})
    return jsonify({"status": "no_active_session"}), 400


# ===== VIEW SUMMARY =====
@app.route('/api/view-summary/<int:history_id>', methods=['GET'])
@login_required
def view_summary(history_id):
    """View summary for a processing history record with optional translation"""
    from main import translate_text
    import os
    
    try:
        # Get optional language parameter
        language = request.args.get('lang', 'en')
        
        history = ProcessingHistory.query.filter_by(
            id=history_id,
            user_id=current_user.id
        ).first()
        
        if not history:
            return jsonify({'error': 'Record not found'}), 404
        
        # Read summary
        summary = ""
        if history.summary_path and os.path.exists(history.summary_path):
            with open(history.summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        
        # Translate if not English
        if language and language != 'en' and summary:
            try:
                print(f"📝 Translating summary to {language} for viewing...")
                summary = translate_text(summary, language)
            except Exception as trans_error:
                print(f"⚠️ Translation failed: {trans_error}")
        
        return jsonify({
            'success': True,
            'filename': history.video_filename,
            'summary': summary,
            'language': language
        }), 200
    except Exception as e:
        print(f"Error viewing summary: {e}")
        return jsonify({'error': str(e)}), 500


# ===== DOWNLOAD PDF SIMPLE (PDF ONLY) =====
@app.route('/api/download-pdf-simple/<int:history_id>', methods=['GET'])
@login_required
def download_pdf_simple(history_id):
    """Download PDF with all metadata for a processing history record"""
    from main import export_to_pdf
    import os
    
    try:
        history = ProcessingHistory.query.filter_by(
            id=history_id,
            user_id=current_user.id
        ).first()
        
        if not history:
            return jsonify({'error': 'Record not found'}), 404
        
        # Read transcript and summary
        transcript = ""
        summary = ""
        
        if history.transcript_path and os.path.exists(history.transcript_path):
            with open(history.transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        
        if history.summary_path and os.path.exists(history.summary_path):
            with open(history.summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        
        # Get base name
        base_name = os.path.splitext(history.video_filename)[0]
        
        # Get language code and name (using LANGUAGE_NAMES imported from main.py)
        language_code = history.language_detected.lower()[:2] if history.language_detected else 'en'
        language_name = LANGUAGE_NAMES.get(language_code, history.language_detected or 'English')

        # Metadata
        metadata = {
            'duration': f"{history.processing_time:.2f}s" if history.processing_time else "N/A",
            'language': language_code,
            'language_name': language_name,
            'compression_ratio': history.compression_ratio or 0,
            'transcript_words': history.transcript_length or 0,
            'summary_words': history.summary_length or 0,
            'created_at': history.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create exports directory if not exists
        os.makedirs('static/exports', exist_ok=True)
        
        # Export to PDF
        file_path = export_to_pdf(base_name, transcript, summary, metadata)
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        print(f"PDF download error: {e}")
        return jsonify({'error': f'Error downloading PDF: {str(e)}'}), 500


# ===== DELETE HISTORY RECORD =====
@app.route('/api/delete-history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history_record(history_id):
    """Delete a processing history record"""
    import os
    
    try:
        history = ProcessingHistory.query.filter_by(
            id=history_id, 
            user_id=current_user.id
        ).first()
        
        if not history:
            return jsonify({'error': 'Record not found'}), 404
        
        # Delete associated files
        files_to_delete = [
            history.transcript_path,
            history.summary_path,
            history.audio_path,
            history.summary_audio_path
        ]
        
        for filepath in files_to_delete:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Warning: Could not delete {filepath}: {e}")
        
        # Delete database record
        db.session.delete(history)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Record deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting history: {e}")
        return jsonify({'error': f'Error deleting record: {str(e)}'}), 500


# ===== DOWNLOAD PDF IN SPECIFIC LANGUAGE =====
@app.route('/api/download-pdf/<int:history_id>/<language>', methods=['GET'])
@login_required
def download_pdf_language(history_id, language):
    """Download PDF in a specific language"""
    from main import export_to_pdf, translate_text
    import os
    
    try:
        history = ProcessingHistory.query.filter_by(
            id=history_id,
            user_id=current_user.id
        ).first()
        
        if not history:
            return jsonify({'error': 'Record not found'}), 404
        
        # Read transcript and summary
        transcript = open(history.transcript_path, 'r', encoding='utf-8').read() if os.path.exists(history.transcript_path) else ""
        summary = open(history.summary_path, 'r', encoding='utf-8').read() if os.path.exists(history.summary_path) else ""
        
        # Translate if needed using the same function as result page
        if language != 'en':
            try:
                print(f"📝 Translating content to {language} for history PDF download...")
                transcript = translate_text(transcript, language) if transcript else ""
                summary = translate_text(summary, language) if summary else ""
            except Exception as e:
                print(f"Translation error: {e}")
        
        # Get base name and metadata (using LANGUAGE_NAMES imported from main.py)
        base_name = os.path.splitext(history.video_filename)[0]
        metadata = {
            'duration': f"{history.processing_time:.2f}s" if history.processing_time else "N/A",
            'language': language,
            'language_name': LANGUAGE_NAMES.get(language, 'English'),
            'compression_ratio': history.compression_ratio or 0
        }
        
        # Export to PDF
        os.makedirs('static/exports', exist_ok=True)
        file_path = export_to_pdf(base_name, transcript, summary, metadata)
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        print(f"PDF download error: {e}")
        return jsonify({'error': f'Error downloading PDF: {str(e)}'}), 500


# ===== FILE CLEANUP FUNCTIONALITY =====
def cleanup_old_files(max_age_days=7):
    """
    Clean up old files from videos, results, audio_cache, and exports directories.
    Files older than max_age_days will be deleted.
    
    Returns:
        dict with cleanup statistics
    """
    import shutil
    from datetime import datetime, timedelta
    
    cleanup_dirs = [
        'static/audio_cache', 
        'static/exports'
    ]
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    stats = {'files_deleted': 0, 'dirs_deleted': 0, 'space_freed_mb': 0, 'errors': []}
    
    for directory in cleanup_dirs:
        if not os.path.exists(directory):
            continue
            
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                # Get file/folder modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                if mtime < cutoff_time:
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path) / (1024 * 1024)
                        os.remove(item_path)
                        stats['files_deleted'] += 1
                        stats['space_freed_mb'] += size
                    elif os.path.isdir(item_path):
                        size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, _, filenames in os.walk(item_path)
                            for filename in filenames
                        ) / (1024 * 1024)
                        shutil.rmtree(item_path)
                        stats['dirs_deleted'] += 1
                        stats['space_freed_mb'] += size
            except Exception as e:
                stats['errors'].append(f"{item_path}: {str(e)}")
    
    # Also clean up orphaned result directories (no matching history record)
    results_dir = 'static/results'
    if os.path.exists(results_dir):
        # Get all video filenames from history
        try:
            with app.app_context():
                history_filenames = {
                    os.path.splitext(h.video_filename)[0] 
                    for h in ProcessingHistory.query.all()
                }
                
                for item in os.listdir(results_dir):
                    item_path = os.path.join(results_dir, item)
                    if os.path.isdir(item_path) and item not in history_filenames:
                        mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                        if mtime < cutoff_time:
                            try:
                                size = sum(
                                    os.path.getsize(os.path.join(dirpath, filename))
                                    for dirpath, _, filenames in os.walk(item_path)
                                    for filename in filenames
                                ) / (1024 * 1024)
                                shutil.rmtree(item_path)
                                stats['dirs_deleted'] += 1
                                stats['space_freed_mb'] += size
                            except Exception as e:
                                stats['errors'].append(f"{item_path}: {str(e)}")
        except Exception as e:
            stats['errors'].append(f"History check failed: {str(e)}")
    
    stats['space_freed_mb'] = round(stats['space_freed_mb'], 2)
    return stats


@app.route('/api/admin/cleanup', methods=['POST'])
@login_required
@limiter.limit("1 per hour")
def admin_cleanup():
    """
    Admin endpoint to trigger file cleanup.
    Rate limited to 1 request per hour.
    """
    try:
        data = request.get_json() or {}
        max_age_days = data.get('max_age_days', 7)
        
        # Validate input
        if not isinstance(max_age_days, int) or max_age_days < 1:
            max_age_days = 7
        
        stats = cleanup_old_files(max_age_days=max_age_days)
        
        return jsonify({
            'success': True,
            'message': f"Cleanup completed (files older than {max_age_days} days)",
            'stats': stats
        }), 200
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


@app.route('/api/admin/storage-stats', methods=['GET'])
@login_required
def storage_stats():
    """Get storage usage statistics"""
    import glob
    
    def get_dir_size(path):
        total = 0
        if os.path.exists(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 2)  # MB
    
    try:
        stats = {
            'results_mb': get_dir_size('static/results'),
            'audio_cache_mb': get_dir_size('static/audio_cache'),
            'exports_mb': get_dir_size('static/exports'),
            'total_history_records': ProcessingHistory.query.count()
        }
        stats['total_mb'] = round(sum([
            stats['results_mb'], 
            stats['audio_cache_mb'], 
            stats['exports_mb']
        ]), 2)
        
        return jsonify({'success': True, 'stats': stats}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== CSRF EXEMPTIONS FOR API ENDPOINTS =====
# Exempt some endpoints from CSRF for API compatibility
@app.before_request
def csrf_exempt_api():
    """Exempt progress endpoint from CSRF (GET request, read-only)"""
    pass  # CSRF is automatically not required for GET requests


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
