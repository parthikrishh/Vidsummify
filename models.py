from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    profile_photo = db.Column(db.String(255), nullable=True)  # Path to profile photo
    password_reset_token = db.Column(db.String(100), nullable=True)  # Token for password reset
    password_reset_expires = db.Column(db.DateTime, nullable=True)  # Token expiration time
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship with processing history
    processing_history = db.relationship('ProcessingHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def get_stats(self):
        """Get user statistics"""
        history = self.processing_history.all()
        return {
            'total_videos': len(history),
            'total_processing_time': sum(h.processing_time or 0 for h in history),
            'avg_compression_ratio': sum(h.compression_ratio or 0 for h in history) / len(history) if history else 0,
            'preferred_language': self._get_preferred_language()
        }
    
    def _get_preferred_language(self):
        """Get most used language"""
        history = self.processing_history.all()
        if not history:
            return 'English'
        languages = [h.language for h in history if h.language]
        return max(set(languages), key=languages.count) if languages else 'English'
    
    def __repr__(self):
        return f'<User {self.username}>'


class ProcessingHistory(db.Model):
    """Model to track video processing history"""
    __tablename__ = 'processing_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    video_filename = db.Column(db.String(255), nullable=False)
    original_video_size = db.Column(db.Float)  # in MB
    language_detected = db.Column(db.String(50), default='English')
    language = db.Column(db.String(50), default='English')  # User selected language
    
    # Processing files
    audio_path = db.Column(db.String(500))
    transcript_path = db.Column(db.String(500))
    summary_path = db.Column(db.String(500))
    summary_audio_path = db.Column(db.String(500))
    
    # Content metrics
    transcript_length = db.Column(db.Integer)  # number of words
    summary_length = db.Column(db.Integer)  # number of words
    compression_ratio = db.Column(db.Float)  # summary_length / transcript_length
    
    # Processing metrics
    processing_time = db.Column(db.Float)  # in seconds
    extraction_time = db.Column(db.Float)
    transcription_time = db.Column(db.Float)
    summarization_time = db.Column(db.Float)
    tts_time = db.Column(db.Float)
    
    # Metadata
    status = db.Column(db.String(50), default='completed')  # completed, failed, pending
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # User preferences
    is_favorite = db.Column(db.Boolean, default=False)  # Bookmark/favorite toggle
    tags = db.Column(db.String(500), default='')  # Comma-separated tags for categorization
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'video_filename': self.video_filename,
            'language': self.language,
            'language_detected': self.language_detected,
            'transcript_length': self.transcript_length,
            'summary_length': self.summary_length,
            'compression_ratio': round(self.compression_ratio, 2) if self.compression_ratio else 0,
            'processing_time': round(self.processing_time, 2) if self.processing_time else 0,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_path': self.summary_path,
            'transcript_path': self.transcript_path,
            'summary_audio_path': self.summary_audio_path
        }
    
    def __repr__(self):
        return f'<ProcessingHistory {self.video_filename} - {self.created_at}>'


class LanguageSupport(db.Model):
    """Model to store supported languages and their configurations"""
    __tablename__ = 'language_support'
    
    id = db.Column(db.Integer, primary_key=True)
    language_code = db.Column(db.String(10), unique=True, nullable=False)
    language_name = db.Column(db.String(50), nullable=False)
    whisper_model = db.Column(db.String(50))  # e.g., "tiny", "base", "small"
    is_supported = db.Column(db.Boolean, default=True)
    
    @staticmethod
    def get_supported_languages():
        """Get list of all supported languages"""
        return {
            'en': 'English',
            'es': 'Español (Spanish)',
            'fr': 'Français (French)',
            'de': 'Deutsch (German)',
            'it': 'Italiano (Italian)',
            'pt': 'Português (Portuguese)',
            'ru': 'Русский (Russian)',
            'ja': '日本語 (Japanese)',
            'zh': '中文 (Chinese)',
            'hi': 'हिन्दी (Hindi)',
            'ar': 'العربية (Arabic)',
            'ko': '한국어 (Korean)',
            'ta': 'தமிழ் (Tamil)',
        }
    
    def __repr__(self):
        return f'<LanguageSupport {self.language_name}>'
