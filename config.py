"""
Configuration file for VidSummify application.
Uses environment variables with sensible defaults.
"""
import os

class Config:
    """Base configuration class."""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # File upload settings
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 500 * 1024 * 1024))  # 500MB default
    ALLOWED_EXTENSIONS = {'mp4'}
    
    # Directory settings
    VIDEOS_DIR = os.environ.get('VIDEOS_DIR', 'static/videos')
    RESULTS_DIR = os.environ.get('RESULTS_DIR', 'static/results')
    
    # Model settings (optimized for speed)
    WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL_SIZE', 'tiny')  # tiny = fastest, small = balanced, base = accurate
    WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'auto')  # auto-detect
    WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'auto')  # auto-detect
    
    # Summarization settings (optimized for speed)
    SUMMARY_MODEL = os.environ.get('SUMMARY_MODEL', 'sshleifer/distilbart-cnn-12-6')
    SUMMARY_MAX_LENGTH = int(os.environ.get('SUMMARY_MAX_LENGTH', 100))  # Reduced for speed
    SUMMARY_MIN_LENGTH = int(os.environ.get('SUMMARY_MIN_LENGTH', 30))  # Reduced for speed


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production!
    if not SECRET_KEY or SECRET_KEY == 'dev-secret-key-change-in-production':
        raise ValueError("SECRET_KEY must be set in production environment!")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
