"""Database initialization script"""
import os
import sys
from app import app, db
from models import User, ProcessingHistory, LanguageSupport

def init_db():
    """Initialize the database"""
    print("🔄 Initializing database...")
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Check if language support data exists
        if LanguageSupport.query.first() is None:
            print("📝 Adding language support data...")
            languages = LanguageSupport.get_supported_languages()
            for code, name in languages.items():
                lang = LanguageSupport(
                    language_code=code,
                    language_name=name,
                    whisper_model='tiny',
                    is_supported=True
                )
                db.session.add(lang)
            db.session.commit()
            print("✅ Language support data added!")
        
        print("✅ Database initialization complete!")
        print(f"📊 Database location: {os.path.abspath('vidsummify.db')}")

if __name__ == '__main__':
    init_db()
