# Vidsummify 🎬📝  

Video-to-Text Summarization using Machine Learning

## 📌 Project Overview

Vidsummify is a Flask-based web application that converts video content into meaningful text summaries.
The system extracts audio from videos, transcribes speech into text using a deep learning model, summarizes
the text using NLP techniques, and generates audio from the summary.

---


## 🚀 Features

- **User Authentication** – Secure login/signup system with profile management
- **Video Upload** – Support for MP4 video files up to 500MB
- **Speech-to-Text** – Transcription using Faster-Whisper AI model
- **Text Summarization** – Using Transformer models (DistilBART)
- **Multi-language Translation** – Support for 13 languages
- **Audio Generation** – Text-to-speech in selected language
- **PDF Export** – Professional PDF reports with proper Unicode fonts
- **Processing History** – Track all processed videos
- **Favorites & Tags** – Organize your videos
- **Dark/Light Mode** – Theme toggle support
- **Video Playback on Result Page** – Uploaded videos are now viewable directly on the result page with a built-in video player.

---

## 🆕 Recent Updates

- **Video Player on Result Page:** After upload and processing, the original video is now accessible and playable on the result page.
- **Improved Static File Handling:** Uploaded videos are copied to a static-accessible directory for reliable playback.
- **Bug Fixes & Code Cleanup:** General improvements for stability and maintainability.

---

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, Flask, SQLAlchemy
- **Machine Learning:** Faster-Whisper, Transformers (HuggingFace)
- **NLP:** DistilBART for summarization
- **Audio Processing:** FFmpeg, MoviePy, PyDub, gTTS
- **Translation:** Google Translate API
- **PDF Generation:** ReportLab with Noto Sans fonts
- **Frontend:** HTML5, CSS3, JavaScript, Font Awesome

---

## 📂 Project Structure

```
VideoTextProject/
├── app.py              # Main Flask application
├── auth.py             # Authentication routes
├── main.py             # Core processing functions
├── models.py           # Database models
├── config.py           # Configuration settings
├── init_db.py          # Database initialization
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
├── .env.example        # Example environment file
│
├── templates/          # HTML templates
│   ├── index.html      # Home/upload page
│   ├── result.html     # Processing results
│   ├── history.html    # Processing history
│   ├── profile.html    # User profile
│   ├── login.html      # Login page
│   ├── signup.html     # Registration page
│   ├── settings.html   # User settings
│   └── error.html      # Error page
│
├── static/
│   ├── fonts/          # Unicode fonts (Noto Sans)
│   ├── results/        # Transcripts & summaries
│   ├── exports/        # PDF exports
│   ├── audio_cache/    # Generated audio
│   └── profile_photos/ # User avatars
│
└── instance/
    └── vidsummify.db   # SQLite database
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/VideoTextProject.git
cd VideoTextProject
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg
```bash
# Windows (using winget)
winget install Gyan.FFmpeg

# Or download from https://ffmpeg.org/download.html
```

### 5. Setup environment
```bash
copy .env.example .env
# Edit .env with your settings
```

### 6. Initialize database
```bash
python init_db.py
```


### 7. Run the application
```bash
python app.py
```

### 8. Open in browser
```
http://localhost:5000
```

---

## ❓ Troubleshooting

- **Video not playing on result page?**
    - Ensure your browser supports MP4 playback.
    - Confirm that FFmpeg is installed and available in your system PATH.
    - Uploaded videos are copied to `static/results/` and served from there. If you encounter a 404 or playback issue, check file permissions and Flask static file settings.

---

---

## 🌍 Supported Languages

| Language | Code | Font |
|----------|------|------|
| English | en | NotoSans |
| Spanish | es | NotoSans |
| French | fr | NotoSans |
| German | de | NotoSans |
| Italian | it | NotoSans |
| Portuguese | pt | NotoSans |
| Russian | ru | NotoSans |
| Japanese | ja | NotoSansJP |
| Chinese | zh | NotoSansSC |
| Korean | ko | NotoSansKR |
| Hindi | hi | NotoSansDevanagari |
| Arabic | ar | NotoSansArabic |
| Tamil | ta | NotoSansTamil |

---

## 📝 License

This project is developed as a Final Year Engineering Project.

---

## 👨‍💻 Author

**Parthiban K B**  
GitHub: https://github.com/parthikrishh

Developed with ❤️ for educational purposes.

---

## 📄 License

This project is for academic and learning purposes.


