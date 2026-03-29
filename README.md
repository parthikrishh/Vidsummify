# Vidsummify

Vidsummify is a production-style Flask application that transforms uploaded videos into structured textual outputs and downloadable artifacts.

The pipeline includes audio extraction, speech-to-text transcription, abstractive summarization, translation, text-to-speech generation, and report export.

## Overview

Vidsummify provides an end-to-end workflow for video understanding:

1. Upload an MP4 video.
2. Extract audio and transcribe speech with Faster-Whisper.
3. Generate concise summaries using Transformer models.
4. Translate summaries into supported languages.
5. Generate audio narration from translated summary text.
6. Export output as PDF and maintain searchable history.

## Core Capabilities

- Secure authentication: signup, login, logout, password reset, profile management
- Long-form video processing with status tracking and cancellation
- AI transcription with Faster-Whisper
- Transformer-based summarization
- Multi-language translation and TTS audio generation
- Searchable processing history with favorite and tag support
- Report export and multilingual font handling
- Admin cleanup and storage statistics endpoints

## Technology Stack

- Backend: Flask, Flask-Login, Flask-SQLAlchemy, Flask-Migrate
- ML and NLP: Faster-Whisper, Transformers, PyTorch, NLTK
- Media: FFmpeg, MoviePy, PyDub, gTTS
- Data: SQLite (via SQLAlchemy)
- Export: ReportLab, python-docx, Pillow
- Utility: python-dotenv, langdetect, googletrans

## Project Structure

```text
VideoTextProject/
|-- app.py
|-- auth.py
|-- main.py
|-- models.py
|-- init_db.py
|-- requirements.txt
|-- .env.example
|-- templates/
|-- static/
|   |-- fonts/
|   |-- exports/
|   |-- audio_cache/
|   |-- profile_photos/
|   \-- results/      (generated at runtime, ignored in git)
\-- instance/
```

## Prerequisites

- Python 3.10+
- FFmpeg installed and available in PATH
- Windows, Linux, or macOS

## Setup

### 1) Clone repository

```bash
git clone https://github.com/parthikrishh/Vidsummify.git
cd Vidsummify
```

### 2) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

```bash
copy .env.example .env
```

Set at least the following in .env:

- SECRET_KEY
- FLASK_DEBUG
- FLASK_HOST
- PORT

### 5) Initialize database

```bash
python init_db.py
```

### 6) Run application

```bash
python app.py
```

Default local URL:

- http://127.0.0.1:5000

## Configuration

Environment-driven settings include:

- SECRET_KEY: required for session security
- FLASK_DEBUG: development mode toggle
- FLASK_HOST and PORT: network bind configuration
- MAX_FILE_SIZE: upload size cap (default example: 500 MB)

Optional model-related parameters are provided in .env.example for whisper and summarization tuning.

## HTTP Endpoints (High-Level)

UI and workflow endpoints:

- GET /
- POST /upload
- GET /result and /result/<history_id>
- GET /history
- GET /settings

Processing and content APIs:

- POST /api/translate
- POST /api/generate-audio
- GET /api/download-pdf-simple/<history_id>
- GET /api/download-pdf/<history_id>/<language>
- DELETE /api/delete-history/<history_id>

History and metadata APIs:

- POST /api/favorite/<history_id>
- POST /api/tags/<history_id>
- GET /api/search
- GET /api/statistics

Admin APIs:

- POST /api/admin/cleanup
- GET /api/admin/storage-stats

Authentication routes are exposed via blueprint in auth.py (signup/login/logout/reset/profile and profile APIs).

## Supported Languages

The application includes multilingual translation and export support for:

- English, Spanish, French, German, Italian, Portuguese
- Russian, Japanese, Chinese, Korean
- Hindi, Arabic, Tamil

Unicode font assets are provided in static/fonts for multilingual PDF output.

## Operational Notes

- Generated runtime media (for example static/results) is intentionally excluded from source control.
- Keep large video/audio artifacts out of git history; use external object storage or Git LFS if permanent versioning is required.
- For production deployment, run behind a WSGI server and disable debug mode.

## Troubleshooting

- Video playback issues:
    - Verify browser MP4 support.
    - Ensure FFmpeg is installed and accessible in PATH.
    - Confirm generated media exists under static/results.
- App fails at startup with SECRET_KEY error:
    - Add a valid SECRET_KEY in .env.
- Slow first run:
    - Model downloads and initialization can be heavy on first execution.

## Author

- Name: Parthiban K B
- GitHub: https://github.com/parthikrishh

## License

This project is for academic and learning purposes.


