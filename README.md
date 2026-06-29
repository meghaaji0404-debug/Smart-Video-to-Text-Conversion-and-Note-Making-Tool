# Smart Video to Text Conversion and Note-Making Tool

This project provides a Flask-based AI workflow for converting uploaded videos or YouTube links into:

- transcripts
- summaries
- keyword lists
- question-answer notes
- a lightweight RAG-style Q&A experience

## Folder Structure

- app.py: Flask entry point
- requirements.txt: Python dependencies
- modules/: processing logic for transcription, summarization, and RAG
- templates/: HTML frontend
- static/: CSS assets
- data/uploads/: uploaded media files
- data/processed/: generated audio and outputs

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000 in your browser.

## Notes

- For best transcription quality, install FFmpeg and ensure it is available on your PATH.
- The app uses a local Ollama endpoint if available for context-aware answers.
