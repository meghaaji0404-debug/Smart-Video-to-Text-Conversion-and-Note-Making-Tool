import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from modules.video_processor import process_media
from modules.rag_pipeline import answer_question

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

UPLOAD_FOLDER = Path("data/uploads")
PROCESSED_FOLDER = Path("data/processed")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    uploaded_file = request.files.get("video_file")
    youtube_url = request.form.get("youtube_url", "").strip()

    if uploaded_file and uploaded_file.filename:
        upload_path = UPLOAD_FOLDER / uploaded_file.filename
        uploaded_file.save(upload_path)
        result = process_media(input_path=str(upload_path), youtube_url=None)
        return render_template("index.html", result=result)

    if youtube_url:
        result = process_media(input_path=None, youtube_url=youtube_url)
        return render_template("index.html", result=result)

    flash("Please upload a video file or provide a YouTube URL.")
    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.form.get("question") or "").strip()
    transcript = (request.form.get("transcript") or "").strip()

    if not question:
        flash("Please enter a question.")
        return redirect(url_for("index"))

    if not transcript:
        flash("No transcript is available yet. Process a video first.")
        return redirect(url_for("index"))

    answer = answer_question(question, transcript)
    return render_template("index.html", answer=answer, question=question, transcript=transcript)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
