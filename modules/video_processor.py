import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests
import whisper
from keybert import KeyBERT


MODEL_NAME = "base"
TRANSCRIPT_MODEL = None
SUMMARIZER = None
KEYWORD_MODEL = None


def load_models():
    global TRANSCRIPT_MODEL, KEYWORD_MODEL
    if TRANSCRIPT_MODEL is None:
        TRANSCRIPT_MODEL = whisper.load_model(MODEL_NAME)
    if KEYWORD_MODEL is None:
        KEYWORD_MODEL = KeyBERT("all-MiniLM-L6-v2")


def extract_audio(input_path: str, output_path: str) -> str:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", output_path]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def transcribe_audio(audio_path: str) -> str:
    load_models()
    result = TRANSCRIPT_MODEL.transcribe(audio_path, fp16=False)
    return result.get("text", "")


def summarize_text(text: str) -> str:
    if not text.strip():
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return ""

    if len(sentences) <= 3:
        return " ".join(sentences)

    summary_sentences = sentences[:3]
    summary = " ".join(summary_sentences)
    if len(summary) > 220:
        summary = summary[:220].rsplit(" ", 1)[0] + "..."
    return summary


def extract_keywords(text: str, top_n: int = 10) -> list:
    load_models()
    keywords = KEYWORD_MODEL.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [word for word, score in keywords]


def process_media(input_path: str | None, youtube_url: str | None) -> dict:
    load_models()

    if youtube_url:
        temp_dir = tempfile.mkdtemp(dir="data/processed")
        video_path = os.path.join(temp_dir, "video.mp4")
        subprocess.run(["yt-dlp", "-o", video_path, youtube_url], check=True, capture_output=True, text=True)
        input_path = video_path

    if not input_path:
        raise ValueError("No media path provided")

    output_audio = Path(input_path).with_suffix(".wav")
    extract_audio(input_path, str(output_audio))
    transcript = transcribe_audio(str(output_audio))
    summary = summarize_text(transcript)
    keywords = extract_keywords(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "keywords": keywords,
        "source": input_path,
    }
