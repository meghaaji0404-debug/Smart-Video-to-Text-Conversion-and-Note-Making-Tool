import json
import os
import urllib.request

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = None
INDEX = None
DOCUMENTS = []


def load_embedder():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(transcript: str):
    global INDEX, DOCUMENTS
    load_embedder()
    chunks = [chunk.strip() for chunk in transcript.split(".") if chunk.strip()]
    if not chunks:
        return None

    DOCUMENTS = chunks
    embeddings = EMBED_MODEL.encode(chunks, convert_to_numpy=True)
    INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    INDEX.add(embeddings.astype("float32"))
    return INDEX


def answer_question(question: str, transcript: str) -> str:
    if not transcript:
        return "No transcript available."

    build_index(transcript)
    load_embedder()
    query_vector = EMBED_MODEL.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = INDEX.search(query_vector, 1)
    best_chunk = DOCUMENTS[int(indices[0][0])]

    # Fallback to a simple local answer style if Ollama is unavailable.
    if os.getenv("OLLAMA_HOST"):
        try:
            payload = json.dumps({"model": "llama3", "prompt": f"Question: {question}\nContext: {best_chunk}\nAnswer:"}).encode()
            req = urllib.request.Request(
                f"{os.getenv('OLLAMA_HOST')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("response", f"Based on the transcript: {best_chunk}")
        except Exception:
            pass

    return f"Based on the transcript, the most relevant context is: {best_chunk}"
