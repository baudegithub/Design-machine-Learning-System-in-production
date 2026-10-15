from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import numpy as np
import tensorflow as tf

from .utils import set_experiment_dirs


class PredictRequest(BaseModel):
    title: str
    top_k: Optional[int] = 3


app = FastAPI(title="News Title Classifier - Demo")

# Globals loaded at startup
MODEL = None
PREPROCESSOR = None
LABELS = None


@app.on_event("startup")
async def load_model_assets():
    global MODEL, PREPROCESSOR, LABELS

    base_dir = os.getenv("BASE_DIR", ".")
    data_dir, model_dir, vocab_dir = set_experiment_dirs(base_dir)

    model_path = model_dir
    print(f"Loading model from {model_path}")
    try:
        MODEL = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # Load labels
    labels_path = os.path.join(vocab_dir, "labels.txt") if os.path.exists(os.path.join(vocab_dir, "labels.txt")) else os.path.join(base_dir, "labels.txt")
    if not os.path.exists(labels_path):
        raise RuntimeError(f"labels.txt not found in {vocab_dir} or {base_dir}")
    LABELS = [l.strip() for l in open(labels_path, "r").read().splitlines()]

    # Load TextVectorization assets
    PREPROCESSOR = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=20)
    assets_dir = vocab_dir if os.path.isdir(vocab_dir) else base_dir
    PREPROCESSOR.load_assets(assets_dir)

    print("Model and assets loaded successfully.")


@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
async def predict(req: PredictRequest):
    if MODEL is None or PREPROCESSOR is None or LABELS is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded")

    title = req.title
    tokens = PREPROCESSOR(tf.constant([title]))
    probs = MODEL.predict(tokens)
    probs = np.asarray(probs)[0]

    top_k = min(req.top_k, len(probs))
    top_idx = probs.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append({
            "label": LABELS[idx] if idx < len(LABELS) else str(idx),
            "probability": float(probs[idx])
        })

    return {"title": title, "predictions": results}
