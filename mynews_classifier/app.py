import argparse
import os

import tensorflow as tf
from flask import Flask, jsonify, request

from mynews_classifier.utils import set_experiment_dirs

app = Flask(__name__)

BASE_DIR = os.environ.get("MYNEWS_BASE_DIR", ".")
_, model_dir, vocab_dir = set_experiment_dirs(BASE_DIR)

model = tf.keras.models.load_model(model_dir)
labels_path = os.path.join(vocab_dir, "labels.txt")
topic_lookup = tf.keras.layers.StringLookup(vocabulary=labels_path, num_oov_indices=0, mask_token=None)

MAX_LENGTH = 20
VOCAB_SIZE = 10000
preprocessor = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH)
preprocessor.load_assets(vocab_dir)


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    title = payload.get("title")
    if not title or not isinstance(title, str):
        return jsonify({"error": "Please send JSON with a string field called 'title'."}), 400

    formatted = tf.constant([title])
    x = preprocessor(formatted)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(tf.argmax(probs).numpy())
    label = topic_lookup.get_vocabulary()[pred_idx]
    return jsonify({
        "title": title,
        "predicted_label": label,
        "confidence": float(probs[pred_idx]),
        "scores": [float(v) for v in probs],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local Flask app for MyNewsClassifier")
    parser.add_argument("--base-dir", default=".", help="Base experiment directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    _, model_dir, vocab_dir = set_experiment_dirs(BASE_DIR)
    model = tf.keras.models.load_model(model_dir)
    topic_lookup = tf.keras.layers.StringLookup(
        vocabulary=os.path.join(vocab_dir, "labels.txt"),
        num_oov_indices=0,
        mask_token=None,
    )
    preprocessor = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH)
    preprocessor.load_assets(vocab_dir)

    app.run(host=args.host, port=args.port)
