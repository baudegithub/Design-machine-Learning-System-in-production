"""
Simple prediction script for single titles.

Usage:
  python -m src.predict "Some news title here" --base-dir . --model-dir models/saved_model

If no title is provided as an argument, the script enters interactive mode.
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from .utils import set_experiment_dirs


def predict_title(title: str, base_dir: str, model_dir: str):
    data_dir, default_model_dir, vocab_dir = set_experiment_dirs(base_dir)
    model_path = model_dir if model_dir else default_model_dir

    # Load model
    model = tf.keras.models.load_model(model_path)

    labels_path = os.path.join(vocab_dir, 'labels.txt') if os.path.exists(os.path.join(vocab_dir, 'labels.txt')) else os.path.join(base_dir, 'labels.txt')
    labels = [l.strip() for l in open(labels_path, 'r').read().splitlines()]

    # Load preprocessor
    title_preprocessor = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=20)
    assets_dir = vocab_dir if os.path.isdir(vocab_dir) else base_dir
    title_preprocessor.load_assets(assets_dir)

    tokens = title_preprocessor(tf.constant([title]))
    probs = model.predict(tokens)
    probs = np.asarray(probs)[0]
    top_idx = probs.argmax()
    print(f"Title: {title}")
    print(f"Predicted label: {labels[top_idx]} (p={probs[top_idx]:.3f})")
    print("Top 3:")
    for idx in probs.argsort()[-3:][::-1]:
        print(f"  {labels[idx]}: {probs[idx]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('title', nargs='?', help='Title to classify. If omitted, interactive mode.')
    parser.add_argument('--base-dir', default='.', help='Repo root or experiment base dir')
    parser.add_argument('--model-dir', default='models/saved_model', help='Saved model directory to load')
    args = parser.parse_args()

    if args.title:
        predict_title(args.title, args.base_dir, args.model_dir)
    else:
        print('Enter titles to classify (empty line to exit):')
        while True:
            s = input('> ').strip()
            if not s:
                break
            predict_title(s, args.base_dir, args.model_dir)


if __name__ == '__main__':
    main()
