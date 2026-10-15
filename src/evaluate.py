"""
Evaluate a saved model on the test split and optionally log results to experiment-tracker.csv

Usage:
  python -m src.evaluate --base-dir . --model-dir models/saved_model
"""

import argparse
import os
import csv
from datetime import datetime
import uuid

import pandas as pd
import tensorflow as tf

from .utils import set_experiment_dirs, df_to_tfdata


def evaluate(base_dir: str, model_dir: str, log_csv: str = 'experiment-tracker.csv'):
    data_dir, default_model_dir, vocab_dir = set_experiment_dirs(base_dir)

    # prefer explicit model_dir if provided
    model_path = model_dir if model_dir else default_model_dir

    test_path = os.path.join(data_dir, 'test_data.csv') if os.path.exists(os.path.join(data_dir, 'test_data.csv')) else os.path.join(base_dir, 'test_data.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test_data.csv not found under {data_dir} or {base_dir}")

    test_df = pd.read_csv(test_path)

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    labels_path = os.path.join(vocab_dir, 'labels.txt') if os.path.exists(os.path.join(vocab_dir, 'labels.txt')) else os.path.join(base_dir, 'labels.txt')
    topic_lookup = tf.keras.layers.StringLookup(vocabulary=labels_path, num_oov_indices=0)

    # load text vectorization
    title_preprocessor = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=20)
    assets_dir = vocab_dir if os.path.isdir(vocab_dir) else base_dir
    title_preprocessor.load_assets(assets_dir)

    test_ds = df_to_tfdata(test_df, topic_lookup, title_preprocessor)

    print('Evaluating...')
    results = model.evaluate(test_ds, verbose=1)

    # results can be [loss, accuracy]
    loss = results[0] if isinstance(results, (list, tuple)) else results
    acc = results[1] if isinstance(results, (list, tuple)) and len(results) > 1 else None

    print(f"loss={loss} acc={acc}")

    # Log
    row = {
        'run_id': uuid.uuid4().hex[:8],
        'timestamp': datetime.utcnow().isoformat(),
        'model_path': model_path,
        'loss': float(loss),
        'accuracy': float(acc) if acc is not None else '',
    }

    write_header = not os.path.exists(log_csv)
    with open(log_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged evaluation to {log_csv}")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='.', help='Repo root or experiment base dir')
    parser.add_argument('--model-dir', default='models/saved_model', help='Saved model directory to load')
    parser.add_argument('--log-csv', default='experiment-tracker.csv', help='CSV file to append metrics to')
    args = parser.parse_args(argv)

    evaluate(args.base_dir, args.model_dir, args.log_csv)


if __name__ == '__main__':
    main()
