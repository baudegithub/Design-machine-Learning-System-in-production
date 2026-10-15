"""
Lightweight experiment runner converted from the notebook.

This version is compatible with both of these layouts:
- E1/data, E1/model, E1/vocab
- repo root files: train_data.csv, test_data.csv, saved_model.pb, vocabulary.txt, labels.txt

Usage:
    python -m src.experiment --base-dir .
    python -m src.experiment --base-dir ./E1
"""

import argparse
import os
import sys

import pandas as pd
import tensorflow as tf

from .utils import set_experiment_dirs, df_to_tfdata


def evaluate_baseline(base_dir: str):
    data_dir, model_dir, vocab_dir = set_experiment_dirs(base_dir)

    print(f"Using experiment base directory: {base_dir}")
    print(f"  data:  {data_dir}")
    print(f"  model: {model_dir}")
    print(f"  vocab: {vocab_dir}\n")

    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv")) if os.path.exists(os.path.join(data_dir, "train_data.csv")) else None
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv")) if os.path.exists(os.path.join(data_dir, "test_data.csv")) else None

    if test_df is None:
        raise FileNotFoundError(f"Could not find test_data.csv under {data_dir}. Expected a data folder or root-level CSV file.")

    # If root-level data is used, try to load the model from the same directory
    model_path = model_dir
    print("Loading SavedModel from disk...")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    topic_lookup = tf.keras.layers.StringLookup(vocabulary=os.path.join(vocab_dir, "labels.txt"), num_oov_indices=0)

    MAX_LENGTH = 20
    VOCAB_SIZE = 10000
    title_preprocessor = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH)
    title_preprocessor.load_assets(vocab_dir)

    print("Converting test dataframe to tf.data.Dataset...")
    test_ds = df_to_tfdata(test_df, topic_lookup, title_preprocessor)

    print("Evaluating model on test set...")
    results = model.evaluate(test_ds, verbose=1)

    print("\nEvaluation results:")
    print(results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the baseline evaluation for the news title classifier")
    parser.add_argument("--base-dir", default=".", help="Experiment directory. Use '.' for repo root or './E1' for the notebook-style layout.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.base_dir):
        print(f"ERROR: base dir {args.base_dir} does not exist.")
        sys.exit(1)

    evaluate_baseline(args.base_dir)


if __name__ == "__main__":
    main()
