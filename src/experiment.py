"""
Lightweight experiment runner converted from the notebook.
Usage:
    python -m src.experiment --base-dir ./E1

This script loads the test dataset, loads the pretrained SavedModel inside the experiment folder,
applies the provided TextVectorization and StringLookup assets, and evaluates the model.

The style and messages are written in a personal, practical voice rather than a lesson.
"""

import argparse
import os
import sys

import tensorflow as tf
import pandas as pd

# Prefer the lab_utils helper from the original repo if it's available
try:
    import lab_utils
    HAS_LAB_UTILS = True
except Exception:
    HAS_LAB_UTILS = False

from .utils import set_experiment_dirs, df_to_tfdata


def evaluate_baseline(base_dir: str):
    data_dir, model_dir, vocab_dir = set_experiment_dirs(base_dir)

    print(f"Using experiment base directory: {base_dir}")
    print(f"  data:  {data_dir}")
    print(f"  model: {model_dir}")
    print(f"  vocab: {vocab_dir}\n")

    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    # Load the saved model
    print("Loading SavedModel from disk...")
    model = tf.keras.models.load_model(model_dir)
    model.summary()

    # Create label lookup
    topic_lookup = tf.keras.layers.StringLookup(vocabulary=f'{vocab_dir}/labels.txt', num_oov_indices=0)

    # Load the title preprocessor used by the prototype (vocabulary assets)
    MAX_LENGTH = 20
    VOCAB_SIZE = 10000
    title_preprocessor = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH)
    title_preprocessor.load_assets(vocab_dir)

    # Convert test dataframe to tf.data.Dataset
    print("Converting test dataframe to tf.data.Dataset...")
    test_ds = df_to_tfdata(test_df, topic_lookup, title_preprocessor)

    print("Evaluating model on test set...")
    results = model.evaluate(test_ds, verbose=1)

    print("\nEvaluation results:")
    if isinstance(results, list):
        print(results)
    else:
        print([results])

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the baseline evaluation for the news title classifier")
    parser.add_argument("--base-dir", default="./E1", help="Path to the experiment folder (default: ./E1)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.base_dir):
        print(f"ERROR: base dir {args.base_dir} does not exist. Make sure you have the E1/ folder from the prototype.")
        sys.exit(1)

    evaluate_baseline(args.base_dir)


if __name__ == "__main__":
    main()
