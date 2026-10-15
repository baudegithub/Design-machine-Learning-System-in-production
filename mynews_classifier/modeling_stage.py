import argparse
import logging
import os

import pandas as pd
import tensorflow as tf

from mynews_classifier.utils import df_to_tfdata, set_experiment_dirs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mynews_classifier.modeling_stage")

MAX_LENGTH = 20
VOCAB_SIZE = 10000


def build_topic_lookup(vocab_dir: str):
    labels_path = os.path.join(vocab_dir, "labels.txt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.txt not found in {vocab_dir}")
    return tf.keras.layers.StringLookup(vocabulary=labels_path, num_oov_indices=0, mask_token=None)


def build_title_preprocessor(vocab_dir: str):
    title_preprocessor = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH,
    )
    title_preprocessor.load_assets(vocab_dir)
    return title_preprocessor


def evaluate_model(model_dir: str, test_df: pd.DataFrame, topic_lookup, title_preprocessor):
    model = tf.keras.models.load_model(model_dir)
    logger.info("Model summary:")
    model.summary()
    logger.info("Model compile config: %s", model.get_compile_config())

    test_ds = df_to_tfdata(test_df, topic_lookup, title_preprocessor, batch_size=64)
    logger.info("Evaluating on the test set...")
    metrics = model.evaluate(test_ds, verbose=1)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def main(base_dir: str = "."):
    data_dir, model_dir, vocab_dir = set_experiment_dirs(base_dir)
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    topic_lookup = build_topic_lookup(vocab_dir)
    title_preprocessor = build_title_preprocessor(vocab_dir)
    evaluate_model(model_dir, test_df, topic_lookup, title_preprocessor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation stage")
    parser.add_argument("--base-dir", default=".", help="Base experiment directory")
    args = parser.parse_args()
    main(args.base_dir)
