import argparse
import logging
import os

import pandas as pd

from mynews_classifier.utils import set_experiment_dirs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mynews_classifier.data_stage")


def load_data_files(data_dir: str):
    train_path = os.path.join(data_dir, "train_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    logger.info("Loading train: %s", train_path)
    logger.info("Loading test: %s", test_path)
    return pd.read_csv(train_path), pd.read_csv(test_path)


def show_sample(train_df: pd.DataFrame, n: int = 10):
    pd.set_option("display.max_colwidth", None)
    logger.info("Showing first %d rows of the training set", n)
    print(train_df[["title", "link", "domain", "published_date", "topic"]].head(n))


def class_proportions(df: pd.DataFrame, col: str = "topic"):
    counts = df[col].value_counts(normalize=True) * 100
    return counts.round(1).astype(str) + "%"


def main(base_dir: str = "."):
    data_dir, _, _ = set_experiment_dirs(base_dir)
    train_df, test_df = load_data_files(data_dir)
    show_sample(train_df, n=10)

    logger.info("Train class proportions:")
    print(class_proportions(train_df).sort_index())
    logger.info("Test class proportions:")
    print(class_proportions(test_df).sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data inspection stage for MyNewsClassifier")
    parser.add_argument("--base-dir", default=".", help="Base experiment directory")
    args = parser.parse_args()
    main(args.base_dir)
