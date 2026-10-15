import argparse

from mynews_classifier.data_stage import main as run_data_stage
from mynews_classifier.modeling_stage import main as run_modeling_stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full project pipeline for MyNewsClassifier")
    parser.add_argument("--base-dir", default=".", help="Base experiment directory")
    args = parser.parse_args()

    run_data_stage(args.base_dir)
    run_modeling_stage(args.base_dir)
