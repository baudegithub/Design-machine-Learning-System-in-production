# News Title Classifier — Personal Project

This repository contains a compact, runnable ML project converted from an interactive notebook into a small project structure so you can run quick experiments locally.

Goal
- Reproduce a baseline evaluation of a pretrained title-based news topic classifier (SavedModel) that was included in the original prototype (E1/model).
- Make it easy to iterate: load data, preprocess with the provided vocabulary, evaluate model metrics locally.

Why I made this
- I wanted a small, portable project I can run and extend quickly on my laptop. It's focused and practical — not a tutorial.

Quick start
1. Create a Python virtual environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Run the baseline evaluation (assumes E1/ is present at repo root):

   python -m src.experiment --base-dir ./E1

What the script does
- Loads data from <base-dir>/data (train_data.csv, test_data.csv)
- Loads vocabulary and labels from <base-dir>/vocab
- Loads the SavedModel from <base-dir>/model
- Prepares tf.data.Dataset using the project's preprocessing and evaluates the model on the test set

Files added
- src/experiment.py — main script to run experiments
- src/utils.py — small helpers to load dirs and datasets
- requirements.txt — minimal dependencies
- README.md — this file

Notes
- The project re-uses the helper module lab_utils from the original materials (the notebook imports it). If lab_utils is missing, see src/utils.py for alternative helpers.
- If you want me to add training, ablations, or a lightweight CLI for hyperparameter sweeps, tell me which experiments you'd like to run and I'll add them.
