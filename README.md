# MyNewsClassifier

A personal project for classifying news article titles into topic categories using a TensorFlow/Keras model.

## Overview

This project converts the notebook workflow into a reusable codebase for:
- data inspection and class distribution analysis
- baseline evaluation of the saved model
- a local prediction API for article titles
- Docker-based local hosting

## Project structure

```text
.
├── mynews_classifier/
│   ├── __init__.py
│   ├── app.py
│   ├── data_stage.py
│   ├── modeling_stage.py
│   ├── run_all.py
│   └── utils.py
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── FinalProject.ipynb
├── train_data.csv
├── test_data.csv
├── labels.txt
├── vocabulary.txt
├── saved_model.pb
├── keras_metadata.pb
├── variables.data-00000-of-00001
├── variables.index
└── lab_utils (1).py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full project pipeline

```bash
python -m mynews_classifier.run_all --base-dir .
```

This runs:
1. data inspection
2. label distribution checks
3. model loading and evaluation on the test set

## Run the local prediction API

```bash
python -m mynews_classifier.app --base-dir . --host 0.0.0.0 --port 5000
```

Then test it:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Apple launches new AI chip for laptops"}'
```

## Docker

```bash
docker build -t mynews-classifier .
docker run -p 5000:5000 mynews-classifier
```

## Notes

This project reuses the notebook logic as a personal project structure and keeps the deployment API local for quick experimentation.
