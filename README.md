# MyNewsClassifier

A small personal project for classifying news article titles into topic categories using the TensorFlow/Keras model from the ML lifecycle case study.

## What this project does

- Loads the news dataset and inspects the labels and feature quality.
- Reuses the notebook workflow for data exploration and baseline evaluation.
- Provides scripts for each stage of the project lifecycle.
- Includes a simple Flask app for local prediction testing.
- Packages the project for local Docker-based execution.

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
├── train_data.csv
├── test_data.csv
├── labels.txt
├── vocabulary.txt
├── saved_model.pb
├── keras_metadata.pb
├── variables.data-00000-of-00001
├── variables.index
└── FinalProject.ipynb
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the data and modeling stages

From the repo root:

```bash
python -m mynews_classifier.run_all --base-dir .
```

This runs the same core logic as the notebook:
- data inspection and class proportions
- model loading
- preprocessing setup
- evaluation on the test set

## Run the local prediction app

```bash
python -m mynews_classifier.app
```

Then send a POST request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Apple launches new AI chip for laptops"}'
```

## Run with Docker

```bash
docker build -t mynews-classifier .
docker run -p 5000:5000 mynews-classifier
```

The app will be available at:

```text
http://localhost:5000/health
http://localhost:5000/predict
```

## Notes

- This project was converted from the notebook workflow into scripts for a personal, reusable setup.
- The code keeps the same logic as the original case study while removing the classroom narrative.
- The app is intended for local development and experimentation, not for production deployment.

## Source material

This project is based on the workflow in the notebook:
- FinalProject.ipynb
- lab_utils (helper logic reused and adapted)
