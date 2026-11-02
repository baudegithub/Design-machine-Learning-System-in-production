# News Article Classifier for Production-Ready ML

## Executive summary

This project builds a machine learning pipeline to automatically classify news article titles into predefined categories such as technology, business, health, sports, and world news. The goal is to reduce manual review and help teams process large volumes of incoming content more efficiently.

The repository combines a notebook-based exploratory workflow with a production-style inference API. It demonstrates a realistic end-to-end ML workflow: data preparation, text preprocessing, model training, evaluation, model artifact storage, and deployment of a simple prediction service.

## Problem

The company receives a large number of news articles from multiple sources. Manually sorting each article into categories is slow, inconsistent, and difficult to scale as the volume of incoming content grows.

A traditional rule-based approach is limited because article titles can be ambiguous and semantically rich. A machine learning model can learn patterns from labeled examples and classify new titles more effectively and at scale.

## Business impact

Automating this classification task supports:
- faster routing of headlines into the correct category,
- better content organization,
- reduced manual effort,
- scalability as the number of sources increases,
- support for operational workflows and downstream content systems.

## Data

The project uses a labeled dataset of article titles mapped to topical categories.

Example labels include:
- ENTERTAINMENT
- HEALTH
- TECHNOLOGY
- WORLD
- BUSINESS
- SPORTS
- NATION
- SCIENCE

The main training and evaluation files are:
- train_data.csv
- test_data.csv
- labels.txt
- vocabulary.txt

These assets support the model’s ability to classify article titles based on patterns in the text.

## Approach

This project follows a typical machine learning pipeline for text classification.

### 1. Data preparation
- load article title and topic label pairs,
- convert labels into model-readable indices,
- prepare train/test splits.

### 2. Text preprocessing
- use TensorFlow TextVectorization,
- convert raw titles into token sequences,
- build a vocabulary for the model input.

### 3. Model training
- train a compact neural network for multi-class classification,
- use an embedding layer and dense layers to learn text patterns,
- apply a softmax output for multi-label topic prediction.

### 4. Model evaluation
- assess model quality on a held-out test set,
- track metrics in experiment-tracker.csv.

### 5. Deployment
- save the trained model artifact,
- serve predictions with a FastAPI application,
- expose an HTTP endpoint for classification requests.

## Key result

The repository includes an evaluation artifact showing a categorical accuracy of approximately 77.8% on the evaluation split. While this is a lightweight prototype, it demonstrates that the model can learn meaningful news-topic patterns from title text and provides a credible baseline for further improvement.

## Typical ML project structure

```text
Design-machine-Learning-System-in-production/
├── README.md
├── Dockerfile
├── requirements.txt
├── config.yaml
├── train_data.csv
├── test_data.csv
├── labels.txt
├── vocabulary.txt
├── saved_model.pb
├── variables.index
├── variables.data-00000-of-00001
├── news_articles.db
├── experiment-tracker.csv
├── FinalProject.ipynb
├── FinalProject (1).ipynb
├── FinalProject (2).ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── experiment.py
│   └── utils.py
├── pipeline_diagram.png
├── ml_lifecycle.png
├── ml_lifecycle_modeling.png
├── network.png
└── other model artifacts
```

This structure reflects a common ML lifecycle:
- data and labels,
- exploratory notebook work,
- reusable training and evaluation scripts,
- saved model artifacts,
- deployment interface for inference.

## Repository components

- FinalProject.ipynb: notebook-based experimentation and model exploration
- src/train.py: training pipeline for the classifier
- src/evaluate.py: evaluation script for test-set validation
- src/predict.py: command-line inference for individual titles
- src/app.py: FastAPI API for serving predictions
- src/utils.py and src/experiment.py: shared utility and environment handling
- Dockerfile: containerized deployment support

## Running the project locally

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the prediction endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"title": "Apple unveils new iPhone", "top_k": 3}'
```

## Docker

```bash
docker build -t news-title-classifier .
docker run -p 8000:8000 --rm -e BASE_DIR=. news-title-classifier
```

## Skills demonstrated

This project showcases practical understanding of:
- machine learning problem framing,
- text classification,
- data preparation and labeling,
- TensorFlow model training,
- model evaluation and experimentation,
- model packaging and deployment,
- API-based inference serving.

## Why this project is valuable

This repository is a good example of an applied ML project that moves beyond notebook-only experimentation. It combines real-world problem solving, end-to-end pipeline thinking, and deployment awareness. It demonstrates how a simple but effective ML model can support business operations in a real use case.

## Future improvements

This project could be extended with:
- stronger text cleaning and normalization,
- hyperparameter tuning,
- larger and more diverse benchmark datasets,
- transformer-based models such as BERT,
- MLOps tooling for retraining and model monitoring,
- a production deployment pipeline with CI/CD and monitoring.

## Summary

This repo presents a practical machine learning solution for automatically classifying news article titles into topic categories. It combines experimentation, model training, evaluation, and a production-style API layer into a compact and understandable project. The work demonstrates a realistic ML workflow that is relevant to both business use cases and applied AI portfolios.
