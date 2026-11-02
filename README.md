# News Article Classifier for Production-Ready ML

## Project overview

This project focuses on a practical machine learning problem: helping a company automatically classify incoming news articles so they can route content, prioritize coverage, and scale operations without relying on manual review.

The business need is straightforward: as the number of news sources grows, manual categorization becomes slow, inconsistent, and expensive. A rule-based system is brittle and often fails to capture the semantic meaning of article titles. A trained text classifier can help automate this process and improve operational efficiency.

## Problem statement

The company receives a large volume of article titles from multiple sources every day. These titles must be grouped into meaningful categories such as technology, business, sports, health, science, and world news.

The challenge is to build a model that:
- reads a news title,
- understands the content and context,
- predicts the correct topic category,
- can be deployed as a simple API for production-like use.

## Dataset

The project uses a labeled dataset of news article titles and their associated topics.

Example labels in the repository include:
- ENTERTAINMENT
- HEALTH
- TECHNOLOGY
- WORLD
- BUSINESS
- SPORTS
- NATION
- SCIENCE

The core training dataset is stored in:
- train_data.csv
- test_data.csv

The model reads the title field and predicts the corresponding topic label.

## Approach

This is a classic text classification workflow using machine learning and TensorFlow:

1. Data preparation
   - Load article titles and topic labels
   - Clean the text input
   - Convert labels to integer indices for training

2. Text preprocessing
   - Use TensorFlow TextVectorization to turn titles into sequence tokens
   - Standardize vocabulary and max sequence length

3. Model building
   - Train a simple neural network for multi-class classification
   - Use an embedding layer + dense layers + softmax output

4. Evaluation
   - Measure classification accuracy on a held-out test set
   - Log experiment metrics to experiment-tracker.csv

5. Serving
   - Expose the trained model through a FastAPI API
   - Accept a title as input and return the top predicted categories with probabilities

## Model and results

The project implements a lightweight deep learning text classifier built with TensorFlow.

The checklist of the model pipeline includes:
- title preprocessing
- label encoding
- model training
- validation/evaluation
- saved model export
- API-based inference

The repository includes a prototype evaluation result showing approximately 77.8% categorical accuracy on the evaluation split. This demonstrates that the model can learn meaningful topic patterns from article titles, even with a simple architecture and a small dataset.

This is a strong starting point for a production system and can be extended with:
- better text cleaning,
- model tuning,
- larger datasets,
- transformer-based architectures,
- logging and monitoring,
- retraining pipelines.

## Production-style architecture

The project follows a typical ML workflow:

- Data collection and labeling
- Experimentation and model training
- Model evaluation
- Artifact saving
- Inference service deployment

## Repository structure

```text
Design-machine-Learning-System-in-production/
├── README.md                          # Project overview and run instructions
├── Dockerfile                         # Container setup for the inference API
├── requirements.txt                   # Python dependencies
├── config.yaml                       # Training configuration
├── train_data.csv                    # Training data
├── test_data.csv                     # Test data
├── labels.txt                        # Label vocabulary
├── vocabulary.txt                    # Token vocabulary
├── saved_model.pb                    # Saved TensorFlow model artifact
├── variables.index                   # TensorFlow model metadata
├── variables.data-00000-of-00001     # TensorFlow model weights
├── news_articles.db                  # SQLite database artifact
├── experiment-tracker.csv            # Evaluation metrics log
├── FinalProject.ipynb                # Notebook-based exploratory work
├── FinalProject (1).ipynb            # Notebook iteration
├── FinalProject (2).ipynb            # Notebook iteration
├── src/
│   ├── __init__.py
│   ├── app.py                        # FastAPI prediction service
│   ├── train.py                      # Training script
│   ├── predict.py                    # Single-title prediction script
│   ├── evaluate.py                   # Evaluation script
│   ├── experiment.py                 # Helper utilities
│   └── utils.py                      # Shared preprocessing helpers
├── pipeline_diagram.png               # Architecture/flow diagram
├── ml_lifecycle.png                   # ML lifecycle visualization
├── ml_lifecycle_modeling.png          # Modeling workflow diagram
└── network.png                       # Network illustration
```

## Typical ML lifecycle implemented

This project is aligned with a standard ML project flow:

1. Problem definition
2. Data acquisition
3. Data exploration
4. Feature engineering / text preprocessing
5. Model training
6. Validation and testing
7. Model saving
8. Deployment for real-world use

## Run locally

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start the API

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 4) Test the endpoint

Open the app in a browser or send a POST request:

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

## Why this project matters

This repository demonstrates a complete applied ML workflow in a compact form:
- real business problem,
- structured data,
- text-based model training,
- evaluation and iteration,
- deployment as a machine learning service.

It is a good example of an entry-level but realistic ML product project that bridges experimentation and production thinking.

## Summary

This project shows how a business problem can be solved with machine learning by turning a manual text-categorization task into an automated classification system. The project covers the full journey from data and modeling to deployment, making it a strong example of applied AI in a real-world setting.

## License

The repository does not currently declare a specific license in the README. Please check repository settings or the codebase if a public license is required for reuse.

