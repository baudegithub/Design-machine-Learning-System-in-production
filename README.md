# Demo instructions

This project now includes a small FastAPI demo server to serve predictions from a saved model.

Run locally (no Docker):
1. Create and activate a virtualenv:
   python -m venv .venv
   source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Start the demo server from the repo root (where train_data.csv, test_data.csv, vocabulary.txt, labels.txt, saved_model.pb live):
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

Open http://localhost:8000/ to check status.

Predict endpoint
POST http://localhost:8000/predict
Content-Type: application/json
Body example:
{"title": "Apple unveils new iPhone", "top_k": 3}

Docker (build and run):
  docker build -t news-title-classifier .
  docker run -p 8000:8000 --rm -e BASE_DIR=. news-title-classifier

If your assets are under ./E1, set BASE_DIR accordingly:
  export BASE_DIR=./E1
  uvicorn src.app:app --reload
