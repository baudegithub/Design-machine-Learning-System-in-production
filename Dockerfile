FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV MYNEWS_BASE_DIR=/app

CMD ["python", "-m", "mynews_classifier.app", "--base-dir", "/app", "--host", "0.0.0.0", "--port", "5000"]
