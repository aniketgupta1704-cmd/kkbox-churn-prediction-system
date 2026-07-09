FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# Render sets $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000
CMD uvicorn app.api:app --host 0.0.0.0 --port $PORT