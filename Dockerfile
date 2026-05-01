FROM python:3.9-slim

WORKDIR /app

# System deps: gcc for some pip builds, libpq for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces Docker runtime requires port 7860
EXPOSE 7860

# 1 worker: loads TabNet weights once; 120s timeout for cold-start inference
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
