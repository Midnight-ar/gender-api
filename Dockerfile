FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Railway sets $PORT (default 8000), so expose that
EXPOSE 8000
ENV PORT=8000

# Start FastAPI with uvicorn (not gunicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
