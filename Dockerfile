# Dockerfile
FROM python:3.10-slim

# Install system deps for audio decoding + basic build utilities (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app source
COPY . /app

# Default port (Railway provides PORT env; default to 8000 if not set)
ENV PORT=8000
EXPOSE 8000

# Use Gunicorn with Uvicorn worker for better worker lifecycle management.
# Use sh -c so $PORT expands at container run time.
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:$PORT --workers 1 --timeout 120"]
