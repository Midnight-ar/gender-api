FROM python:3.10-slim

# System deps (ffmpeg helps decode mp3/ogg/flac reliably)
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py /app/app.py

# Hugging Face Spaces expects your app to listen on $PORT (default 7860)
ENV PORT=7860
EXPOSE 7860

# Start FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
