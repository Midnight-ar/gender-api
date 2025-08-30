# Gender Detection API (Render Ready)

This FastAPI app performs gender detection from audio files using a pretrained Wav2Vec2 model.

## Deploy on Render

1. Push this repo to GitHub.
2. On Render, create a **Web Service** from the repo.
3. **Build Command**: 
```
pip install -r requirements.txt
```
4. **Start Command**: 
```
uvicorn app:app --host 0.0.0.0 --port $PORT
```
5. Do NOT set a custom PORT environment variable — Render injects it.

Once deployed, Render will provide a public URL for your API.
