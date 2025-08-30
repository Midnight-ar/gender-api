# app.py
import os
import tempfile
import subprocess
from pathlib import Path

import torch
# Limit PyTorch threads to reduce memory/CPU pressure on small containers
torch.set_num_threads(1)

import torchaudio
import soundfile as sf
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# NOTE: we lazy-load these inside get_model()
processor = None
model = None

TARGET_SR = 16000  # wav2vec2 expects 16 kHz

def get_model():
    """
    Lazily load processor and model on first call and cache them globally.
    Call inside request handlers to avoid heavy startup on cold starts.
    """
    global processor, model
    if processor is None or model is None:
        print("🔁 Loading HF processor & model (this may take 10-60s on first request)...")
        from transformers import Wav2Vec2Processor, AutoModelForAudioClassification
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = AutoModelForAudioClassification.from_pretrained(
            "prithivMLmods/Common-Voice-Gender-Detection"
        )
        model.eval()
        print("✅ Model & processor loaded.")
    return processor, model


app = FastAPI(title="Gender Detection API (lazy model load)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
      <body>
        <h2>Upload Audio for Gender Detection</h2>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept=".wav,.mp3,.flac,.ogg" />
          <input type="submit" value="Upload" />
        </form>
        <p>POST /predict (multipart form-data, field name "file")</p>
      </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/labels")
async def labels():
    proc, mdl = get_model()
    return mdl.config.id2label


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        proc, mdl = get_model()

        # Save upload to a temporary file
        suffix = Path(file.filename or "").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await file.read()
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            # Try to read using soundfile (libsndfile)
            try:
                waveform_np, sr = sf.read(tmp_path, dtype="float32")
            except Exception as e:
                # If soundfile fails (some mp3/ogg), try using ffmpeg to convert to WAV then read
                print("⚠️ soundfile could not read directly, trying ffmpeg conversion:", e)
                converted = tmp_path + ".converted.wav"
                # Use ffmpeg CLI (ffmpeg must be installed in the container)
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", tmp_path,
                    "-ar", str(TARGET_SR), "-ac", "1", converted
                ]
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                waveform_np, sr = sf.read(converted, dtype="float32")
                try:
                    os.unlink(converted)
                except Exception:
                    pass

        finally:
            # remove uploaded tmp file as soon as possible
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # waveform_np shape: (n_samples,) or (n_samples, channels)
        if waveform_np.ndim > 1:
            # average channels to mono
            waveform_np = waveform_np.mean(axis=1)

        # Convert to torch tensor shape [1, n_samples]
        waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)

        # Resample if necessary
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            sr = TARGET_SR

        # Prepare inputs for HF model
        inputs = proc(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = mdl(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        labels_map = mdl.config.id2label
        result = {labels_map[i]: float(probs[i]) for i in range(len(labels_map))}
        top_idx = int(probs.argmax())

        return JSONResponse(content={"top": labels_map[top_idx], "scores": result})

    except Exception as e:
        import traceback
        print("🔥 Error in /predict:", e)
        traceback.print_exc()
        # Return the error string (400) so client can see the reason
        return JSONResponse(status_code=400, content={"error": str(e)})


if __name__ == "__main__":
    # Local dev fallback (Railway/Gunicorn uses CMD from Dockerfile)
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
