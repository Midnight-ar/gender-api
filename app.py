import io
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification

# -----------------------------
# Model & processor
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForAudioClassification.from_pretrained(
    "prithivMLmods/Common-Voice-Gender-Detection"
)
model.eval()

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Gender Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SR = 16000


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
      </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/labels")
async def labels():
    return model.config.id2label


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save upload to a temp file
        suffix = Path(file.filename or "").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await file.read()
            tmp.write(raw)
            tmp_path = tmp.name

        # Load audio
        waveform, sr = torchaudio.load(tmp_path)

        # Clean up
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            sr = TARGET_SR

        # Preprocess for wav2vec2
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        labels = model.config.id2label
        result = {labels[i]: float(probs[i]) for i in range(len(labels))}
        top_idx = int(probs.argmax())

        return JSONResponse(content={"top": labels[top_idx], "scores": result})

    except Exception as e:
        import traceback
        # 🔥 Debugging logs for Railway
        print("🔥 Error in /predict:", e)
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

