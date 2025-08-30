from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification
import torch
import torchaudio

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForAudioClassification.from_pretrained("prithivMLmods/Common-Voice-Gender-Detection")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h2>Upload Audio for Gender Detection</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".wav,.mp3,.flac,.ogg">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    waveform, sr = torchaudio.load(file.file)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
    labels = model.config.id2label
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return JSONResponse(content=result)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
