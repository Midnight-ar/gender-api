# Gender Detection API (FastAPI on HF Spaces)

Endpoints:
- `GET /health` → `{"status":"ok"}`
- `GET /labels` → model labels
- `POST /predict` (multipart form, field name `file`) → JSON with top label and scores

Example:
```bash
curl -X POST -F "file=@sample.wav" https://USERNAME-PROJECTNAME.hf.space/predict
```
