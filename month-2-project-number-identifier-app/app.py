"""
Day 59 — FastAPI endpoint: image in, JSON prediction out.
NIZAM AI · 150-Day AI Engineering Protocol

Run: uvicorn app:app --reload
Then POST an image to /predict (8x8 grayscale, like sklearn digits).
"""

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from train_and_save import DigitMLP

app = FastAPI(title="Digit Recognition API")

model = DigitMLP()
model.load_state_dict(torch.load("digit_model.pt", map_location="cpu"))
model.eval()


@app.get("/")
def home():
    return {"status": "ok", "message": "POST an 8x8 grayscale image to /predict"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((8, 8))
    pixels = np.array(img, dtype=np.float32) / 255.0 * 16.0  # match sklearn digits scale (0-16)
    x = torch.from_numpy(pixels.flatten()).unsqueeze(0) / 16.0  # normalize like training data

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())

    return {
        "prediction": pred,
        "confidence": round(float(probs[pred]), 4),
        "all_probabilities": [round(float(p), 4) for p in probs],
    }
