# Day 59 — Digit Recognition App (Model + API)

Part of **NIZAM AI · 150-Day AI Engineering Protocol** · Month 2 Project

## 📌 Concept

`torch.save(model.state_dict())` writes trained weights to a file;
`load_state_dict()` loads them back — no retraining needed. **FastAPI**
wraps the model behind an HTTP endpoint: send an image, get a JSON
prediction back — usable from any app (web, mobile, etc).

(Same sklearn `digits` dataset as Days 57-58 — no network access for real MNIST.)

## 📂 Files

| File | Description |
|---|---|
| `train_and_save.py` | Trains the digit MLP, saves weights to `digit_model.pt` |
| `app.py` | FastAPI app: `POST /predict` takes an image, returns prediction JSON |
| `test_app.py` | 3 tests using FastAPI's TestClient (no server needed) |

## ▶️ Usage

```bash
pip install torch scikit-learn fastapi uvicorn pillow python-multipart httpx

python3 train_and_save.py     # trains model, saves digit_model.pt
python3 test_app.py           # run tests

# To actually serve it:
uvicorn app:app --reload
# then POST an image to http://localhost:8000/predict
```

## ✅ Results

```
Test accuracy: 0.9889
3/3 checks passed
```

## 📚 Sources

- FastAPI docs
- (Gradio/Streamlit docs — alternative for a quick UI instead of raw API)

---
*NIZAM AI Day 59/150*
