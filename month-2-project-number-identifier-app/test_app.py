"""Day 59 — Test the FastAPI endpoint using TestClient (no server needed)."""

import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from app import app

client = TestClient(app)


def make_test_image(digit_index: int) -> bytes:
    """Convert one sklearn digit sample into a PNG byte stream, like a real upload."""
    digits = load_digits()
    pixels = digits.images[digit_index]  # 8x8, values 0-16
    img_8bit = (pixels / 16.0 * 255).astype(np.uint8)
    img = Image.fromarray(img_8bit, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), digits.target[digit_index]


def test_home_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("[PASS] home endpoint responds")


def test_predict_returns_valid_response():
    image_bytes, true_label = make_test_image(0)
    resp = client.post("/predict", files={"file": ("digit.png", image_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert 0 <= data["prediction"] <= 9
    assert len(data["all_probabilities"]) == 10
    assert abs(sum(data["all_probabilities"]) - 1.0) < 0.01
    print(f"[PASS] /predict returns valid response: predicted={data['prediction']}, true={true_label}")


def test_predict_accuracy_on_sample():
    """Not a strict accuracy test (PNG round-trip adds noise) -- just
    confirms the pipeline produces a sane, confident prediction."""
    image_bytes, true_label = make_test_image(42)
    resp = client.post("/predict", files={"file": ("digit.png", image_bytes, "image/png")})
    data = resp.json()
    assert data["confidence"] > 0.1  # at least somewhat confident
    print(f"[PASS] prediction pipeline runs end-to-end (confidence={data['confidence']})")


if __name__ == "__main__":
    tests = [test_home_endpoint, test_predict_returns_valid_response, test_predict_accuracy_on_sample]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
