import os, io, sys
from pathlib import Path
import numpy as np, pytest
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
DEP = ROOT / "models" / "deploy"
pytestmark = pytest.mark.skipif(not (DEP / "classifier.onnx").exists(), reason="exported models not found")

def png(w=300, h=200):
    arr = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8); b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c: yield c

def test_health(client): assert client.get("/health").json()["status"] == "ok"
def test_predict_keys_and_probabilities(client):
    j = client.post("/predict", files={"file": ("a.png", png(), "image/png")}).json()
    assert set(j) >= {"cls", "confidence", "calibration", "probabilities"} and abs(sum(j["probabilities"].values()) - 1) < 1e-4
def test_segment(client): assert "tumor_area_px" in client.post("/segment", files={"file": ("a.png", png(), "image/png")}).json()
def test_rejects_non_image(client): assert client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400
def test_analyze_schema(client):
    j = client.post("/analyze", files={"file": ("a.png", png(), "image/png")}).json()
    assert j["trust_level"] in ("HIGH", "MEDIUM", "LOW") and "class" in j and "disclaimer" in j
