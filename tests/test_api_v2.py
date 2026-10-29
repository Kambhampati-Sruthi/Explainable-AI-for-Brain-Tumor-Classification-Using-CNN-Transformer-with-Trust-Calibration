import io, sys
from pathlib import Path
import numpy as np, pytest
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "deploy" / "classifier.onnx").exists(), reason="exported models not found")

def png(w=300, h=200):
    b = io.BytesIO(); Image.fromarray((np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)).save(b, "PNG"); return b.getvalue()

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c: yield c

def test_info(client):
    j = client.get("/api/v2/info").json(); assert j["classes"] and "disclaimer" in j and len(j["limitations"]) >= 3

def test_v2_payload(client):
    j = client.post("/api/v2/analyze", files={"file": ("a.png", png(), "image/png")}).json()
    assert j["trust"]["level"] in ("HIGH", "MEDIUM", "LOW") and abs(sum(j["prediction"]["probabilities"].values()) - 1) < 1e-4
    assert j["images"]["base"] and j["preprocessing"]["original_size"] == [300, 200]

def test_ui_is_served_with_csp(client):
    r = client.get("/ui/"); assert r.status_code == 200 and "script-src 'self'" in r.headers["content-security-policy"]

def test_bad_upload(client):
    assert client.post("/api/v2/analyze", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400