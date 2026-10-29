import sys
from pathlib import Path
import numpy as np, pytest, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
CKPT = ROOT / "models" / "deploy" / "classifier_best.pt"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="classifier checkpoint not found")


@pytest.fixture(scope="module")
def setup():
    from src.classification.model import build_efficientnet_b0
    from src.xai import methods as XM
    m = build_efficientnet_b0(pretrained=False); m.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); m.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 256, 256, generator=g) * 0.5
    return m, x, XM


def test_gradcam_range_and_shape(setup):
    m, x, XM = setup; mc = XM.MultiGradCAM(m, {"L8": m.features[8]})
    cam = mc(x, target=0)[0]["L8"]; mc.remove()
    assert cam.shape == (1, 256, 256) and float(cam.min()) >= 0 and abs(float(cam.max()) - 1) < 1e-5


def test_gradcam_equals_captum_layergradcam(setup):
    from captum.attr import LayerGradCam, LayerAttribution
    m, x, XM = setup; mc = XM.MultiGradCAM(m, {"L8": m.features[8]})
    ours = mc(x, target=1, normalize=False)[0]["L8"][0].flatten(); mc.remove()
    cap = LayerAttribution.interpolate(LayerGradCam(m, m.features[8]).attribute(x, target=1, relu_attributions=True), (256, 256), interpolate_mode="bilinear")[0, 0].flatten().detach()
    assert float(torch.corrcoef(torch.stack([ours, cap]))[0, 1]) > 0.9999


def test_cam_formula_for_gap_linear_head(setup):
    """Grad-CAM must equal CAM = ReLU(sum_k W[c,k] A^k) (Grad-CAM paper: strict generalisation of CAM). This justifies the Stage 7 CAM loss."""
    m, x, XM = setup; c = 2; mc = XM.MultiGradCAM(m, {"L8": m.features[8]})
    cam, logits, _ = mc(x, target=c); A = mc.acts["L8"]; mc.remove()
    ref = F.relu(torch.einsum("bkhw,k->bhw", A.detach(), m.classifier[1].weight[c].detach()))[:, None]
    ref = F.interpolate(ref, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]; ref = ref / ref.amax().clamp_min(1e-12)
    assert float(torch.corrcoef(torch.stack([cam[0].flatten(), ref[0].flatten()]))[0, 1]) > 0.9999


def test_guided_model_has_identical_forward(setup):
    m, x, XM = setup; gm, n = XM.make_guided_model(m)
    assert n > 0
    with torch.no_grad(): assert torch.allclose(m(x), gm(x), atol=1e-5)          # only the BACKWARD rule changes


def test_integrated_gradients_completeness(setup):
    m, x, XM = setup; base = torch.zeros_like(x)
    attr, delta = XM.ig_attr(m, x, 0, base, n_steps=100)
    with torch.no_grad(): gap = (m(x)[0, 0] - m(base)[0, 0]).item()
    assert abs(attr.sum().item() - gap) < 0.05 * max(abs(gap), 1e-3) + 0.02


def test_deletion_auc_below_insertion_for_oracle_map(setup):
    """A map built from the true gradient magnitude should delete faster than it inserts for the class it explains (sanity, not a proof)."""
    from src.xai import evaluate as XE
    m, x, XM = setup; base = torch.zeros_like(x); xg = x.clone().requires_grad_(True)
    t = int(m(x).argmax()); g = torch.autograd.grad(m(xg)[0, t], xg)[0].abs().sum(1)[0]
    d, i = XE.del_ins(m, x, g, t, base)
    assert 0 <= d <= 1 and 0 <= i <= 1