"""Local pipeline: YOUR trained checkpoints -> class + calibrated confidence + tumor mask + Grad-CAM (library) + trust level."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.classification.model import build_efficientnet_b0
from src.segmentation.model import UNetPlusPlus, EffB1Encoder
from local_app import core

# name shown in the UI -> class in the `grad-cam` library (pip install grad-cam)
#CAM_METHODS = {"Grad-CAM": "GradCAM", "Grad-CAM++": "GradCAMPlusPlus", "HiRes-CAM": "HiResCAM", "XGrad-CAM": "XGradCAM",
               #"Layer-CAM": "LayerCAM", "Eigen-CAM": "EigenCAM", "Score-CAM (slow)": "ScoreCAM"}
CAM_METHODS = {"Grad-CAM": "GradCAM", "Grad-CAM++": "GradCAMPlusPlus", "Layer-CAM": "LayerCAM"}
LAYERS = {3: "features[3] (early, fine)", 5: "features[5] (middle)", 8: "features[8] (last conv, 8x8; used for trust)"}


class BrainTumorXAI:
    method_names = list(CAM_METHODS)
    def __init__(self, cls_ckpt=None, seg_ckpt=None, deploy_dir=None, device=None):
        self.deploy = Path(deploy_dir or ROOT / "models" / "deploy"); self.cfg = core.load_cfg(self.deploy)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); self.classes = self.cfg["classes"]
        cls_ckpt = Path(cls_ckpt or self.deploy / "classifier_best.pt")
        if seg_ckpt is None:
            prim = json.load(open(ROOT / "configs" / "segmentation_primary.json"))["primary"]; seg_ckpt = ROOT / "models" / f"{prim}_best.pt"
        for p in (cls_ckpt, Path(seg_ckpt)):
            if not Path(p).exists(): raise FileNotFoundError(f"{p} not found - copy your downloaded checkpoint there")
        ck = torch.load(cls_ckpt, map_location=self.device, weights_only=False)
        self.cls = build_efficientnet_b0(pretrained=False).to(self.device); self.cls.load_state_dict(ck["model"]); self.cls.eval()
        cs = torch.load(seg_ckpt, map_location=self.device, weights_only=False); c = cs.get("cfg", {})
        self.seg = UNetPlusPlus(pretrained=False, nested=c.get("nested", True))
        if c.get("variant", "b0") == "b1": self.seg.encoder = EffB1Encoder(pretrained=False)
        self.seg.load_state_dict(cs["model"]); self.seg = self.seg.to(self.device).eval()

    # ---------- explanations ----------
    def _own_cam(self, x, k, layer):
        """The project's own Grad-CAM (the one the trust thresholds were validated with)."""
        from src.xai.methods import MultiGradCAM
        with torch.enable_grad():
            mc = MultiGradCAM(self.cls, {"L": self.cls.features[layer]}); cams, _, _ = mc(x.clone(), target=k); mc.remove()
        return cams["L"][0].detach().cpu().numpy()

    def _lib_cam(self, x, k, method, layer):
        """CAM from the `pytorch-grad-cam` library. Returns a 256x256 map in [0,1]."""
        import pytorch_grad_cam as pgc
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        cam = getattr(pgc, CAM_METHODS[method])(model=self.cls, target_layers=[self.cls.features[layer]])
        try:
            with torch.enable_grad(): g = cam(input_tensor=x.clone(), targets=[ClassifierOutputTarget(k)])[0]
        finally:
            if hasattr(cam, "activations_and_grads"): cam.activations_and_grads.release()
        return g.astype(np.float32)

    def explain(self, x, k, method="Grad-CAM", layer=8, backend="lib"):
        if backend == "own":
            if method != "Grad-CAM": raise ValueError("backend 'own' only implements Grad-CAM; use backend 'lib' for the other methods")
            return self._own_cam(x, k, layer)
        return self._lib_cam(x, k, method, layer)

    # ---------- full analysis ----------
    def _infer(self, im: Image.Image):
        """Classifier + segmenter forward pass shared by analyze() and compare()."""
        cfg = self.cfg; pr = core.prepare(im, cfg)
        xc, xs = torch.from_numpy(pr["xc"]).to(self.device), torch.from_numpy(pr["xs"]).to(self.device)
        with torch.no_grad():
            z_raw = self.cls(xc)[0].float().cpu().numpy()
            pm = torch.sigmoid(self.seg(xs))[0, 0].float().cpu().numpy()
        sm = lambda z: (lambda e: e / e.sum())(np.exp(z - z.max()))
        p, p_raw = sm(z_raw / cfg["temperature"]), sm(z_raw)                          # calibrated (temperature-scaled) and raw probabilities
        k = int(p.argmax()); cls = self.classes[k]
        mask_s = pm > cfg["seg_thr"]; px = int(mask_s.sum()); pct = float(mask_s.mean() * 100)
        return dict(pr=pr, xc=xc, k=k, cls=cls, conf=float(p[k]), p=p, p_raw=p_raw, px=px, pct=pct, mask=core.to_cls_frame(mask_s, pr["hw"], cfg),
                    tumor=cls != "no_tumor", found=px > core.MIN_TUMOR_PX)

    def analyze(self, im: Image.Image, method="Grad-CAM", layer=8, backend="lib"):
        cfg = self.cfg; s = self._infer(im); xc, k, mask = s["xc"], s["k"], s["mask"]
        ref = self._own_cam(xc, k, 8)                                                  # validated map -> used for the trust signal
        cam = ref if (backend == "own" and method == "Grad-CAM" and layer == 8) else self.explain(xc, k, method, layer, backend)
        consist = float((ref * mask).sum() / max(ref.sum(), 1e-12)) if (s["tumor"] and s["px"] > 0) else None
        lib_vs_own = None
        if backend == "lib" and method == "Grad-CAM":                                   # sanity check: library vs project implementation
            own = ref if layer == 8 else self._own_cam(xc, k, layer); lib_vs_own = float(np.corrcoef(cam.ravel(), own.ravel())[0, 1])
        P = lambda v: {c: float(x) for c, x in zip(self.classes, v)}
        return dict(arr_c=s["pr"]["arr_c"], mask=mask, cam=cam, cls=s["cls"], k=k, conf=s["conf"], probs=P(s["p"]), probs_raw=P(s["p_raw"]),
                    calibration=core.band(s["conf"], cfg), trust=core.trust_level(s["conf"], s["tumor"] == s["found"], consist), consistency=consist,
                    tumor=s["tumor"], found=s["found"], px=s["px"], pct=s["pct"], T=cfg["temperature"], disclaimer=cfg["disclaimer"],
                    text=core.explanation_text(s["cls"], s["tumor"], s["found"], s["pct"], consist), lib_vs_own_corr=lib_vs_own,
                    method_label=f"{method} (features[{layer}], {backend})", method=method, layer=layer, backend=backend)

    def compare(self, im: Image.Image, layer=8):
        """Heat-maps of the predicted class from several library methods (at `layer`) and from Grad-CAM at layers 3/5/8. A failing method is reported, not fatal."""
        s = self._infer(im); out = dict(arr_c=s["pr"]["arr_c"], mask=s["mask"], cls=s["cls"], k=s["k"], tumor=s["tumor"], methods=[], layers=[])
        def one(method, lyr):
            try: return dict(cam=self.explain(s["xc"], s["k"], method, lyr, "lib"), error=None)
            except Exception as e: return dict(cam=None, error=f"{type(e).__name__}: {e}")
        for m in CAM_METHODS:
            if m.startswith("Score-CAM"): continue                                       # too slow for an interactive grid
            out["methods"].append(dict(label=m, layer=layer, **one(m, layer)))
        for lyr in LAYERS: out["layers"].append(dict(label=f"Grad-CAM, {LAYERS[lyr].split(' (')[0]}", layer=lyr, **one("Grad-CAM", lyr)))
        return out