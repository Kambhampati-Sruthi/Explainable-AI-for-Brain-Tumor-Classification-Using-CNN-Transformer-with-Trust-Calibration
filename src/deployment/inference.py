"""Inference engine: ONNX Runtime for prediction/segmentation; PyTorch only for /explain (Grad-CAM needs gradients)."""
import base64, io, json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from src.preprocessing.cache import process_image          # SAME preprocessing as training

MEAN_IN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def _b64(arr_u8):
    buf = io.BytesIO(); Image.fromarray(arr_u8).save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

class Engine:
    def __init__(self, deploy_dir, torch_ckpt=None):
        d = Path(deploy_dir); self.cfg = json.load(open(d / "deploy_config.json")); self.classes = self.cfg["classes"]
        prov = ["CPUExecutionProvider"]
        self.cls = ort.InferenceSession(str(d / "classifier.onnx"), providers=prov)
        self.seg = ort.InferenceSession(str(d / "segmenter.onnx"), providers=prov)
        self.torch_ckpt, self._torch = torch_ckpt, None

    # --- preprocessing: any image -> uint8 256x256 (pad policy) -> normalised (1,3,256,256) for each model ---
    def prep(self, im: Image.Image):
        arr = process_image(np.array(im.convert("L")), self.cfg["resize_policy"], self.cfg["img_size"])
        def norm(mean, std):
            x = (arr.astype(np.float32) / 255.0)[None, None].repeat(3, 1)
            return ((x - np.array(mean, np.float32).reshape(1, 3, 1, 1)) / np.array(std, np.float32).reshape(1, 3, 1, 1)).astype(np.float32)
        if self.cfg["cls_norm"] == "imagenet": c = norm(*MEAN_IN)
        elif self.cfg["cls_norm"] == "dataset": c = norm((self.cfg["stats"][0],) * 3, (self.cfg["stats"][1],) * 3)
        else: c = norm((0, 0, 0), (1, 1, 1))
        return arr, c, norm(*MEAN_IN)                      # segmenter was trained with ImageNet normalization

    def band(self, conf):
        for b in self.cfg["bands"]:
            if b["lo"] <= conf < b["hi"] or (conf >= 1.0 and b["hi"] >= 1.0): return b["label"]
        return "Insufficient data"

    def predict(self, im):
        arr, xc, _ = self.prep(im)
        z = self.cls.run(["logits"], {"input": xc})[0][0] / self.cfg["temperature"]          # temperature-scaled logits
        p = np.exp(z - z.max()); p /= p.sum(); k = int(p.argmax())
        return dict(arr=arr, cls=self.classes[k], k=k, conf=float(p[k]), probs={c: float(v) for c, v in zip(self.classes, p)}, calibration=self.band(float(p[k])))

    def segment(self, im):
        arr, _, xs = self.prep(im)
        pm = 1 / (1 + np.exp(-self.seg.run(["mask_logits"], {"input": xs})[0][0, 0]))
        mask = pm > self.cfg["seg_thr"]
        return dict(arr=arr, mask=mask, px=int(mask.sum()), pct=float(mask.mean() * 100))

    def explain(self, im, k=None, layer=5):
        """Grad-CAM with PyTorch (needs gradients, so not ONNX). Returns a 256x256 map in [0,1]."""
        import torch
        from src.classification.model import build_efficientnet_b0
        from src.xai.methods import MultiGradCAM
        if self._torch is None:
            m = build_efficientnet_b0(pretrained=False); m.load_state_dict(torch.load(self.torch_ckpt, map_location="cpu")["model"]); self._torch = m.eval()
        arr, xc, _ = self.prep(im)
        mc = MultiGradCAM(self._torch, {"L": self._torch.features[layer]})
        cams, logits, _ = mc(torch.from_numpy(xc), target=k); mc.remove()
        return arr, cams["L"][0].numpy()

    @staticmethod
    def overlay_png(arr, heat=None, mask=None):
        import matplotlib
        rgb = np.repeat(arr[..., None], 3, 2).astype(np.float32) / 255
        if heat is not None: rgb = 0.55 * rgb + 0.45 * matplotlib.colormaps["jet"](heat)[..., :3]
        if mask is not None:
            from scipy import ndimage
            edge = mask & ~ndimage.binary_erosion(mask); rgb[edge] = [0, 1, 0]
        return _b64((rgb * 255).astype(np.uint8))
