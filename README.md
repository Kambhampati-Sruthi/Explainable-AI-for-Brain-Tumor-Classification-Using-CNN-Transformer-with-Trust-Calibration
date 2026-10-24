# Explainable AI for Brain Tumor Classification and Segmentation (EfficientNet-B0 + U-Net++) with Trust Calibration

> **Research prototype. Not a medical device. Not for clinical use.**

![architecture](outputs/figures/fig_11_architecture.png)

## What this is
A clinical-decision-support prototype on **BRISC2025** that (1) classifies glioma / meningioma / pituitary / no tumor (EfficientNet-B0),
(2) segments tumors (U-Net++ with an EfficientNet-B0 encoder), (3) explains predictions (Grad-CAM, Guided Grad-CAM, Integrated Gradients, Occlusion via Captum),
(4) reports calibrated confidence and a trust level, and (5) serves everything through FastAPI + ONNX.

## Headline results (test set; official / de-duplicated)
| Task | Metric | Official | De-duplicated |
|---|---|---|---|
| Classification | macro-F1 | 0.9878 | 0.9855 |
| Segmentation | mean Dice | 0.8685 | 0.8577 |
Numbers are generated from `outputs/tables/`. The de-duplicated set removes 101 test images with a pixel-identical twin in train (98 duplicate groups across splits).

## Leakage-controlled protocol (the part most repositories skip)
SHA-256 checks, decoded-pixel duplicate audit, group-aware train/validation split, a metadata-only shortcut baseline, format-shift interventions, and a single test pass per task.

## Repository
`data/` (splits, cache) | `notebooks/` | `src/{preprocessing,classification,segmentation,xai,trust,evaluation,deployment,utils}` | `models/` | `outputs/` | `api/` | `configs/` | `tests/` | `docs/`

## Run
```bash
pip install -r requirements.txt
CLS_CKPT=models/deploy/classifier_best.pt uvicorn api.main:app --port 8000   # UI at /ui/
pytest -q tests
```
Endpoints: `/health`, `/predict`, `/segment`, `/explain`, `/analyze`.

## Limitations
Single public dataset assembled from several sources (class-specific image formats were found; see the shortcut analysis); exact-duplicate removal only (near-duplicates and subject-level
independence are not guaranteed by the dataset authors); 2D slices; no external validation; human trust study is a protocol (no human data); Grad-CAM resolution is coarse (8x8 at the last layer).
EU AI Act (Regulation 2024/1689): transparency (Art. 13) and human oversight (Art. 14) motivate the explanation and trust features; check EUR-Lex for current application dates.

## Citation
Fateh et al., *BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification*, Sci. Data 13:361 (2026).
