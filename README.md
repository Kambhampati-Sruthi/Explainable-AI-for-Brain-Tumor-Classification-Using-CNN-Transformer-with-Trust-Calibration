# Explainable AI for Brain Tumor Classification and Segmentation

Upload a single 2D brain MRI slice and get:

1. a **classification** (glioma, meningioma, pituitary tumor or no tumor) with a calibrated confidence,
2. a **segmentation** (tumor outline, area and bounding box),
3. an **explanation** (Grad-CAM heat-map showing where the classifier looked),
4. a **trust level** (HIGH / MEDIUM / LOW) built from confidence, classifier-segmenter agreement and heat-map consistency.

The models were trained in a Kaggle notebook. This repository contains the code, the trained checkpoints, the evaluation results and a FastAPI web app that runs everything on a CPU.

> **Research prototype. Not a medical device. Not for clinical use.** Do not upload patient-identifiable data. Every result must be reviewed by a qualified clinician.


<!-- [Result view] -->

---

## Contents

1. [Features](#1-features)
2. [Results](#2-results)
3. [How it works](#3-how-it-works)
4. [Tech stack](#4-tech-stack)
5. [Project structure](#5-project-structure)
6. [Dataset and data-integrity safeguards](#6-dataset-and-data-integrity-safeguards)
7. [How the models were trained](#7-how-the-models-were-trained)
8. [Quick start](#8-quick-start)
9. [Using the web app](#9-using-the-web-app)
10. [HTTP API reference](#10-http-api-reference)
11. [Explainability](#11-explainability)
12. [Calibration and trust level](#12-calibration-and-trust-level)
13. [Integrity and safety safeguards](#13-integrity-and-safety-safeguards)
14. [Reproducing the results](#14-reproducing-the-results)
15. [Deployment](#15-deployment)
16. [Troubleshooting](#16-troubleshooting)
17. [Limitations](#17-limitations)
18. [Future work](#18-future-work)
19. [References and acknowledgements](#19-references-and-acknowledgements)

---

## 1. Features

| Capability | What you get |
|---|---|
| Classification | 4 classes, temperature-scaled probabilities, calibration band, uncalibrated probabilities on hover |
| Segmentation | Tumor outline and fill, area in pixels and percent of the slice, bounding box, centre |
| Explainability | Grad-CAM from the `pytorch-grad-cam` library and from the project's own implementation, at three network depths, with opacity, swipe and zoom controls |
| Method comparison | A tab that shows several CAM methods and several depths for the same image |
| Trust assessment | HIGH / MEDIUM / LOW with the three checks behind it shown explicitly |
| Reports | Download a PNG report and a JSON file of each result |
| Input validation | File type, size, pixel count, blank or tiny images, 16-bit images, colour photos, non-square images |
| Model integrity | SHA-256 check of the model files at start-up, shown in the page footer |
| Privacy | Images are processed in memory and never written to disk; responses are marked `no-store` |

---

## 2. Results

All numbers come from the held-out **test set of 1,000 images**, which was not used for training, model selection or calibration. The "de-duplicated" view removes 101 test images that have a near-identical twin in the training set (see [section 6](#6-dataset-and-data-integrity-safeguards)).

### 2.1 Classification (EfficientNet-B0)

| Model | Accuracy (official, n=1000) | Macro-F1 | Accuracy (de-duplicated, n=899) | Macro-F1 |
|---|---|---|---|---|
| Seed 0 (**primary**, used by the app) | 98.70% | 0.9877 | 98.55% | 0.9857 |
| Seed 1 | 98.80% | 0.9896 | 98.67% | 0.9875 |
| Seed 2 | 98.50% | 0.9860 | 98.33% | 0.9834 |
| 3-seed ensemble | 98.70% | 0.9878 | 98.55% | 0.9857 |

Primary model, official test set: 95% bootstrap confidence interval for accuracy 98.0% to 99.4%; AUC about 0.9995.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.992 | 0.976 | 0.984 |
| meningioma | 0.974 | 0.990 | 0.982 |
| no tumor | 0.986 | 1.000 | 0.993 |
| pituitary | 0.997 | 0.987 | 0.992 |

The primary model makes 13 errors in 1,000 images (5 glioma called meningioma, 3 pituitary called meningioma, 2 meningioma called "no tumor", 1 glioma called pituitary, 1 meningioma called glioma, 1 pituitary called glioma). No healthy slice was called a tumor. All 13 errors also lie in the de-duplicated set, so removing duplicates did not hide any mistakes.

### 2.2 Segmentation (U-Net++ with EfficientNet-B0 encoder)

| View | Dice | 95% CI | IoU | Median boundary error (HD95) |
|---|---|---|---|---|
| Official (860 tumor images) | 0.869 | 0.856 to 0.881 | 0.801 | 3.2 px |
| De-duplicated (759 tumor images) | 0.858 | 0.843 to 0.870 | 0.787 | 3.4 px |

| Tumor class (de-duplicated) | Mean Dice | Images with Dice below 0.5 |
|---|---|---|
| meningioma | 0.941 | 0.5% |
| pituitary | 0.879 | 2.7% |
| glioma | 0.764 | 12.6% |

The segmenter draws no false region on any of the 140 healthy test slices. Model cost: 6.76 M parameters, 13.8 GFLOPs, about 3.5 ms per image on a GPU and about 148 ms on a CPU.

### 2.3 Calibration

Temperature scaling fitted on the validation set gives **T = 1.318**. Expected calibration error on the test set fell from 0.0100 to 0.0064.

### 2.4 Trust level on the test set (de-duplicated, n=899)

| Trust level | Images | Errors |
|---|---|---|
| HIGH | 174 | 0 (100% accurate) |
| MEDIUM | 702 | 6 |
| LOW | 23 | 7 |

No error fell into HIGH. HIGH is reserved for cases where confidence, classifier-segmenter agreement and heat-map consistency all pass; most cases are MEDIUM, which means "review carefully". Images with calibrated confidence of 0.95 or higher are 99.65% accurate (n=854); the rest are 77.8% accurate (n=45).

---

## 3. How it works

```
 MRI slice (PNG/JPEG/BMP/TIFF/WebP)
        |
        v
 [Validation] type, size, pixels, blank/tiny, 16-bit -> 8-bit, colour warning
        |
        +--> [Classifier input: resize "stretch" to 256x256, ImageNet normalisation]
        |         |
        |         v
        |    EfficientNet-B0 --> logits --> / T (1.318) --> softmax --> class + calibrated confidence
        |                           |
        |                           +--> Grad-CAM (library or own) on features[3|5|8] for the predicted class
        |
        +--> [Segmenter input: resize "pad" to 256x256, ImageNet normalisation]
                  |
                  v
             U-Net++ --> sigmoid --> threshold 0.5 --> tumor mask --> re-mapped into the classifier frame
                                                              |
        +-----------------------------------------------------+
        v
 [Trust] confidence + (tumor class <=> region > 50 px) + (share of Grad-CAM attention inside the region)
        |
        v
 JSON response: prediction, trust checks, segmentation stats, explanation, 512x512 aligned layers, report PNG
        |
        v
 Browser viewer (heat-map, fill, outline, opacity, swipe, zoom, compare tab, downloads)
```

* The classifier was trained on **stretch**-resized images and the segmenter on **pad**-resized images. The app uses the right policy for each model (`cls_resize_policy` and `seg_resize_policy` in `models/deploy/deploy_config.json`) and re-maps the mask into the classifier frame, so heat-map, mask and overlay line up even for non-square images.
* The trust level always uses the project's own Grad-CAM on `features[8]`, the map its thresholds were set with, even when another method is displayed.

---

## 4. Tech stack

| Layer | Technology | Role |
|---|---|---|
| Models | PyTorch, torchvision | EfficientNet-B0 classifier and U-Net++ segmenter (ImageNet-pretrained, fine-tuned), loaded from `.pt` checkpoints |
| Explainability | `pytorch-grad-cam`, own Grad-CAM (`src/xai/methods.py`), Captum | Heat-maps; Guided Grad-CAM, Integrated Gradients and Occlusion are also implemented in `src/xai` |
| Image processing | OpenCV, Pillow, NumPy | Reading, resizing or padding, overlay layers |
| Plots and statistics | Matplotlib, SciPy, pandas, scikit-learn | Colormaps, report figure, evaluation tables and metrics |
| Backend | FastAPI, uvicorn, python-multipart | Serves the page and the JSON API |
| Request logic | Plain Python (`webapp/service.py`) | Validation, response building, checksum verification |
| Front end | Plain HTML, CSS and JavaScript | Viewer and controls; no framework, no build step, no external CDN |
| Export | ONNX, ONNX Runtime, TorchScript | `models/deploy/*.onnx` and `*_ts.pt` exports of both models |
| Training | Kaggle notebook (GPU, mixed precision) | Training and evaluation |

---

## 5. Project structure

```
BrainTumor_XAI_Project/
|-- README.md
|-- MODEL_CARD.md                  model card
|-- configs/                       selected settings (JSON): preprocessing, final classifier and segmenter,
|                                  segmentation_primary.json, calibration.json, ...
|-- data/
|   `-- splits.csv                 the split file used for training, validation and testing
|-- docs/                          methodology and results notes
|-- outputs/
|   |-- tables/                    result tables, per-image predictions, test logits
|   `-- figures/                   figures used in the report
|-- models/
|   |-- final_s0/1/2_best.pt       classifier checkpoints (3 seeds)
|   |-- segfinal_s0/1/2_best.pt    segmenter checkpoints (3 seeds)
|   `-- deploy/
|       |-- classifier_best.pt     primary classifier (identical to final_s0_best.pt)
|       |-- classifier.onnx, classifier_ts.pt, segmenter.onnx, segmenter_ts.pt
|       |-- deploy_config.json     temperature, thresholds, classes, resize policies, disclaimer
|       `-- SHA256SUMS.txt         recorded checksums of the model files
|-- src/                           model, training, preprocessing, XAI, calibration and evaluation code
|-- local_app/                     pipeline: loads the checkpoints; classify, segment, explain, compare
|-- webapp/                        FastAPI server, request logic, rendering, static front end, launcher
|-- tests/                         unit tests
`-- requirements-app.txt           runtime dependencies of the web app
```

---

## 6. Dataset and data-integrity safeguards

Data: BRISC2025, 2D brain MRI slices with class labels and tumor masks (Kaggle `briscdataset/brisc2025`).

* The manifest lists 15,586 files: 10,793 images and 4,793 masks.
* Classification: 6,000 labelled images, split into **4,002 train**, **998 validation** (group-aware, carved from the official training set) and **1,000 test** images.

| Class | Train | Validation | Test |
|---|---|---|---|
| glioma | 900 | 247 | 254 |
| meningioma | 1,051 | 278 | 306 |
| no tumor | 872 | 195 | 140 |
| pituitary | 1,179 | 278 | 300 |

* The segmentation test set has 860 tumor images (healthy images have no mask).

Safeguards built into the study:

1. **File integrity.** Every file was checked against its SHA-256 hash and every image was fully decoded.
2. **Duplicate-aware evaluation.** 98 groups of near-duplicate images cross the train/test boundary; 101 test images have a twin in training. Results are therefore reported on both the official test set (1,000) and a de-duplicated one (899 classification, 759 segmentation), and the validation split is group-aware so duplicates cannot leak between training and validation.
3. **Shortcut checks.** Classes differ in image format and brightness, and a model that only sees such metadata can already score well above chance. The study counters this with resize/crop/zoom augmentation and a format-shift test (zoom-out, side-crop, squeeze, mirror) to confirm the classifier does not rely on format alone.

---

## 7. How the models were trained

Both networks use **transfer learning with staged fine-tuning** from an ImageNet-pretrained EfficientNet-B0, trained on a GPU with mixed precision (float16). The best checkpoint is chosen on the **validation** set; the test set was used only for final evaluation.

**Input pipeline.** Grayscale slice, resized to 256 x 256 (stretch for the classifier, pad for the segmenter), copied to 3 channels and normalised with the ImageNet mean and standard deviation. Training augmentation (applied identically to image and mask): horizontal flip, rotation/shift/scale, crop-and-pad, contrast and brightness, blur, down-then-up sampling, Gaussian noise.

| | Classifier | Segmenter |
|---|---|---|
| Architecture | EfficientNet-B0, new head (dropout 0.2, linear 1280 to 4) | U-Net++ (nested skips), EfficientNet-B0 encoder, new decoder |
| Loss | Cross-entropy | Binary cross-entropy + Dice (healthy slices included as negatives) |
| Optimizer | AdamW, weight decay 0.01 (none on biases and BatchNorm) | AdamW, weight decay 0.01 |
| Learning rates | backbone 1e-4, head 1e-3 | encoder 1e-4, decoder 1e-3 |
| Phase 1 | 3 epochs, only the head learns | 2 epochs, only the decoder learns |
| Phase 2 | up to 25 epochs, all layers, 1-epoch warm-up then cosine decay | up to 40 epochs, all layers, warm-up then cosine decay |
| Other | batch size 32, gradient clipping 1.0 | batch size 32 |
| Early stopping | patience 6 on validation macro-F1 | patience 8 on validation Dice |
| Final seeds | 0, 1, 2 (primary: seed 0, validation macro-F1 0.9925) | 0, 1, 2 (primary: seed 2, validation Dice 0.8767) |

**Selection.** Six classifier variants (base, dataset normalisation, stretch resizing, class weights, light augmentation, training from scratch) and five segmenter variants (base, plain U-Net, Dice+focal loss, partial fine-tuning, training with healthy slices) were compared with three seeds each. A rule written before the final runs picked the winner on validation scores only. Training with healthy slices cut false tumor regions on healthy slices from about 57% to 0.17% at similar Dice.

**After training.** Temperature scaling is fitted on the validation set; the models are then evaluated on the test set and exported to ONNX and TorchScript. Explanations are computed after training and add nothing to the weights.

---

## 8. Quick start

### 8.1 Prerequisites

* Python 3.11 or 3.12
* About 2 GB of free disk space

### 8.2 Install

```bash
git clone <this repository>
cd <repository folder>
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate

# CPU build of PyTorch; use the CUDA index from pytorch.org if you have an NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
```

### 8.3 Check and run

```bash
python local_app/selfcheck.py     # loads the checkpoints and runs Grad-CAM on a synthetic image
python webapp/run.py              # serves http://127.0.0.1:8000
```

`selfcheck.py` ends with `SELF-CHECK PASSED`; it uses a synthetic image, so its prediction is meaningless. In the browser the status pill should read **Models ready** and the footer **Model files verified (SHA-256)**.

Required files: `models/deploy/classifier_best.pt`, `models/deploy/deploy_config.json`, `models/deploy/SHA256SUMS.txt`, `configs/segmentation_primary.json` and the segmenter checkpoint it names (`models/segfinal_s2_best.pt`).

---

## 9. Using the web app

1. **Upload** a slice by dragging it onto the drop area, clicking to browse, or pasting with Ctrl+V. Accepted: PNG, JPEG, BMP, TIFF, WebP, up to 15 MB. For DICOM, export a slice as PNG first.
2. **Verdict card:** predicted class, calibrated confidence with its calibration band, and the trust level with a one-line summary. Buttons download a **PNG report** and a **JSON** file.
3. **Viewer:** toggle the heat-map, region fill and outline; change heat-map opacity; use **Reveal overlays (swipe)** to wipe between the plain image and the overlays; zoom and move the mouse to pan.
4. **Controls:** change the explanation method, the network layer (`features[3]`, `[5]`, `[8]`) or the implementation (library or the project's own Grad-CAM). The analysis re-runs automatically.
5. **Class probabilities** (calibrated; hover for uncalibrated), **Trust checks** (each with its rule and a pass / warn / fail mark), **Tumor region** (area, bounding box, centre).
6. **Compare methods** tab: the same image explained by several methods and by Grad-CAM at three depths. A method that fails is reported without breaking the page.
7. **Details** tab: technical details and the full JSON (without image data).

Warnings appear above the viewer, for example: colour photo, low resolution, non-square image, 16-bit image rescaled, very small region, or classifier-segmenter disagreement.

---

## 10. HTTP API reference

Interactive docs are served at `/docs`.

### `GET /api/health`

Returns `ready`, `error`, `integrity` (`status`: `verified`, `mismatch` or `unknown`, plus per-file results), `device`, `methods`, `layers`, `max_mb`, `classes` and `disclaimer`.

### `POST /api/analyze` (multipart form)

| Field | Values | Default |
|---|---|---|
| `file` | image file | required |
| `method` | an entry of `methods` from `/api/health` | `Grad-CAM` |
| `layer` | `3`, `5`, `8` | `8` |
| `backend` | `lib` (pytorch-grad-cam) or `own` (project Grad-CAM, Grad-CAM only) | `lib` |

Response keys: `prediction` (class, confidence, calibration band, temperature, calibrated and uncalibrated probabilities), `trust` (level, summary, checks), `segmentation` (region found, area, bounding box, centroid, disagreement flag), `explanation` (method, layer, backend, share of attention inside the mask, library-vs-project correlation), `text`, `warnings`, `images` (base64 PNGs: `base`, `heat`, `mask_fill`, `mask_edge`, `panel`; all 512 x 512 and aligned), `meta` (request id, input SHA-256 prefix, input info, runtime, disclaimer).

### `POST /api/compare` (multipart form)

Fields `file` and `layer`. Returns `methods` and `layers`, each a list of `{label, layer, image, error}`.

### Example

```bash
curl -F "file=@slice.png" -F "method=Grad-CAM" -F "layer=8" -F "backend=lib" http://127.0.0.1:8000/api/analyze
```

On Windows PowerShell use `curl.exe`.

### Error codes

| Code | Meaning |
|---|---|
| 400 | Empty upload |
| 413 | File larger than 15 MB, or too many pixels |
| 415 | Not a readable image, or unsupported format |
| 422 | Invalid parameters, image too small (under 64 px) or almost uniform |
| 503 | Models not loaded (the message says why) |
| 500 | Server error (details in the server log; no stack trace is sent to the browser) |

---

## 11. Explainability

* **Own Grad-CAM** (`src/xai/methods.py`): forward hooks plus `torch.autograd.grad` on the layer output, channel weights from averaged gradients, ReLU, bilinear upsampling, division by the maximum. It avoids backward hooks because EfficientNet's SiLU activations are in-place. The trust thresholds were set with this implementation.
* **Library Grad-CAM** (`pytorch-grad-cam`): the same algorithm with its own normalisation (min-max, then resize). For plain Grad-CAM the app shows the correlation between the two maps; expect a value close to 1.
* **Layers:** `features[8]` is the last convolution (8 x 8 map, used for the trust check). `features[5]` and `features[3]` are earlier and finer.
* **Methods in the app:** Grad-CAM, Grad-CAM++ and Layer-CAM by default (edit `CAM_METHODS` in `local_app/pipeline.py` to add or remove methods supported by the library).
* **Also implemented** in `src/xai/methods.py`: Guided Grad-CAM (adapted for SiLU), Integrated Gradients and Occlusion (via Captum).

A heat-map shows where the classifier looked for its predicted class. It is a visual aid, not proof of why a prediction is correct, and it is coarse (8 x 8 at the last layer).

---

## 12. Calibration and trust level

**Calibration.** Logits are divided by the temperature T = 1.318 (fitted on validation) before the softmax. Confidence bands are defined in `deploy_config.json`.

**Trust level.** The thresholds were fixed before the test set was used.

| Check | Passes when |
|---|---|
| Calibrated confidence | at least 0.95 (HIGH), at least 0.80 (MEDIUM) |
| Classifier and segmenter agree | a tumor class is predicted if and only if the segmenter finds a region of more than 50 px |
| Heat-map inside the tumor region | at least 30% of the Grad-CAM attention lies inside the segmented region (only for tumor predictions with a region) |

`LOW` if confidence is below 0.80 or the two models disagree. `MEDIUM` if confidence is below 0.95 or the heat-map check fails. Otherwise `HIGH`. Calibrated confidence is the strongest signal; the other two checks add a consistency view.

---

## 13. Integrity and safety safeguards

* **Model files:** `models/deploy/SHA256SUMS.txt` lists the SHA-256 of the exported `.onnx` and `.pt` files. The server verifies the files that are present at start-up and the page shows a warning on mismatch.
* **Matching preprocessing:** the exact `process_image` function that built the training cache is used for inference, with the resize policy each model was trained on.
* **Aligned layers:** mask, heat-map and overlay are produced in one frame, including for non-square images.
* **One source of truth:** the trust level is computed by one function; the response only describes its checks.
* **Upload validation:** type check by decoding, size limit, pixel limit (decompression bombs), minimum size, blank-image rejection, first frame only for multi-frame files.
* **Privacy:** images are processed in memory and never written to disk; responses carry `Cache-Control: no-store`.
* **One analysis at a time:** a lock protects the Grad-CAM hooks, which are not thread-safe. Run a single worker.

---

## 14. Reproducing the results

* The classification and segmentation results in [section 2](#2-results) can be recomputed without a dataset or GPU from the files in `outputs/tables/` (for example `final_s0_test_logits.npz`, `final_predictions_test.csv`, `final_seg_test_per_image.csv`).
* The full training and evaluation pipeline is in the Kaggle notebook linked at the top; its source files are in `src/` and the selected settings in `configs/`.
* Training needs a GPU and the BRISC2025 dataset (`briscdataset/brisc2025` on Kaggle). GPU training is not perfectly deterministic, so a re-run gives slightly different weights.

---

## 15. Deployment

**Local (default).** `python webapp/run.py` serves on `127.0.0.1:8000`, reachable only from your computer.

**Docker.**

```bash
docker build -t brain-tumor-xai .
docker run -p 8000:7860 brain-tumor-xai       # then open http://127.0.0.1:8000
```

The container listens on the port in the `PORT` environment variable (default 7860) and binds `0.0.0.0`.

**Hosting.** The app needs a CPU and about 2 GB of RAM. When hosting publicly: put it behind HTTPS, consider adding authentication, never accept patient data, keep the research-prototype disclaimers visible, and log requests without storing images. Free hosting tiers may put idle apps to sleep, so the first load can be slow.

---

## 16. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: src` or `local_app` | Run the commands from the repository root. |
| `FileNotFoundError ... checkpoint` | A weight file is missing; the message names it. See the required files in section 8.3. |
| Page says "Models not loaded" | Same as above; the red banner shows the exact error. |
| Red "Model checksum MISMATCH" | A `.pt` or `.onnx` file differs from `SHA256SUMS.txt` (corrupt or replaced). Restore the original file. |
| `No module named 'pytorch_grad_cam'` | `pip install grad-cam` |
| Port already in use | `python webapp/run.py --port 8001` |
| Compare tab is slow | It computes several maps; on a CPU expect several seconds. |
| Low trust on a normal-looking image | Check the warnings above the viewer (colour photo, non-MRI image, low resolution). |

---

## 17. Limitations

* **One public dataset, 2D slices only.** No external or prospective validation; performance on other scanners, sequences or populations is unknown.
* **Small error counts.** With 13 test errors the confidence intervals are about +/- 0.7 percentage points. Do not compare directly with work that uses different splits.
* **Glioma segmentation is weaker** (Dice 0.76) than meningioma (0.94) or pituitary (0.88).
* **Heat-maps are coarse and qualitative.** The last Grad-CAM map is 8 x 8, so small tumors cannot be localised precisely, and a heat-map does not prove the model's reasoning.
* **No clinical validation.** The trust level is a research aid, not a safety guarantee.
* **Pixel sizes are unknown for PNG/JPEG,** so no physical tumor size is reported.
* **Preprocessing must match training.** Other resize or normalisation settings change the predictions.

---

## 18. Future work

* Quantitative evaluation of explanation faithfulness and localisation against random and centre-prior baselines.
* Tune the trust rule on validation data (for example a size-normalised attention score) and evaluate once on test.
* Improve glioma segmentation and report boundary errors per class.
* External validation on a different brain MRI dataset.
* A reader study with clinicians or trained readers.
* Multi-slice or 3D context, and uncertainty estimates for segmentation (for example across seeds).
* Authentication and a hardened container for shared deployments.

---

## 19. References and acknowledgements

* Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
* Jiang et al., *LayerCAM: Exploring Hierarchical Class Activation Maps for Localization*, IEEE TIP 2021.
* Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
* Zhou et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018.
* `pytorch-grad-cam` by Jacob Gildenblat; Captum, the PyTorch model interpretability library.
* Dataset: BRISC2025 on Kaggle (`briscdataset/brisc2025`). Add the dataset's own citation and licence here.
