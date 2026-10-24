# Methodology and Results (IEEE-style skeleton; every number is read from outputs/tables)

## I. Dataset and audit
BRISC2025: 6,000 classification images (5,000/1,000) and 4,793 image-mask pairs (3,933/860), T1 contrast-enhanced, from public sources. We verified 15,586 file checksums, found 98 cross-split
exact-duplicate groups (205 classification records, 410 task records), 13 plane-label conflicts, class-specific image formats (Cramer's V 0.58 for storage mode, 0.66 for resolution), and
the segmentation set to be byte-identical to the tumor part of the classification set.
## II. Protocol
Official split kept; validation split group-aware and stratified; results reported on the official and the de-duplicated test sets; test touched once per task; selections made on validation under pre-declared rules.
## III. Preprocessing
Grayscale, pad-to-square, 256 px (area/linear), masks binarised at 127 and shrunk with area + threshold; joint image-mask augmentation including shortcut-breaking crop-pad.
## IV. Models
EfficientNet-B0 (staged fine-tuning, AdamW, warm-up + cosine, AMP). U-Net++ with EfficientNet-B0 encoder (nested skip pathways, Dice-based loss). Swin-HAFNet is discussed, not trained.
## V. Results
Classification macro-F1 0.9878 official / 0.9855 de-duplicated. Segmentation Dice 0.8685 / 0.8577 (HD95 median 3.2 px at 256 px).
Compare with BRISC Table 3/4 only with the protocol caveats (their resizing, thresholds and IoU averaging are not specified).
## VI. Explainability, trust and debugging
Insert: fig_5_explanations, table_8_xai_eval, fig_6_calibration, table_6_calibration, trust-level table, table_7_before_after.
## VII. Threats to validity
Shortcut/format bias, near-duplicates, small number of errors (wide intervals), single dataset, simulated (not human) reader data, coarse Grad-CAM grid.
