# Model card (research prototype)
- **Intended use:** research and teaching on public data. NOT a medical device; not validated for clinical use; not for diagnosis.
- **Models:** EfficientNet-B0 classifier (4 classes), U-Net++ (EfficientNet-B0 encoder) segmenter. Trained on BRISC2025 (T1 contrast-enhanced, 2D slices).
- **Known limitations:** single public dataset assembled from several sources with class-specific image formats; duplicates across train/test were found and handled by an
  explicit leakage-controlled protocol; no external or prospective validation; no subject-level independence guarantee; coarse Grad-CAM resolution.
- **Metrics:** see README and outputs/tables (official and de-duplicated test sets; intervals and seeds reported).
- **Human oversight:** every output carries a trust level and a disclaimer; a clinician must review all results (cf. EU AI Act Art. 13/14 motivation).
- **Data privacy:** the API does not store uploads. Do not upload patient-identifiable data to any public demo.
