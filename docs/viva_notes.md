# Viva preparation: likely questions with the project-specific answers
1. **Why EfficientNet-B0 and U-Net++?** Cost-performance and the project spec; BRISC reports B0 at 0.9920 weighted F1 and B1 at 0.9903 (not separable); in BRISC's Table 3 U-Net++ did NOT beat U-Net (75.3 vs 75.7), so we ran a same-encoder ablation instead of assuming.
2. **Why not Swin-HAFNet?** Heavier; used as a literature benchmark and for ideas (shared backbone, HAF, CBE). Its parameter/FLOP figures are not in the paper.
3. **What is the main methodological finding?** The dataset has class-specific image formats; a metadata-only model reached 61.6% (format only) and 79.7% (plus brightness) accuracy without seeing the tumor.
4. **How did you handle duplicates?** Kept all files; flagged 98 cross-split groups; group-aware validation; report official and de-duplicated test.
5. **Why is the de-duplicated test smaller (899/759)?** 101 classification test images have a twin in train (96 meningioma, 5 pituitary); segmentation shares them (tumor images only).
6. **Why pad instead of stretch/crop?** Preserves geometry and content; stretch distorts shapes; crop lost tumor pixels in 4 of 216 non-square masks. Policy kept as an ablation switch.
7. **Why 256 px?** Mask ceiling: Dice 0.979 at 256 vs 0.982 at 384 (area+threshold); 4x cheaper than 512.
8. **Why area + threshold for masks?** Nearest-neighbour is unreliable at non-integer scale factors (Dice ceiling 0.936 at 224).
9. **Why AdamW, warm-up + cosine, AMP?** Decoupled weight decay; avoids early large steps; float16 on T4 halves memory.
10. **Why macro-F1?** Test class shares differ from train (no_tumor 14% vs 21%).
11. **Why is the test set used only once?** Selections would otherwise be tuned to it; we pre-registered rules on validation.
12. **What is Grad-CAM mathematically?** alpha_k = mean of dy/dA_k; L = ReLU(sum alpha_k A_k); equals CAM for GAP+linear heads.
13. **Why did Captum's Guided Grad-CAM not work out of the box?** EfficientNet uses SiLU; Captum rewires only ReLU; we showed the degeneration numerically and used a SiLU-adapted rule (our adaptation).
14. **Why is Grad-CAM coarse here?** 256 px / 32 = 8x8 grid; we compared layers 8, 5, 3.
15. **Do the heatmaps actually point at tumors?** Measured (energy in mask, pointing game, IoU) against a center-prior baseline; the Swin-HAFNet and Diagnostics papers show this only qualitatively.
16. **What is ECE and why temperature scaling?** Binned |accuracy - confidence|; one scalar fitted on validation NLL; with ~99% accuracy ECE rests on few errors, so we report bootstrap intervals.
17. **What is your trust-calibration contribution?** Calibrated confidence bands + a no-ground-truth consistency signal (Grad-CAM vs predicted mask) + classifier-segmenter agreement, combined into HIGH/MEDIUM/LOW with measured error rates.
18. **Did you run a human study?** No. A full protocol (ARR, TAS, mixed-effects analysis) and a clearly labelled simulation for power; no human claims.
19. **What did explanation-guided debugging change?** See table_7: attention-in-mask, background attention and format sensitivity before/after, paired by seed; negative results are reported.
20. **Why is segmentation gated by the classifier?** The tumor-only segmenter drew tumor on about half of healthy validation scans; gating plus the optional negatives arm address it.
21. **Why are your numbers lower than the Diagnostics paper's 0.9055 Dice?** Different protocol (their random 70/15/15 re-split puts near-duplicate slices across train/test), different resolution; not comparable.
22. **EU AI Act relevance?** Articles 13 (transparency) and 14 (human oversight, automation-bias awareness); medical AI is high-risk via Art. 6(1) when MDR class IIa+. Dates must be checked on EUR-Lex.
23. **Is it clinically usable?** No: single public dataset, 2D slices, no prospective validation; the API states it is a research prototype.
24. **What would you do next?** Subject-level splits, external datasets, near-duplicate detection, multi-task shared backbone as in Swin-HAFNet, real reader study.
