# Human trust-calibration study protocol (radiologist-style readers)

**Status: protocol only. No human data has been collected.** Design adapted from the Grad-CAM paper's human studies (Selvaraju et al., 2017, Sec. 5.2: relative reliability
rating on a -2..+2 scale) and the automation-bias literature (Parasuraman & Manzey 2010; Lee & See 2004; Bansal et al. 2021; Bucinca et al. 2021; verify citations before use).

## Question
Does showing an attention-map explanation help readers rely on the AI *appropriately* (follow it when it is right, override it when it is wrong)?

## Design
Within-subject, 3 conditions, Latin-square order, one block of 36 cases per condition (108 trials; each case appears once per reader).
- A: AI label + calibrated confidence only.  B: A + Grad-CAM overlay.  C: B + segmentation outline (AI-predicted tumor).
Case sampling (from the de-duplicated TEST set, fixed seed): 75% AI-correct / 25% AI-wrong, stratified by class and by confidence band,
including AI-wrong cases with high confidence and AI-correct cases whose attention is off the tumor.
Participants: radiologists/residents (target N >= 12; power analysis below). Cases are public, anonymised BRISC slices (no patient identifiers).
Procedure per trial: (1) reader gives an independent diagnosis from the image alone, (2) sees the AI output for the condition, (3) gives a final diagnosis and a confidence (0-100%).
This 'initial answer before AI' step isolates reliance from independent skill.

## Measures (per reader and condition)
- Appropriate reliance rate  ARR = [#(final = AI and AI correct) + #(final != AI and AI wrong)] / #trials
- Over-reliance = P(final = AI | AI wrong);  Under-reliance = P(final != AI | AI correct)
- Reliance calibration error  RCE = sum_b (n_b/N) |P(follow AI | AI-confidence band b) - AI accuracy in band b| ;  Trust Alignment Score  TAS = 1 - RCE
- Human confidence calibration: Brier score and ECE of the reader's stated confidence vs their own correctness
- Relative reliability rating of AI A vs AI B explanations (-2..+2), as in the Grad-CAM paper
- Subjective trust (Likert 1-7) and time per case; NASA-TLX optional

## Analysis plan
Primary: ARR, B vs A (paired). Mixed-effects logistic regression of 'followed AI' ~ condition x AI-correct + (1|reader) + (1|case);
secondary: TAS, over-reliance. Paired sign-flip permutation tests, Holm correction, effect sizes with bootstrap CIs. Pre-register hypotheses:
H1: B lowers over-reliance without raising under-reliance. H2: C further improves TAS when AI-wrong cases have off-tumor attention.
## Risks
Explanations can increase over-reliance (Bansal et al. 2021); a cognitive-forcing step (initial answer first) is used for that reason. Ethics approval is required before recruiting readers.
