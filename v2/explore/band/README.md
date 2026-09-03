# Workspace band for gpt2-small

`band.py` (paper Fig 28 metrics) and `cka.py` (dictionary similarity, controlling for the dominant
direction).  48 wikitext-103 validation sequences × 128 tokens; dictionary metrics on a 3000-token
vocabulary subsample of the centered J-lens vectors.  CPU, ~2 min + ~4 min.

## Layer-wise statistics (`band.json`)

| L | top-1 agree | top-10 agree | excess kurtosis | autocorr − null | effdim (90% var) |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.5 | +0.001 | 0.13 |
| 1 | 0.00 | 0.01 | 0.6 | -0.000 | 0.16 |
| 2 | 0.00 | 0.01 | 0.6 | +0.000 | 0.19 |
| 3 | 0.00 | 0.01 | 0.4 | +0.002 | 0.20 |
| 4 | 0.00 | 0.02 | 0.5 | +0.004 | 0.22 |
| 5 | 0.00 | 0.03 | 0.4 | +0.002 | 0.22 |
| 6 | 0.01 | 0.04 | 0.4 | +0.007 | 0.26 |
| 7 | 0.02 | 0.10 | 0.4 | +0.014 | 0.30 |
| 8 | 0.04 | 0.13 | 0.5 | +0.021 | 0.35 |
| 9 | 0.11 | 0.29 | 0.6 | +0.023 | 0.43 |
| 10 | 0.24 | 0.51 | 0.5 | +0.020 | 0.50 |
| 11 | 1.00 | 1.00 | 0.4 | +0.008 | 0.66 |

* **top-k agreement** of the lens readout with the model's own next-token prediction: ≈0 through
  L6, 2/4/11% at L7/8/9, then 24% at L10 and 100% at L11 (the lens target).
* **autocorrelation** of lens top-1 across adjacent positions, minus the shuffled null: ≈0 through
  L5, then +.008 (L6), +.013, +.020, +.024 (L9), +.021 (L10), +.008 (L11).
* **excess kurtosis** of the readout logits is flat (0.4–0.6) at every layer.  The paper's kurtosis
  rise through the workspace band does not occur.
* **effective dimensionality** rises smoothly (.13 → .50 over L0–10) with no onset.

## Dictionary similarity (`cka.json`)

Plain linear CKA between layer dictionaries is ≥0.94 for every pair in L0–10 (L11 ≈ 0.45).  That is
one direction: after mean-centering, the top principal component still carries 26–37% of each
dictionary's variance at L0–10 (2% at L11).  Removing the top 5 PCs, or taking the mean squared
canonical correlation between top-50 subspaces, gives a smooth diagonal drift and no block:

* adjacent-layer similarity, top-5 PCs removed (L0↔1 … L10↔11): 0.95 0.95 0.96 0.98 0.96 0.96 0.96 0.93 0.93 0.95 0.92
* adjacent-layer similarity, mean CCA r=50:                     0.89 0.88 0.92 0.92 0.90 0.89 0.88 0.85 0.83 0.78 0.76
* two-apart, mean CCA r=50 (L0↔2 … L9↔11):                      0.82 0.82 0.87 0.84 0.85 0.83 0.77 0.75 0.69 0.63

Similarity decays with layer distance at every depth and, if anything, decays faster late (L9↔10
0.78) than early (L2↔3 0.92).  The paper's shared-subspace block across the workspace band is absent.

## Choice

Band = layers **7–9**: autocorrelation above the null from L6, top-1 agreement with the output still
<15% at L9, L10 clearly motor.  L6 and L10 are both arguable; nothing in the structural statistics
picks out 7–9 sharply, and the post should say so rather than present the band as given.
