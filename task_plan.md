# Task Plan: Phase 2 — Improved Utility & Interpretability

## Objective
Implement the three reviewer-requested improvements from `phase2.md` and produce updated figures for the manuscript.

---

## Phase 1: Clarification
Gather answers to blocking questions before writing any code.

**Status:** COMPLETE — 2026-04-05

---

## Phase 2: Granular Frameshift Breakdown (Task 1)
**Goal:** Break the single frameshift R² metric into per-reading-frame ratios: +1, +2, and in-frame (+0).

### Steps
- [ ] Add `forecast_frame_specific_ratios(y, pred, indels)` to `inDecay/analysis_fn.py`
  - Returns per-frame probability sums: (y_frame0, y_frame1, y_frame2, pred_frame0, pred_frame1, pred_frame2)
- [ ] Add `assessment_recipe_frame_breakdown(Y_lookup, pred_lookup)` that aggregates per-frame R² across all oligos
- [ ] Update `notebooks/Benchmarking.ipynb` to run new metrics and save to a new CSV (e.g., `results/benchmarking/frame_breakdown_MMDD.csv`)
- [ ] Update `notebooks/Figure2.ipynb` to add 3 new subplot panels (R² for +1, +2, in-frame) — location TBD after clarification

**Depends on:** Data in pkl prediction files, clarification on figure layout

---

## Phase 3: SHAP Feature Importance (Task 2)
**Goal:** Apply SHAP to the trained MLP over 61 information-dense features; generate summary plots.

### Steps
- [ ] Locate and load trained MLP weights (need path clarification)
- [ ] Extract 61 features from training data (13 del-specific + 7 ins-specific + 41 shared)
- [ ] Set up `shap.Explainer` (KernelSHAP or DeepSHAP depending on model architecture)
- [ ] Compute SHAP values across the test set
- [ ] Generate SHAP summary/beeswarm plots grouped by feature category
- [ ] Save figure as `results/benchmarking/SHAP_summary_MMDD.pdf`

**Depends on:** Model weight file location, feature extraction pipeline

---

## Phase 4: Embryo QC Table (Task 3)
**Goal:** Comprehensive per-locus table for 52 mouse + 12 livestock loci with attrition, success, and Sanger quality.

### Steps
- [ ] Locate raw embryo data (zygote directory, Sanger ab1 files, or CSV)
- [ ] Write aggregation script to compute: N_initial, N_survived, N_edited, edit_rate, sanger_r2 per locus
- [ ] Produce table as CSV + LaTeX-ready format
- [ ] Add supplementary figure / table to notebook

**Depends on:** Zygote/embryo data location (not found at default path)

---

## Phase 5: Figure Assembly
- [ ] Update `Figure2.ipynb` with frame breakdown panels
- [ ] Create new figure for SHAP (new notebook or existing)
- [ ] Create new table figure for embryo QC
- [ ] Export final PDFs

---

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-05 | Planning phase started | Reviewer requests phase 2 |
| 2026-04-05 | Frame breakdown: add 3 panels to Figure 2 + new supp figure | User Q1 answer |
| 2026-04-05 | SHAP on K562 pretrained model: `pretrained/K562_featv5_pretrained.ckpt` | User Q2+Q3 answer |
| 2026-04-05 | Task 3 (embryo QC) skipped — data not on HPC; checklist generated instead | User Q4 answer |
