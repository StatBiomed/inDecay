# Progress Log

## Session: 2026-04-05

### Status: IMPLEMENTATION COMPLETE

Explored codebase, read phase2.md, Benchmarking.ipynb, Figure2.ipynb, Figure3.ipynb, analysis_fn.py, transformation.py, models.py, PATH.py, zygote.py.

Created task_plan.md and findings.md.

### Blocking Questions for User

**Q1 (Task 1 — Frame Breakdown, Figure Layout):**
- The current Figure 2 has 6 panels comparing 5 models. Should the 3 per-frame R² metrics (R² for +1 fs, +2 fs, in-frame) be:
  - (a) Added as 3 new panels in Figure 2 (making it 9 panels)?
  - (b) A new separate supplementary figure?
  - (c) Replace the existing "R² of Frameshift ratio" panel with a breakdown?

**Q2 (Task 1 — Which models to include):**
- Should the per-frame breakdown compare all 5 models (inDelphi, Lindel, FORECasT, inDecay, replication), or just inDecay?

**Q3 (Task 2 — SHAP Model Weights):**
- Where are the trained MLP weight files on HPC? The path `/ssd/users/louisayu/inDecay/pretrained/` isn't accessible here. Are they in `pl_trainer_log/` or elsewhere?
- The SHAP analysis requires running inference on a background dataset — is the training data available locally (HPC)?

**Q4 (Task 2 — SHAP Output):**
- Should SHAP results go into an existing figure (Figure 4/5?) or a new supplementary figure?
- Should we show SHAP for all 5 cell types or pick one representative?

**Q5 (Task 3 — Embryo QC Data):**
- `zygote/` directory exists but has no `mouse/` or `livestock/` subdirectories. Where is the raw embryo data (Sanger ab1 files or processed CSVs)?
- Is the Sanger r² quality already computed somewhere, or does it need to be extracted from ab1 files?
- Is the attrition data (N_initial zygotes, N_blastocyst, N_edited) in a spreadsheet/CSV already?

**Q6 (Task 3 — Scope):**
- The current `data/mouse/` has 49 files, but phase2.md says 52 mouse sites. Similarly `data/livestock/` has 12 but phase2.md says 12. Are any mouse loci missing data?

### Next Steps (after answers)
- Implement `forecast_frame_specific_ratios()` in `analysis_fn.py`
- Run updated Benchmarking.ipynb to generate new CSV
- Update Figure2.ipynb based on layout decision
- Set up SHAP notebook once model paths confirmed
- Build embryo QC table once data location confirmed
