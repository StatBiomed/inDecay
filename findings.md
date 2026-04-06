# Findings

## Codebase Architecture
- Main package: `inDecay/` — analysis functions, model, transformation utilities
- `inDecay/analysis_fn.py` — all metric computation. Key functions:
  - `assessment_recipe_forecast()` — somatic cell evaluation (returns dict with frameshift R², top-k recall, KLD, etc.)
  - `assessment_recipe_IDL_forecast()` → wraps `assessment_recipe_41IDL_forecast()` — indel length distribution metrics
  - `forecast_frameshift(y, pred, indels)` — computes single total frameshift ratio (any indel where size % 3 != 0)
  - `forecast_delratio(y, pred, indels)` — computes total deletion ratio
- `inDecay/transformation.py` — label transforms between 557/912/894 class encodings, frameshift masks
- `qrguide` package provides `tokFullIndel()`, `IndelLen_transform()` etc.

## Current Metrics (as of Oct 2024)
Stored in `results/benchmarking/featv5_perform_Oct15.csv`:
- `KL divergence`, `Top5 events recall`, `Top10 events recall`
- `R2 of Frameshift ratio` — single aggregate, not per-frame
- `Coll_I_Top5`, `Coll_I_Top10` — insertion-collapsed top-k
- `KLD_IDL`, `Top5_IDL`, `Top10_IDL`, `W1-distance_IDL`, `delratio_r2`, `Kendall_tau_IDL`

## Figure 2 Layout (Oct 2024 version)
- 6 panels (2 col × 3 row): Top5_IDL, Top5 events recall, Kendall_tau_IDL, KLD_IDL, R2 of Frameshift ratio, delratio_r2
- Compares 5 models: inDelphi, Lindel, FORECasT, inDecay (ours), replication
- Saved to `results/benchmarking/manuscript_figure_2_6metrics_Oct15.pdf`

## Figure 3 Layout
- Transfer learning: CHO → iPSC and CHO → mESC
- 5 metrics line plots vs. N_sample with full-data baseline dashed line
- Data from `results/Transfer/C500/CHO-{BOB,E14TG2A}/`

## Indel Encoding
- 557-class format: `{start}+{length}` for deletions, `{length}+{nucleotide}` for insertions
- 912-class format: extended version
- 894-class format: U-net model format
- Frameshift transform: binary mask where `indel_length % 3 != 0`

## Data Locations
- Somatic test predictions: `pl_trainer_log/ST_featv5fast_*/` and `/ssd/users/louisayu/inDecay/pretrained/`
- Mouse locus data: `data/mouse/` (49 loci CSVs: `mXxx_SelfTarget.csv`)
- Livestock locus data: `data/livestock/` (12 CSVs)
- Zygote raw data: **NOT FOUND** at expected `zygote/` path — needs clarification
- Benchmarking CSVs: `results/benchmarking/`

## Frame-Specific Breakdown — Feasibility
The existing `tokFullIndel()` function parses indel identifiers and returns `(indel_type, isize, details, muts)`.
Frame can be computed as `isize % 3 ∈ {0, 1, 2}`.
For the ForeCast-format pkl data, `Indel` column contains full identifiers like `D5_L-6C4R4` or `I1_L-2C1R0`.
Implementation is straightforward — extend `forecast_frameshift` to return per-frame sums.

## 61 Features (MLP input)
From `phase2.md`:
- 13 deletion-specific: raw deletion length (l_ld), deletion start site (ss), decay terms
- 7 insertion-specific: insertion length, complementary nucleotide presence
- 41 shared: GC-ratio of gRNA, total MH strength, one-hot 9-bp sequence upstream of PAM

## SHAP Feasibility
- `shap` package needs to be installed
- Model architecture: MLP (need to confirm layers/activation from scripts or saved weights)
- For neural networks, DeepSHAP or GradientSHAP are preferred; KernelSHAP works model-agnostic
- Need to find saved model weights — not found in repo yet, likely at `/ssd/` path

## K562 Frame Breakdown Results (validated 2026-04-05)
- frame0_r2 (in-frame +0): 0.846
- frame1_r2 (+1 frameshift): 0.823
- frame2_r2 (+2 frameshift): 0.835
- Verified: frame ratios sum to 1.0 per oligo ✓

## SHAP Notes
- K562 test oligos (Oligo17460+) NOT in SelfTarget_NewScaffold.fasta
- SHAP uses 6234 available SelfTarget scaffold oligos as background + explanation data
- shap 0.51.0 installed in inDecay conda env
- GradientExplainer on del_regressor (raw logit, before softmax)

## Blocking Questions (resolved)
1. Frame breakdown: which figure panels to add/replace?
2. SHAP: where are trained MLP weights on HPC?
3. Embryo QC: where is the zygote raw data?
4. Embryo QC: what format is the Sanger r² quality — is it already processed?
