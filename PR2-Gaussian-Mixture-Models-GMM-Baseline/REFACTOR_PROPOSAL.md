# Project 2 Refactor Proposal: Match Professional Modular Structure

## Current State vs Specification

| Spec requirement | Current state | Proposed change |
|------------------|---------------|-----------------|
| **`src/models/gmm_engine.py`** (fit GMM, BIC/AIC selection, joblib) | `src/models/gmm_model.py` has all of this | **Rename** `gmm_model.py` → `gmm_engine.py` and update all imports. |
| **`src/analysis/regime_diagnostics.py`** (Persistence, Transition Matrix, Profile Table) | `src/analysis/regime_stats.py` has `compute_profiles`, `compute_persistence`, `compute_transition_matrix`, `export_probs` | **Rename** `regime_stats.py` → `regime_diagnostics.py` and update imports. |
| **`src/visualization/regime_plots.py`** (BIC curves, Cluster Heatmap, Shaded Price Chart) | Already present with `plot_bic_aic_silhouette`, `plot_regime_heatmap`, `plot_regimes_on_price` | **No structural change.** Imports will switch to `gmm_engine`. |
| **`main_gmm.py`** (clean entry point importing from `src/`) | Pipeline lives in `main.py` | **Add** `main_gmm.py` with the full pipeline; keep `main.py` as a thin wrapper so `python -m main` still works. |
| **Walk-Forward Stability** (60% vs 100%) | Implemented as **50%** vs 100% | **Change** to **60%** of data for the “first portion” to align with spec and Matuozzo non-stationarity theme. |
| **Standardization** (scaler fit on train only, 2010–2017) | Already correct: `standardise_features(df, train_end)` fits on `idx <= train_end` | **No change.** |
| **README: Methodology** | Present (BIC/AIC/Silhouette, walk-forward) | **Keep**; optionally tighten wording. |
| **README: Regime Definitions table** | Missing | **Add** a table (Regime ID | Label | Description). |
| **README: Key Findings** (persistence, transitions) | Missing | **Add** a section with placeholders / instructions to record findings. |

---

## File Change Summary

1. **Rename** `src/models/gmm_model.py` → `src/models/gmm_engine.py`  
   - No logic change; update references in `main.py`, `main_gmm.py`, `regime_diagnostics.py`, `regime_plots.py`.

2. **Rename** `src/analysis/regime_stats.py` → `src/analysis/regime_diagnostics.py`  
   - No logic change; update its import from `gmm_model` to `gmm_engine`; update `main.py` / `main_gmm.py` to import from `regime_diagnostics`.

3. **Add** `main_gmm.py` at project root  
   - Move the full pipeline from `main.py` into `main_gmm.py` (imports from `src.models.gmm_engine`, `src.analysis.regime_diagnostics`, `src.visualization.regime_plots`).  
   - Implement walk-forward check using **first 60%** vs full sample; print whether optimal \(k\) changes.

4. **Slim** `main.py`  
   - Replace with a thin wrapper that calls `main_gmm.main()` so existing `python -m main` and any references to `main` still work.

5. **Update** `scripts/gmm_regime_analysis.py`  
   - Call the new entry point: `from main_gmm import main` (then `main()`).

6. **Extend** `README.md`  
   - **Regime Definitions**: table describing detected regimes (e.g. Regime 0 = Low Vol/Bull).  
   - **Key Findings**: subsection for average persistence and most likely transitions (template/placeholder to fill after a run).  
   - Mention running via `python -m main_gmm` (and that `python -m main` is supported).

---

## Resulting Structure

```
PR2-Gaussian-Mixture-Models-GMM-Baseline/
├── main_gmm.py              # Entry point: full pipeline
├── main.py                  # Wrapper: from main_gmm import main; main()
├── scripts/
│   └── gmm_regime_analysis.py   # Thin wrapper → main_gmm.main()
├── src/
│   ├── models/
│   │   └── gmm_engine.py        # Fit GMM, BIC/AIC, PCA, joblib
│   ├── analysis/
│   │   └── regime_diagnostics.py # Profiles, persistence, transition matrix, export_probs
│   └── visualization/
│       └── regime_plots.py       # BIC curves, heatmap, shaded price chart
├── data/processed/
└── figures/
```

No new dependencies; behavior preserved except walk-forward using 60% instead of 50%.
