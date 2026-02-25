# Project 2 – GMM Statistical Baseline

This project implements a **Gaussian Mixture Model (GMM) baseline** to detect hidden market regimes using the engineered features from **Project 1 – Financial Feature Engineering**. The code is organised in a package-based structure aligned with Project 1.

---

## Package structure

| Path | Role |
|------|------|
| `main_gmm.py` | **Entry point**: full pipeline (load → standardise → optional PCA → GMM selection → fit → diagnostics → plots → export). |
| `main.py` | Thin wrapper so `python -m main` still works. |
| `src/models/gmm_engine.py` | Feature loading, standardisation, PCA, GMM fitting, BIC/AIC selection, joblib serialization. |
| `src/analysis/regime_diagnostics.py` | Profile table, persistence (regime duration), transition matrix, export of regime probabilities. |
| `src/visualization/regime_plots.py` | BIC/AIC/Silhouette curves, cluster heatmap, shaded price chart. |

---

## Prerequisites

- Complete **Project 1** and ensure:
  - `PR1-Financial-Feature-Engineering/data/processed/features.csv` exists (14 engineered features + targets).
  - `PR1-Financial-Feature-Engineering/data/raw/raw_panel_target_2010-01-01_2020-12-31.csv` exists (target index price series).
- Use the same Python environment as Project 1 (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`).

---

## Running the pipeline

From the **Project 2** directory, with your venv active:

```bash
python -m main_gmm
```

or, equivalently:

```bash
python -m main
```

Optional arguments (e.g. override paths, enable PCA, change train period):

```bash
python -m main_gmm \
  --features-path "../PR1-Financial-Feature-Engineering/data/processed/features.csv" \
  --raw-price-path "../PR1-Financial-Feature-Engineering/data/raw/raw_panel_target_2010-01-01_2020-12-31.csv" \
  --max-k 6 \
  --train-end "2017-12-31" \
  --use-pca
```

---

## Methodology

### Why GMM as the baseline

- GMM provides a **probabilistic** partition of the feature space into regimes without requiring labels, suitable for discovering latent states (e.g. Bull, Bear, sideways) from the 14 engineered features.
- It is a standard benchmark in the regime-detection literature (e.g. Matuozzo 2023) and gives a clear baseline before more complex models (e.g. HMMs or deep learning).

### Standardization

- **StandardScaler** is fitted **only on the training period** (default: data with index ≤ 2017-12-31), then applied to the full sample. This matches Project 1’s 2010–2017 train split and avoids look-ahead bias.

### Model selection: BIC, AIC, Silhouette

- **BIC (Bayesian Information Criterion)** is the **primary criterion** for choosing the number of regimes \(k\). It penalises complexity more than AIC, which helps avoid overfitting to noise.
- **AIC** is reported alongside BIC; agreement between the two supports the chosen \(k\).
- **Silhouette score** measures cluster separation in the feature space. If the BIC-optimal \(k\) is near a Silhouette peak, regimes are better separated.
- The pipeline fits both **full** and **diag** covariance types per \(k\) and selects the combination \((k, \text{cov type})\) with the best BIC, with an optional “simplicity” rule that can favour \(k=2\) or \(3\) when Silhouette drops at high \(k\).

### Walk-Forward Stability Check (non-stationarity)

- The pipeline runs a **Walk-Forward Stability Check**: it fits the GMM on the **first 60%** of the data and on the **full sample**, then reports whether the optimal \(k\) changes.
- If \(k\) differs between 60% and 100%, the regime structure may be **evolving over time** (non-stationarity), which is a key theme in Matuozzo (2023) and should be discussed in the thesis.

---

## Regime definitions

After a run, regime labels are assigned from the **profile table** (mean return, volatility, and feature means per regime). The logic maps regimes to interpretable names; the exact table depends on your data. Example schema:

| Regime ID | Label | Description (example) |
|-----------|--------|------------------------|
| 0 | Low-Vol Bull | Positive mean return, low volatility. |
| 1 | High-Vol Bull | Positive mean return, higher volatility. |
| 2 | High-Vol Bear | Negative mean return, high volatility. |
| 3 | Mean-Reverting / Stagnant | Mean return near zero; choppy, range-bound. |
| … | Stagnant / Neutral | Other low-signal regimes. |

The actual **Regime Definitions** table for your run is in **`data/processed/gmm_regime_profiles.csv`** (columns include feature means, `mean_return`, `volatility`, and `label`). Use that file to fill the table in your report.

---

## Key findings (to be filled after a run)

Use this section to record results from your dataset.

### Average persistence

- **Persistence** = consecutive days in the same regime (average, median, max per regime).
- Output: printed to console and summarised in the profile/diagnostics. Example: *“Regime 0 (Low-Vol Bull) persists on average ~X days; Regime 2 (High-Vol Bear) ~Y days.”*

### Most likely transitions

- The **transition matrix** \(P_{ij} = P(\text{regime}_{t+1}=j \mid \text{regime}_t=i)\) is saved as **`data/processed/gmm_transition_matrix.csv`**.
- Note the **highest off-diagonal** entries (e.g. Bull → Bear, Bear → Stagnant) and the **diagonal** (regime stability). Example: *“From Regime 1, the most likely next regime is … with probability ….”*

---

## Outputs

- **`data/processed/gmm_model.joblib`** – fitted GMM.  
- **`data/processed/gmm_regime_profiles.csv`** – regime means, volatility, and labels (for Regime Definitions table).  
- **`data/processed/gmm_transition_matrix.csv`** – \(P(\text{regime}_{t+1}=j \mid \text{regime}_t=i)\).  
- **`data/processed/gmm_probs.csv`** – per-day regime probabilities and labels.  
- **`figures/gmm_bic_aic_silhouette.png`** – BIC/AIC/Silhouette vs \(k\).  
- **`figures/gmm_regime_feature_heatmap.png`** – cluster heatmap.  
- **`figures/gmm_regimes_price_colored.png`** – shaded price chart by regime.

Optional with `--use-pca`: **`data/processed/pca_model.joblib`**.
