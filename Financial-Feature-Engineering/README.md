## Project 1 – Financial Feature Engineering

This subproject focuses on reproducing the 14 engineered features used in Matuozzo (2023) for market regime detection, starting from daily index and volatility data.

### Environment setup (Python 3.x + venv)

1. **Create a virtual environment**

```bash
cd Financial-Feature-Engineering
python3 -m venv .venv
```

2. **Activate the environment**

- macOS / Linux:

```bash
source .venv/bin/activate
```

- Windows (PowerShell):

```bash
.venv\\Scripts\\Activate.ps1
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: TA-Lib may require system-level libraries (e.g. `brew install ta-lib` on macOS) before `pip install` succeeds.

### Data fetching entry points

With the environment active, you can fetch index and volatility data directly from a Python session:

```python
from src.data.fetch import fetch_named_indices, fetch_vol_indices

indices = fetch_named_indices(names=("sxxp", "dax"))
vols = fetch_vol_indices(names=("v2x", "vdax", "vix"))
```

This will download data via `yfinance`, write cleaned close-price series into `data/raw/`, and return `pandas` DataFrames ready for feature engineering.

For the full raw panel used in Project 1 (target index + proxies) use:

```python
from src.data.data_loader import fetch_raw_data

panel = fetch_raw_data(target_ticker="^STOXX50E")  # or "^GDAXI"
```

This creates a forward-filled price panel for the target, VSTOXX (`^V2TX`), Gold (`GC=F`), and Bunds (`^FGBL`), plus daily log-returns for target, gold, and bunds.

### Feature engineering overview

Core feature logic lives in `src/features`:

- **Statistical** (`statistical.py` / `feature_engineer.py`):
  - 66-day Shannon entropy on the sign of returns (`entropy_66d`), with the rolling probability **lagged by one day** (`.shift(1)`).
  - 126-day vol-scaled returns (`vol_scaled_return_126d`): return divided by lagged rolling 126-day volatility.
  - 126-day rolling skewness (`skewness_126d`), also lagged one day.
  - Shannon’s entropy is computed on a binary sequence \(S_t\) where \(S_t = 1\) if the daily return is positive and \(S_t = 0\) otherwise, following the paper’s novel feature-engineering design.
- **Sentiment / macro** (`sentiment_macro.py`):
  - Implied-vol index daily change (`implied_vol_change_log_return`).
  - Vol-of-vol on VSTOXX log-returns (`vstoxx_vol_of_vol_66d`), based on a 66-day rolling std lagged one day.
- **Technical** (`technical.py`):
  - Bollinger %B (20d, `bollinger_pct_b_20d`), CMCI/CCI (20d, `cmci_20d`), Stochastic %K (14d, `stoch_k_14d`), RSI (14d, `rsi_14d`), Williams %R (14d, `williams_r_14d`).
- **Momentum** (`momentum.py`):
  - 3‑month (`momentum_63d`) and 12‑month (`momentum_252d`) momentum as cumulative log-returns, with the rolling sum lagged by one day.

All rolling statistics are constructed so that the feature value at time \(t\) only depends on information up to \(t-1\), avoiding look-ahead bias.

To build the main feature panel from a raw data frame:

```python
from src.data.data_loader import fetch_raw_data
from src.features.feature_engineer import build_feature_panel

raw_panel = fetch_raw_data()
features = build_feature_panel(raw_panel)
```

### Diagnostics and stationarity checks

Use `src/diagnostics.py` to audit look-ahead risks and stationarity:

- **Look-ahead scan**:

```python
from src.diagnostics import check_lookahead_in_feature_code
check_lookahead_in_feature_code()
```

- **ADF tests + differencing** (after saving `data/processed/features.csv`):

```bash
python -m src.diagnostics
```

This writes `data/processed/adf_report.csv` (ADF stats for all numeric features) and `data/processed/adf_report_diff.csv` (ADF stats for first‑order percentage-change versions of any non-stationary features).

### Entropy visualisation and regime shifts

The entropy indicator is used to highlight changes in market regimes, similar to Figure 2 in Matuozzo (2023). The helper `plot_entropy_vs_price` in `src/visualize.py` overlays the 66‑day entropy series on top of the index price and annotates major events such as the **2015 ECB Quantitative Easing** announcements and the **2020 COVID shock**, allowing you to visually inspect how persistent trends are reflected in lower entropy.

### Results and stationarity summary

After you have run the full ADF diagnostics (`python -m src.diagnostics`) on your final engineered feature set, you should briefly document which of the 14 features required differencing (e.g. long-horizon momentum or certain level-type series) to achieve stationarity. A concise example entry for your report might look like:

- **Stationary at 5%**: `entropy_66d`, `vol_scaled_return_126d`, `skewness_126d`, `vstoxx_vol_of_vol_66d`, `bollinger_pct_b_20d`, `cmci_20d`, `stoch_k_14d`, `rsi_14d`, `williams_r_14d`, `gold_log_return`, `bund_log_return`.
- **Non-stationary, differenced**: `momentum_63d`, `momentum_252d`, `implied_vol_change_log_return` (used `%Δ` versions in models).

Recording these outcomes (with your actual test results) demonstrates that you not only implemented the diagnostics but also **acted on them** when preparing data for GMM/HMM and deep learning models.

### Final tensors and data for Project 2

The final preparation script in `src/prepare_final_data.py` standardises features and builds the 3D tensors used by downstream models:

```bash
python -m src.prepare_final_data
```

This pipeline:

- Loads `data/processed/features.csv`.
- Applies `StandardScaler` **fit only on 2010–2017 data** and then transforms the full 2010–2020 panel.
- Uses a fixed, explicit list of **14 features** (`FEATURE_COLUMNS_14` in `prepare_final_data.py`) to build 10‑day windows.
- Creates tensors `X_train`, `X_test` with shape `(samples, 10, 14)` and aligned next‑day targets `y_train`, `y_test`.
- Saves them as `.npy` files under `data/processed/`:
  - `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`.

These arrays, together with the ADF reports, form the input to Project 2’s GMM/HMM and deep learning regime-detection models.

