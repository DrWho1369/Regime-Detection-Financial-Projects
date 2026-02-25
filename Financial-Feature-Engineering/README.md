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

With the environment active, you can fetch index data directly from a Python session:

```python
from src.data.fetch import fetch_named_indices, fetch_vol_indices

indices = fetch_named_indices(names=("sxxp", "dax"))
vols = fetch_vol_indices(names=("v2x", "vdax", "vix"))
```

This will download data via `yfinance`, write cleaned close-price series into `data/raw/`, and return `pandas` DataFrames ready for feature engineering.

