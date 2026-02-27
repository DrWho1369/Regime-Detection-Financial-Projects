from __future__ import annotations

"""
Build the engineered gas feature set for Project 1a.

This script:
- Fetches the raw gas panel (gas + proxies + storage/HDD where available).
- Constructs the core 14-feature gas panel using `build_gas_feature_panel`.
- Attaches the gas log return as the target column.
- Saves the result to data/processed/gas_features.csv for downstream models.
"""

from pathlib import Path

from src.data.gas_loader import fetch_gas_panel
from src.features.gas_feature_engineer import build_gas_feature_panel
from src.config import DATA_DIR


def main() -> Path:
    # 1) Raw gas panel (TTF + Brent + Carbon + Coal + storage/HDD)
    panel = fetch_gas_panel()

    # 2) Core 14-feature gas panel (statistical, technical, momentum, macro)
    features = build_gas_feature_panel(panel)

    # 3) Attach gas_log_return as target for later tensor creation
    if "gas_log_return" not in panel.columns:
        raise KeyError("Expected 'gas_log_return' in raw gas panel.")
    features_with_ret = features.join(panel[["gas_log_return"]])

    # 4) Save to data/processed/gas_features.csv
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "gas_features.csv"
    features_with_ret.to_csv(out_path, index=True, date_format="%Y-%m-%d")
    print(f"Saved gas features to {out_path}")
    return out_path


if __name__ == "__main__":
    main()

