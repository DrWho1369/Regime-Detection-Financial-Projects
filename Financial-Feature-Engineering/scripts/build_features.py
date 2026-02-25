from __future__ import annotations

"""
Command-line entry point to build the engineered feature set for Project 1.

This script:
- Fetches the raw data panel (target index + proxies).
- Constructs the core 14-feature panel (entropy, vol-scaled returns, skewness,
  vol-of-vol, technical indicators, momentum, macro features).
- Adds the target log return and a 1-day forward return label.
- Saves the result to data/processed/features.csv.
"""

from pathlib import Path

from src.data.data_loader import fetch_raw_data
from src.features.feature_engineer import build_feature_panel
from src.visualize import add_forward_return_label, save_processed_features


def main(target_ticker: str = "^STOXX50E") -> Path:
    # 1) Raw panel (target index + VSTOXX, Gold, Bunds)
    panel = fetch_raw_data(target_ticker=target_ticker)

    # 2) Core features (entropy, vol-scaled returns, skewness, vol-of-vol, technicals)
    features = build_feature_panel(panel)

    # 3) Add target/gold/bund return columns and 1-day forward label
    features_with_ret = features.join(
        panel[["target_log_return", "gold_log_return", "bund_log_return"]]
    )
    features_labeled = add_forward_return_label(
        features_with_ret,
        return_col="target_log_return",
        label_name="y_forward_1d",
    )

    # Rename bund returns into a more explicit macro proxy name
    df_features = features_labeled.rename(columns={"bund_log_return": "bund_rets"})

    # Quick health check on sentiment and macro proxies
    cols_to_show = [c for c in ("vstoxx_chg", "vol_of_vol", "bund_rets") if c in df_features.columns]
    if cols_to_show:
        print(df_features[cols_to_show].dropna().head())

    # 4) Save engineered features for later use
    path = save_processed_features(df_features, filename="features.csv")
    return path


if __name__ == "__main__":
    out_path = main()
    print(f"Saved engineered features to {out_path}")

