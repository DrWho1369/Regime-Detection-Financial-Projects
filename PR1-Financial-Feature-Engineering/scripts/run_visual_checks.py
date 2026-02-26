from __future__ import annotations

"""
Run the main visual sanity checks for Project 1:

- Correlation heatmap of the engineered features + target return.
- Entropy vs index price plot with key regime-shift annotations.
"""

from pathlib import Path

import pandas as pd

from src.data.data_loader import fetch_raw_data
from src.features.feature_engineer import shannon_entropy_binary
from src.visualize import plot_correlation_heatmap, plot_entropy_vs_price


def main(
    features_path: str = "data/processed/features.csv",
    corr_fig_path: str = "figures/corr_heatmap.png",
    entropy_fig_path: str = "figures/entropy_vs_price.png",
) -> None:
    # 1) Correlation heatmap (matches Figure 3 style)
    features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    corr_out = Path(corr_fig_path)
    corr_out.parent.mkdir(parents=True, exist_ok=True)
    plot_correlation_heatmap(
        features,
        target_return_col="target_log_return",
        output_path=corr_out,
    )
    print(f"Saved correlation heatmap to {corr_out}")

    # 2) Entropy vs price (Figure 2-style)
    panel = fetch_raw_data()
    entropy = shannon_entropy_binary(panel["target_log_return"], window=66)

    entropy_out = Path(entropy_fig_path)
    entropy_out.parent.mkdir(parents=True, exist_ok=True)
    plot_entropy_vs_price(
        price=panel["target"],
        entropy=entropy,
        output_path=entropy_out,
    )
    print(f"Saved entropy vs price figure to {entropy_out}")


if __name__ == "__main__":
    main()

