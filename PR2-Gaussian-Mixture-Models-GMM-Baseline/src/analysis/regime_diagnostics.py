from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.models.gmm_engine import FEATURE_COLUMNS_14


def compute_profiles(
    df_regimes: pd.DataFrame,
    regime_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-regime feature means, mean return, volatility, and an
    interpretable label (Bull/Bear/Stagnant/Mean-Reverting). Profile table
    used for regime definitions and heatmap.
    """
    df_regimes = df_regimes.copy()
    df_regimes["regime_id"] = regime_ids

    feature_means = df_regimes.groupby("regime_id")[FEATURE_COLUMNS_14].mean()

    stats = df_regimes.groupby("regime_id")["target_log_return"].agg(
        mean_return="mean",
        volatility="std",
    )

    profiles = feature_means.join(stats)

    vols = profiles["volatility"]
    v_low, v_high = vols.quantile([1 / 3, 2 / 3])
    # Treat near-zero mean returns as mean-reverting/stagnant, regardless of volatility
    ret_threshold = profiles["mean_return"].abs().quantile(1 / 3)

    labels = []
    for _, row in profiles.iterrows():
        mu = row["mean_return"]
        sigma = row["volatility"]

        if abs(mu) <= ret_threshold:
            labels.append("Mean-Reverting / Stagnant")
        elif mu > 0 and sigma <= v_low:
            labels.append("Low-Vol Bull")
        elif mu > 0 and sigma > v_low:
            labels.append("High-Vol Bull")
        elif mu < 0 and sigma >= v_high:
            labels.append("High-Vol Bear")
        else:
            labels.append("Stagnant / Neutral")

    profiles["label"] = labels
    return profiles


def compute_persistence(regime_ids: np.ndarray) -> pd.DataFrame:
    """
    Compute average, median, and max consecutive days spent in each regime
    (regime duration / persistence statistics).
    """
    ids = regime_ids
    if len(ids) < 2:
        raise ValueError("Need at least two observations to compute persistence.")

    boundaries = (ids[1:] != ids[:-1]).astype(int)
    segments = np.cumsum(np.concatenate([[0], boundaries]))
    runs = (
        pd.DataFrame({"regime_id": ids, "segment": segments})
        .groupby("segment")
        .agg(regime=("regime_id", "first"), length=("regime_id", "size"))
    )

    persistence = runs.groupby("regime")["length"].agg(
        avg_run="mean",
        median_run="median",
        max_run="max",
    )
    return persistence


def compute_transition_matrix(regime_ids: np.ndarray, k: int) -> pd.DataFrame:
    """
    Estimate the empirical transition matrix P(regime_{t+1} = j | regime_t = i).
    """
    ids = regime_ids
    counts = np.zeros((k, k), dtype=int)
    for t in range(len(ids) - 1):
        i, j = ids[t], ids[t + 1]
        counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(counts, row_sums, where=row_sums > 0)

    trans_mat = pd.DataFrame(P, index=range(k), columns=range(k))
    return trans_mat


def export_probs(
    df: pd.DataFrame,
    regime_ids: np.ndarray,
    probs: np.ndarray,
    profiles: pd.DataFrame,
    output_path,
) -> None:
    """
    Export per-day regime probabilities and labels to CSV.
    """
    prob_cols: Dict[str, np.ndarray] = {
        f"regime_{j}_prob": probs[:, j] for j in range(probs.shape[1])
    }
    probs_df = pd.DataFrame(prob_cols, index=df.index)
    probs_df["regime_id"] = regime_ids

    label_map = profiles["label"].to_dict()
    probs_df["regime_label"] = probs_df["regime_id"].map(label_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    probs_df.to_csv(output_path, index=True, date_format="%Y-%m-%d")
