from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureStackerConfig:
    """
    Configuration for stacking raw features with regime probabilities.

    Paths are relative to the PR4 project directory by default but can be
    overridden if needed.
    """

    # Equity inputs
    equity_features_path: Path = Path(
        "../PR1-Financial-Feature-Engineering/data/processed/features.csv"
    )
    equity_gmm_probs_path: Path = Path(
        "../PR2-Gaussian-Mixture-Models-GMM-Baseline/data/processed/gmm_probs.csv"
    )
    # HMM equity probs are expected to be written by PR3 with --out-prefix hmm
    equity_hmm_probs_path: Path = Path(
        "../PR3-HMM-Hidden-Markov-Models/data/processed/hmm_probs.csv"
    )

    # Gas inputs
    gas_features_path: Path = Path(
        "../PR1a-European-Gas-Feature-Engineering/data/processed/gas_features.csv"
    )
    # Optional gas GMM probs (if you later fit a gas-specific GMM)
    gas_gmm_probs_path: Path | None = None
    gas_hmm_probs_path: Path = Path(
        "../PR3-HMM-Hidden-Markov-Models/data/processed/hmm_gas_probs.csv"
    )


def _load_csv(path: Path, index_col: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV at {path} but it does not exist.")
    df = pd.read_csv(path, index_col=index_col, parse_dates=True).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def _extract_gmm_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract regime probability columns from a GMM probs file.

    Expects columns named 'regime_{j}_prob' plus metadata columns
    'regime_id' and 'regime_label'. Only the probability columns are
    returned, ordered by regime index.
    """
    prob_cols: list[str] = [
        c for c in df.columns if c.startswith("regime_") and c.endswith("_prob")
    ]
    if not prob_cols:
        raise KeyError("No regime_*_prob columns found in GMM probs DataFrame.")

    # Sort by numeric regime index
    def _regime_index(col: str) -> int:
        inner = col.removeprefix("regime_").removesuffix("_prob")
        return int(inner)

    prob_cols_sorted = sorted(prob_cols, key=_regime_index)
    return df[prob_cols_sorted]


def _extract_hmm_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract regime probability columns from an HMM regime frame.

    Expects columns named 'hmm_regime_{j}_prob' alongside 'hmm_regime_id'.
    Only the probability columns are returned, ordered by regime index.
    """
    prob_cols: list[str] = [
        c for c in df.columns if c.startswith("hmm_regime_") and c.endswith("_prob")
    ]
    if not prob_cols:
        raise KeyError("No hmm_regime_*_prob columns found in HMM probs DataFrame.")

    def _regime_index(col: str) -> int:
        inner = col.removeprefix("hmm_regime_").removesuffix("_prob")
        return int(inner)

    prob_cols_sorted = sorted(prob_cols, key=_regime_index)
    return df[prob_cols_sorted]


def build_stacked_frame(
    asset: Literal["equity", "gas"],
    config: FeatureStackerConfig | None = None,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Build a stacked feature frame combining raw features with GMM/HMM
    regime probabilities.

    Parameters
    ----------
    asset:
        'equity' for Project 1 features, 'gas' for Project 1a.
    config:
        Paths to the underlying CSVs. If None, defaults are used.
    feature_cols:
        Optional explicit ordering of the 14 canonical features. If None,
        equity mode infers them from PR1, and gas mode infers from PR1a.

    Returns
    -------
    stacked : pd.DataFrame
        DataFrame indexed by date with columns:
        - 14 raw features (canonical order)
        - k_gmm GMM regime probabilities (if available)
        - k_hmm HMM regime probabilities
        - target return column ('target_log_return' or 'gas_log_return')
    """
    config = config or FeatureStackerConfig()

    if asset == "equity":
        feat_path = config.equity_features_path
        gmm_path = config.equity_gmm_probs_path
        hmm_path = config.equity_hmm_probs_path
        return_col = "target_log_return"
    elif asset == "gas":
        feat_path = config.gas_features_path
        gmm_path = config.gas_gmm_probs_path
        hmm_path = config.gas_hmm_probs_path
        return_col = "gas_log_return"
    else:
        raise ValueError(f"Unsupported asset type: {asset!r}")

    # Load base feature frame
    base = _load_csv(feat_path)

    # Infer canonical feature columns if not provided
    if feature_cols is None:
        # For equity, reuse the known 14-feature subset (exclude return/labels)
        if asset == "equity":
            known_non_features = {return_col, "y_forward_1d"}
        else:
            known_non_features = {return_col}
        feature_cols = [
            c for c in base.columns if c not in known_non_features
        ]

    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols if c not in base.columns]
    if missing:
        raise KeyError(
            f"The following feature columns are missing from {feat_path}: {missing}"
        )

    # Load and align GMM probabilities (optional for gas)
    stacked = base.copy()
    pieces: list[pd.DataFrame] = [stacked[feature_cols]]

    if gmm_path is not None:
        gmm = _load_csv(Path(gmm_path))
        gmm_probs = _extract_gmm_probs(gmm)
        pieces.append(gmm_probs)

    # Load and align HMM probabilities
    hmm = _load_csv(Path(hmm_path))
    hmm_probs = _extract_hmm_probs(hmm)
    pieces.append(hmm_probs)

    # Inner-join on common dates to avoid look-ahead across sources
    merged = pd.concat(pieces + [stacked[[return_col]]], axis=1, join="inner")
    merged = merged.dropna(subset=[return_col])
    return merged


def save_stacked_frame(
    df: pd.DataFrame,
    asset: Literal["equity", "gas"],
    base_dir: Path | None = None,
) -> Path:
    """
    Save a stacked frame to PR4 data/processed with a conventional name.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"stacked_{asset}.csv"
    df.to_csv(out_path, index=True, date_format="%Y-%m-%d")
    return out_path

