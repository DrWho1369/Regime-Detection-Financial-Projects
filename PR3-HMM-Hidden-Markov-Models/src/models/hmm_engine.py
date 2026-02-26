from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# Canonical 14-feature schema (kept consistent with Projects 1 and 2)
FEATURE_COLUMNS_14: List[str] = [
    "entropy_66d",
    "vol_scaled_return_126d",
    "skewness_126d",
    "vol_of_vol",
    "bollinger_pct_b_20d",
    "cmci_20d",
    "stoch_k_14d",
    "rsi_14d",
    "williams_r_14d",
    "momentum_63d",
    "momentum_252d",
    "vstoxx_chg",
    "gold_log_return",
    "bund_rets",
]


def load_features(features_path: Path) -> pd.DataFrame:
    """
    Load the engineered feature panel from Project 1.
    """
    df = pd.read_csv(features_path, index_col=0, parse_dates=True).sort_index()
    missing = [c for c in FEATURE_COLUMNS_14 if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected the following feature columns in {features_path}: {missing}"
        )
    return df


def standardise_features(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
) -> Tuple[np.ndarray, StandardScaler, pd.DatetimeIndex, pd.DataFrame]:
    """
    Standardise the 14-feature matrix in a time-safe way.

    The scaler is fit only on observations with index <= train_end and then
    applied to the full feature matrix (after dropping NaNs). This mirrors
    the Project 1/2 convention (e.g. 2010–2017 as the training period).
    """
    X = df[FEATURE_COLUMNS_14].copy().dropna()
    idx = X.index

    train_mask = idx <= train_end
    if not train_mask.any():
        raise ValueError("No training data before train_end for scaling.")

    X_train = X.loc[train_mask]
    scaler = StandardScaler().fit(X_train.values)
    X_scaled = scaler.transform(X.values).astype(np.float64)

    # Drop any rows that are still non-finite after scaling
    finite_mask = np.isfinite(X_scaled).all(axis=1)
    X_scaled = X_scaled[finite_mask]
    idx = idx[finite_mask]
    return X_scaled, scaler, idx, df.loc[idx]


def _num_hmm_params(n_components: int, n_features: int, covariance_type: str) -> int:
    """
    Approximate number of free parameters in a GaussianHMM.

    Includes:
      - initial state probabilities (k - 1)
      - transition matrix rows (k * (k - 1))
      - means (k * n_features)
      - covariances:
          * full: k * n_features * (n_features + 1) / 2
          * diag: k * n_features
    """
    k = n_components
    # Initial state probabilities and transition matrix
    pi_params = k - 1
    trans_params = k * (k - 1)

    means_params = k * n_features
    if covariance_type == "full":
        cov_params = int(k * (n_features * (n_features + 1) / 2))
    elif covariance_type == "diag":
        cov_params = k * n_features
    else:
        raise ValueError(f"Unsupported covariance_type for HMM: {covariance_type!r}")

    return pi_params + trans_params + means_params + cov_params


def fit_hmms_and_scores(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    covariance_type: str = "full",
    reg_covar: float = 1e-4,
    random_state: int = 42,
) -> Tuple[Dict[int, GaussianHMM], Dict[int, float], Dict[int, float]]:
    """
    Fit GaussianHMMs for k in [k_min, k_max] and compute BIC and AIC.

    The reg_covar parameter is passed through to GaussianHMM as min_covar to
    stabilise covariance estimates during EM iterations.
    """
    n_samples, n_features = X_scaled.shape
    models: Dict[int, GaussianHMM] = {}
    bics: Dict[int, float] = {}
    aics: Dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        hmm = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=500,
            random_state=random_state,
            min_covar=reg_covar,
        )
        hmm.fit(X_scaled)

        logL = hmm.score(X_scaled)
        n_params = _num_hmm_params(k, n_features, covariance_type)
        bic_k = -2.0 * logL + n_params * np.log(n_samples)
        aic_k = -2.0 * logL + 2.0 * n_params

        models[k] = hmm
        bics[k] = bic_k
        aics[k] = aic_k

    return models, bics, aics


def choose_k_opt(bics: Dict[int, float]) -> int:
    """
    Select the optimal number of hidden states as the minimiser of BIC.
    """
    if not bics:
        raise ValueError("No BIC scores provided to choose_k_opt.")
    return min(bics, key=bics.get)


def fit_final_hmm(
    X_scaled: np.ndarray,
    k_opt: int,
    covariance_type: str = "full",
    reg_covar: float = 1e-4,
    random_state: int = 42,
) -> GaussianHMM:
    """
    Fit the final GaussianHMM model with the chosen number of components and
    covariance type, applying covariance regularisation.

    The reg_covar parameter is passed as min_covar to GaussianHMM so that
    covariance updates remain numerically stable during EM training.
    """
    model = GaussianHMM(
        n_components=k_opt,
        covariance_type=covariance_type,
        n_iter=500,
        random_state=random_state,
        min_covar=reg_covar,
    )
    model.fit(X_scaled)
    return model


def compute_transition_matrix(model: GaussianHMM) -> pd.DataFrame:
    """
    Wrap the learned HMM transition matrix in a DataFrame for inspection
    and CSV export.
    """
    A = model.transmat_
    k = A.shape[0]
    return pd.DataFrame(A, index=range(k), columns=range(k))


def build_hmm_regime_frame(
    df_aligned: pd.DataFrame,
    model: GaussianHMM,
    X_scaled: np.ndarray,
    return_col: str = "target_log_return",
) -> pd.DataFrame:
    """
    Decode the most likely HMM state sequence and posterior probabilities
    and attach them to a copy of the aligned feature DataFrame.
    """
    logprob, state_sequence = model.decode(X_scaled, algorithm="viterbi")
    posteriors = model.predict_proba(X_scaled)
    k_opt = model.n_components

    df_hmm = df_aligned.copy()
    df_hmm["hmm_regime_id"] = state_sequence
    for j in range(k_opt):
        df_hmm[f"hmm_regime_{j}_prob"] = posteriors[:, j]

    # Optional: compute simple per-regime mean return for later labelling
    if return_col in df_hmm.columns:
        means = (
            df_hmm.groupby("hmm_regime_id")[return_col]
            .mean()
            .rename("hmm_mean_return")
        )
        df_hmm = df_hmm.join(means, on="hmm_regime_id")

    return df_hmm

