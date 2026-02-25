from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Default regularization for covariance diagonal (avoids singular matrices)
DEFAULT_REG_COVAR = 1e-4
COVARIANCE_TYPES = ("full", "diag")


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
    applied to the full feature matrix (after dropping NaNs). Matches Project 1's
    train period (e.g. 2010–2017) for consistency.
    """
    X = df[FEATURE_COLUMNS_14].copy()
    X = X.dropna()
    idx = X.index

    train_mask = idx <= train_end
    if not train_mask.any():
        raise ValueError("No training data before train_end for scaling.")

    X_train = X.loc[train_mask]
    scaler = StandardScaler().fit(X_train.values)
    X_scaled = scaler.transform(X.values).astype(np.float64)
    # Clip extreme z-scores to avoid overflow/underflow in GMM (exp/log in density)
    X_scaled = np.clip(X_scaled, -10.0, 10.0)
    # Drop any rows that are still non-finite (e.g. from 0/0 if a column had zero variance)
    finite_mask = np.isfinite(X_scaled).all(axis=1)
    X_scaled = X_scaled[finite_mask]
    idx = idx[finite_mask]
    return X_scaled, scaler, idx, df.loc[idx]


def reduce_with_pca(
    X: np.ndarray,
    train_mask: np.ndarray,
    variance_explained: float = 0.95,
    n_components_max: int = 7,
    n_components_min: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality with PCA on training data. Keeps the minimum number
    of components in [n_components_min, n_components_max] that explain at least
    variance_explained of the variance.
    """
    n_components_max = min(n_components_max, X.shape[1], np.count_nonzero(train_mask))
    n_components_min = min(n_components_min, n_components_max)
    pca = PCA(n_components=n_components_max, random_state=random_state)
    pca.fit(X[train_mask])
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # First index where cumvar >= variance_explained (0-based), so n_components = idx + 1
    idx = np.searchsorted(cumvar, variance_explained)
    n = int(np.clip(idx + 1, n_components_min, n_components_max))
    pca = PCA(n_components=n, random_state=random_state)
    pca.fit(X[train_mask])
    X_reduced = pca.transform(X)
    return X_reduced, pca


def fit_gmms_and_scores(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    random_state: int = 42,
    reg_covar: float = DEFAULT_REG_COVAR,
) -> Tuple[
    Dict[Tuple[int, str], GaussianMixture],
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str], float],
]:
    """
    Fit GMMs for each k in [k_min, k_max] and each covariance_type in
    ('full', 'diag'), with diagonal regularization. Return models and
    BIC, AIC, Silhouette keyed by (k, cov_type).
    """
    models: Dict[Tuple[int, str], GaussianMixture] = {}
    bics: Dict[Tuple[int, str], float] = {}
    aics: Dict[Tuple[int, str], float] = {}
    sils: Dict[Tuple[int, str], float] = {}

    for k in range(k_min, k_max + 1):
        for cov_type in COVARIANCE_TYPES:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message=r".*(divide by zero|overflow|invalid value).*",
                )
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    reg_covar=reg_covar,
                    random_state=random_state,
                    n_init=10,
                ).fit(X)
            key = (k, cov_type)
            models[key] = gmm
            bics[key] = gmm.bic(X)
            aics[key] = gmm.aic(X)
            try:
                labels_k = gmm.predict(X)
                sils[key] = silhouette_score(X, labels_k)
            except Exception:
                sils[key] = np.nan

    return models, bics, aics, sils


def _aggregate_scores_per_k(
    bics: Dict[Tuple[int, str], float],
    aics: Dict[Tuple[int, str], float],
    sils: Dict[Tuple[int, str], float],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """For each k, take best BIC (min), best AIC (min), best Silhouette (max) over cov_type."""
    bic_per_k: Dict[int, float] = {}
    aic_per_k: Dict[int, float] = {}
    sil_per_k: Dict[int, float] = {}
    ks = sorted({k for k, _ in bics.keys()})
    for k in ks:
        bic_per_k[k] = min(bics[(k, ct)] for ct in COVARIANCE_TYPES)
        aic_per_k[k] = min(aics[(k, ct)] for ct in COVARIANCE_TYPES)
        sil_vals = [sils.get((k, ct), np.nan) for ct in COVARIANCE_TYPES]
        finite = [s for s in sil_vals if np.isfinite(s)]
        sil_per_k[k] = max(finite) if finite else np.nan
    return bic_per_k, aic_per_k, sil_per_k


def choose_k_opt(
    bics: Dict[Tuple[int, str], float],
    sils: Dict[Tuple[int, str], float],
    silhouette_drop_threshold: float = 0.1,
) -> Tuple[int, str]:
    """
    Select (k_opt, cov_type_opt). Prefer lower BIC. If Silhouette at the
    largest k is significantly lower than at k=2, prefer a simpler model
    (k=2 or 3) for interpretable Bull/Bear regimes.
    """
    # Best BIC per k (over cov_type)
    best_bic_per_k: Dict[int, float] = {}
    best_cov_per_k: Dict[int, str] = {}
    for (k, cov_type), v in bics.items():
        if k not in best_bic_per_k or v < best_bic_per_k[k]:
            best_bic_per_k[k] = v
            best_cov_per_k[k] = cov_type

    k_bic = min(best_bic_per_k, key=best_bic_per_k.get)
    ks = sorted(best_bic_per_k.keys())
    k_min_grid, k_max_grid = min(ks), max(ks)
    sil_k2_vals = [sils.get((2, ct), np.nan) for ct in COVARIANCE_TYPES]
    sil_kmax_vals = [sils.get((k_max_grid, ct), np.nan) for ct in COVARIANCE_TYPES]
    sil_k2 = max((s for s in sil_k2_vals if np.isfinite(s)), default=0.0)
    sil_kmax = max((s for s in sil_kmax_vals if np.isfinite(s)), default=0.0)

    # Prefer simpler k if largest k has much worse Silhouette than k=2
    if k_max_grid >= 4 and sil_kmax < sil_k2 - silhouette_drop_threshold:
        candidate_k = [k for k in best_bic_per_k if k in (2, 3)]
        if candidate_k:
            k_opt = min(candidate_k, key=best_bic_per_k.get)
        else:
            k_opt = k_bic
    else:
        k_opt = k_bic

    cov_type_opt = best_cov_per_k[k_opt]
    return k_opt, cov_type_opt


def fit_final_gmm(
    X: np.ndarray,
    k_opt: int,
    cov_type_opt: str = "full",
    random_state: int = 42,
    reg_covar: float = DEFAULT_REG_COVAR,
) -> GaussianMixture:
    """
    Fit the final GaussianMixture with chosen k and covariance type and
    diagonal regularization. Used for joblib serialization and downstream
    regime assignment.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*(divide by zero|overflow|invalid value).*",
        )
        gmm = GaussianMixture(
            n_components=k_opt,
            covariance_type=cov_type_opt,
            reg_covar=reg_covar,
            random_state=random_state,
            n_init=10,
        ).fit(X)
    return gmm
