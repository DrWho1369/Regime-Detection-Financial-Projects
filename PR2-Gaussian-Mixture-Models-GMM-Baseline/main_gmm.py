"""
GMM regime analysis: clean entry-point that imports from src/ and runs
the full end-to-end pipeline (load → standardise → optional PCA → GMM
selection → fit final model → diagnostics → plots → export).
"""
from __future__ import annotations

from pathlib import Path

import argparse
import joblib
import numpy as np
import pandas as pd

from src.models.gmm_engine import (
    DEFAULT_REG_COVAR,
    FEATURE_COLUMNS_14,
    load_features,
    standardise_features,
    reduce_with_pca,
    fit_gmms_and_scores,
    _aggregate_scores_per_k,
    choose_k_opt,
    fit_final_gmm,
)
from src.analysis.regime_diagnostics import (
    compute_profiles,
    compute_persistence,
    compute_transition_matrix,
    export_probs,
)
from src.visualization.regime_plots import (
    plot_bic_aic_silhouette,
    plot_regime_heatmap,
    plot_regimes_on_price,
)


def run_walk_forward_stability_check(
    X_scaled: np.ndarray,
    max_k: int,
    reg_covar: float,
) -> None:
    """
    Walk-Forward Stability Check: fit GMM on first 60% and on full sample;
    print whether the optimal k (number of regimes) changes. Addresses
    non-stationarity (Matuozzo theme).
    """
    n_60 = int(X_scaled.shape[0] * 0.6)
    if n_60 < 10:
        return
    X_60 = X_scaled[:n_60]
    _, bics_60, aics_60, sils_60 = fit_gmms_and_scores(
        X_60, k_min=2, k_max=max_k, reg_covar=reg_covar
    )
    k_opt_60, cov_60 = choose_k_opt(bics_60, sils_60)
    print(
        f"\nWalk-forward stability (non-stationarity sensitivity): "
        f"k_opt (first 60%) = {k_opt_60} (cov={cov_60!r}), "
        f"k_opt (full sample) = see below."
    )
    print(
        "If these differ, regime structure may be evolving over time."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GMM regime analysis baseline.")
    parser.add_argument(
        "--features-path",
        type=str,
        default="../PR1-Financial-Feature-Engineering/data/processed/features.csv",
        help="Path to Project 1 engineered features CSV.",
    )
    parser.add_argument(
        "--raw-price-path",
        type=str,
        default="../PR1-Financial-Feature-Engineering/data/raw/raw_panel_target_2010-01-01_2020-12-31.csv",
        help="Path to Project 1 raw price panel CSV.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Maximum number of GMM components to consider.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2017-12-31",
        help="End date for scaler training period (YYYY-MM-DD). Matches Project 1 split.",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Reduce 14 features to 5-7 PCA components (95%% variance) before GMM.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Target variance explained by PCA (default 0.95).",
    )
    parser.add_argument(
        "--reg-covar",
        type=float,
        default=DEFAULT_REG_COVAR,
        help="GMM diagonal regularization (default 1e-4).",
    )

    args = parser.parse_args()

    features_path = Path(args.features_path)
    raw_price_path = Path(args.raw_price_path)
    max_k = int(args.max_k)
    train_end = pd.to_datetime(args.train_end)
    use_pca = getattr(args, "use_pca", False)
    pca_variance = getattr(args, "pca_variance", 0.95)
    reg_covar = getattr(args, "reg_covar", DEFAULT_REG_COVAR)

    # 1) Load and standardise (scaler fit on train period only, e.g. 2010–2017)
    df = load_features(features_path)
    X_scaled, scaler, idx, df_aligned = standardise_features(df, train_end)
    pca = None
    if use_pca:
        train_mask = (idx <= train_end).values
        X_scaled, pca = reduce_with_pca(
            X_scaled,
            train_mask,
            variance_explained=pca_variance,
            n_components_min=5,
            n_components_max=7,
        )
        print(f"PCA: reduced to {pca.n_components_} components ({pca_variance:.0%} variance).")

    # 2) Fit GMMs for k=2..max_k, both full and diag covariance
    models, bics, aics, sils = fit_gmms_and_scores(
        X_scaled, k_min=2, k_max=max_k, reg_covar=reg_covar
    )
    bic_per_k, aic_per_k, sil_per_k = _aggregate_scores_per_k(bics, aics, sils)

    print("k | BIC (best) | AIC (best) | Silhouette (best)")
    print("-" * 50)
    for k in sorted(bic_per_k.keys()):
        print(f"{k} | {bic_per_k[k]:.2f} | {aic_per_k[k]:.2f} | {sil_per_k.get(k, np.nan):.4f}")

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_bic_aic_silhouette(
        bic_per_k,
        aic_per_k,
        sil_per_k,
        figures_dir / "gmm_bic_aic_silhouette.png",
    )

    # Walk-Forward Stability Check: 60% vs 100%
    run_walk_forward_stability_check(X_scaled, max_k, reg_covar)

    k_opt, cov_type_opt = choose_k_opt(bics, sils)
    print(f"\nSelected k_opt={k_opt}, covariance_type={cov_type_opt!r} (BIC + simplicity check).")

    # 3) Fit final GMM and save (joblib)
    gmm = fit_final_gmm(
        X_scaled, k_opt, cov_type_opt=cov_type_opt, reg_covar=reg_covar
    )
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(gmm, Path("data/processed/gmm_model.joblib"))
    if pca is not None:
        joblib.dump(pca, Path("data/processed/pca_model.joblib"))

    probs = gmm.predict_proba(X_scaled)
    regime_ids = probs.argmax(axis=1)

    df_regimes = df_aligned.copy()
    df_regimes["regime_id"] = regime_ids

    # 4) Regime diagnostics: profile table, persistence, transition matrix
    profiles = compute_profiles(df_regimes, regime_ids)
    profiles_path = Path("data/processed/gmm_regime_profiles.csv")
    profiles.to_csv(profiles_path)

    persistence = compute_persistence(regime_ids)
    print("\nRegime persistence (consecutive days statistics):")
    print(persistence)

    trans_mat = compute_transition_matrix(regime_ids, k_opt)
    print("\nPseudo-transition matrix P[i,j] = P(regime_{t+1}=j | regime_t=i):")
    print(trans_mat)
    trans_mat.to_csv(Path("data/processed/gmm_transition_matrix.csv"))

    export_probs(
        df_aligned,
        regime_ids,
        probs,
        profiles,
        Path("data/processed/gmm_probs.csv"),
    )

    # 5) Publication-quality plots
    plot_regime_heatmap(
        profiles,
        figures_dir / "gmm_regime_feature_heatmap.png",
    )
    df_regimes_labels = df_regimes[["regime_id"]].assign(
        regime_label=df_regimes["regime_id"].map(profiles["label"].to_dict())
    )
    plot_regimes_on_price(
        raw_price_path,
        df_regimes_labels,
        profiles,
        figures_dir / "gmm_regimes_price_colored.png",
    )


if __name__ == "__main__":
    main()
