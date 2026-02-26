"""
HMM regime analysis for European Gas (Project 1a).

This entry point mirrors the equity HMM pipeline but operates on the
gas feature panel from Project 1a-European-Gas-Feature-Engineering.
"""
from __future__ import annotations

from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.hmm_engine import (
    load_features,
    standardise_features,
    fit_hmms_and_scores,
    choose_k_opt,
    fit_final_hmm,
    compute_transition_matrix,
    build_hmm_regime_frame,
)
from src.visualization.hmm_plots import plot_hmm_price


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM regime analysis for European Gas (Project 1a)."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="../PR1a-European-Gas-Feature-Engineering/data/processed/gas_features.csv",
        help="Path to Project 1a engineered gas features CSV.",
    )
    parser.add_argument(
        "--raw-price-path",
        type=str,
        default="../PR1a-European-Gas-Feature-Engineering/data/raw/raw_panel_gas_2010-01-01_2020-12-31.csv",
        help="Path to Project 1a raw gas price panel CSV.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2017-12-31",
        help="End date for scaler training period (YYYY-MM-DD). Matches equity split.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of hidden states to consider.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=6,
        help="Maximum number of hidden states to consider.",
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="full",
        choices=["full", "diag"],
        help="Covariance type for GaussianHMM.",
    )
    parser.add_argument(
        "--reg-covar",
        type=float,
        default=1e-4,
        help="Regularisation for covariances (passed as min_covar to GaussianHMM).",
    )

    args = parser.parse_args()

    features_path = Path(args.features_path)
    raw_price_path = Path(args.raw_price_path)
    train_end = pd.to_datetime(args.train_end)
    k_min = int(args.k_min)
    k_max = int(args.k_max)
    covariance_type = args.covariance_type
    reg_covar = float(args.reg_covar)

    # 1) Load and standardise (scaler fit on train period only, e.g. 2010–2017)
    df = load_features(features_path)
    X_scaled, scaler, idx, df_aligned = standardise_features(df, train_end)

    # 2) Fit HMMs for k=k_min..k_max and collect BIC/AIC
    models, bics, aics = fit_hmms_and_scores(
        X_scaled,
        k_min=k_min,
        k_max=k_max,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
    )

    print("k | BIC | AIC")
    print("-" * 30)
    ks_sorted = sorted(bics.keys())
    for k in ks_sorted:
        print(f"{k} | {bics[k]:.2f} | {aics[k]:.2f}")

    # 3) Plot BIC/AIC curves for gas HMM
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    bic_vals = [bics[k] for k in ks_sorted]
    aic_vals = [aics[k] for k in ks_sorted]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(ks_sorted, bic_vals, marker="o", label="BIC")
    ax.plot(ks_sorted, aic_vals, marker="o", label="AIC")
    ax.set_xlabel("Number of hidden states (k)")
    ax.set_ylabel("Information criterion")
    ax.set_title("HMM gas regimes: BIC and AIC")
    ax.legend()
    fig.tight_layout()
    bic_aic_path = figures_dir / "hmm_gas_bic_aic.png"
    fig.savefig(bic_aic_path, dpi=200)
    plt.close(fig)

    # 4) Choose k_opt (BIC minimiser) and fit final HMM
    k_opt = choose_k_opt(bics)
    print(f"\nSelected k_opt={k_opt} (BIC minimiser) for gas HMM.")

    model = fit_final_hmm(
        X_scaled,
        k_opt=k_opt,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
    )

    # 5) Build HMM regime frame (Viterbi path + posterior probabilities)
    df_hmm = build_hmm_regime_frame(df_aligned, model, X_scaled)

    # 6) Transition matrix and regime probabilities / Viterbi path outputs
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    trans_mat = compute_transition_matrix(model)
    trans_path = data_dir / "hmm_gas_transition_matrix.csv"
    trans_mat.to_csv(trans_path)

    print("\nGas HMM transition matrix:")
    print(trans_mat)

    probs_path = data_dir / "hmm_gas_probs.csv"
    df_hmm.to_csv(probs_path)

    # 7) Price chart with HMM regimes shaded (gas price column is 'gas')
    plot_hmm_price(
        raw_price_path=raw_price_path,
        hmm_probs_path=probs_path,
        output_path=figures_dir / "hmm_gas_price_regimes.png",
        price_col="gas",
        hmm_regime_col="hmm_regime_id",
    )


if __name__ == "__main__":
    main()

