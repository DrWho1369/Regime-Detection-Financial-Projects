from __future__ import annotations

import inspect
import re
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .config import DATA_DIR
from .features import feature_engineer as fe


def check_lookahead_in_feature_code() -> None:
    """
    Naive static check for potential look-ahead in feature calculations.

    Scans ``feature_engineer.py`` for occurrences of ``.rolling(`` and reports
    lines where a subsequent ``.shift(1)`` is not present on the same line.

    This is intentionally conservative: it is meant as a reminder that any
    rolling statistic used as a feature at time t should be shifted by one
    period so that it only uses data up to t-1.
    """
    src = inspect.getsource(fe)
    lines = src.splitlines()
    pattern = re.compile(r"\.rolling\(")

    print("=== Look-ahead diagnostic for rolling features ===")
    for lineno, line in enumerate(lines, start=1):
        if pattern.search(line):
            has_shift = ".shift(1)" in line
            status = "OK (shift(1) found)" if has_shift else "POTENTIAL LOOK-AHEAD (no shift(1))"
            print(f"Line {lineno:4d}: {status} :: {line.strip()}")
    print("=== End of rolling look-ahead diagnostic ===")


def run_adf_tests(
    features: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller tests across a set of feature columns.

    Returns a DataFrame with test statistics, p-values, and stationarity flags.
    """
    if feature_cols is None:
        feature_cols = [c for c in features.columns if np.issubdtype(features[c].dtype, np.number)]

    records = []
    for col in feature_cols:
        series = features[col].dropna()
        if len(series) < 50:
            records.append(
                {
                    "feature": col,
                    "n_obs": len(series),
                    "adf_stat": np.nan,
                    "p_value": np.nan,
                    "stationary": False,
                    "note": "too few observations",
                }
            )
            continue

        result = adfuller(series.values, autolag="AIC")
        adf_stat, p_value, usedlag, nobs, crit_vals, icbest = result

        records.append(
            {
                "feature": col,
                "n_obs": int(nobs),
                "adf_stat": float(adf_stat),
                "p_value": float(p_value),
                "stationary": bool(p_value <= alpha),
                "note": "",
            }
        )

    report = pd.DataFrame.from_records(records).set_index("feature").sort_values("p_value")

    print("=== ADF Stationarity Report ===")
    print(report[["n_obs", "adf_stat", "p_value", "stationary"]])
    print("=== End of ADF report ===")

    return report


def difference_and_retest(
    features: pd.DataFrame,
    adf_report: pd.DataFrame,
    alpha: float = 0.05,
    suffix: str = "_pct_change",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For any feature failing the ADF test, apply a first-order difference
    (percentage change) and re-run the ADF.

    Returns:
    - A new features DataFrame including the differenced columns.
    - A second ADF report for the differenced columns only.
    """
    non_stationary = adf_report[adf_report["stationary"] == False]  # noqa: E712
    if non_stationary.empty:
        print("All features appear stationary at the specified alpha; no differencing applied.")
        return features.copy(), pd.DataFrame()

    new_features = features.copy()
    diff_cols = []
    for feature_name in non_stationary.index:
        col = features[feature_name]
        diff_col_name = f"{feature_name}{suffix}"
        new_features[diff_col_name] = col.pct_change()
        diff_cols.append(diff_col_name)

    print("Re-running ADF tests on differenced (percentage change) features...")
    diff_report = run_adf_tests(new_features, feature_cols=diff_cols, alpha=alpha)

    return new_features, diff_report


def run_full_stationarity_diagnostics(
    features_path: Optional[str] = None,
    alpha: float = 0.05,
) -> None:
    """
    Convenience entry point to run ADF tests and differencing diagnostics
    on the final engineered feature set.

    - Loads ``data/processed/features.csv`` by default.
    - Writes ``adf_report.csv`` and ``adf_report_diff.csv`` to ``data/processed``.
    """
    # 1) Quick static look-ahead scan on feature code
    check_lookahead_in_feature_code()

    # 2) Load engineered features
    if features_path is None:
        path = DATA_DIR / "processed" / "features.csv"
    else:
        path = DATA_DIR / "processed" / features_path

    features = pd.read_csv(path, index_col=0, parse_dates=True)

    print(f"Running ADF tests on features from {path} (alpha={alpha})...")
    adf_report = run_adf_tests(features, alpha=alpha)

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    adf_csv = processed_dir / "adf_report.csv"
    adf_report.to_csv(adf_csv)
    print(f"Saved ADF report to {adf_csv}")

    print("Applying first-order percentage change to non-stationary features and re-testing...")
    features_diff, diff_report = difference_and_retest(features, adf_report, alpha=alpha)

    if not diff_report.empty:
        diff_csv = processed_dir / "adf_report_diff.csv"
        diff_report.to_csv(diff_csv)
        print(f"Saved differenced-feature ADF report to {diff_csv}")
    else:
        print("No non-stationary features detected; no differenced report written.")


if __name__ == "__main__":
    run_full_stationarity_diagnostics()

