from __future__ import annotations

"""
Convenience wrapper to build final 3D gas tensors for downstream models.

This script:
- Loads data/processed/gas_features.csv.
- Standardises the canonical 14 gas features using only 2010–2017 data.
- Builds 10-day windows and 1-day forward gas returns.
- Saves X_gas_train, y_gas_train, X_gas_test, y_gas_test as .npy.
"""

from src.prepare_gas_data import prepare_and_save_gas_data


def main() -> None:
    prepare_and_save_gas_data()


if __name__ == "__main__":
    main()

