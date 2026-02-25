from __future__ import annotations

"""
Convenience wrapper to build final 3D tensors for Project 2.

This script:
- Loads data/processed/features.csv.
- Standardises features using only 2010–2017 data.
- Builds 10-day windows over the explicit 14-feature set.
- Saves X_train, y_train, X_test, y_test as .npy in data/processed/.
"""

from src.prepare_final_data import prepare_and_save_final_data


def main() -> None:
    prepare_and_save_final_data()


if __name__ == "__main__":
    main()

