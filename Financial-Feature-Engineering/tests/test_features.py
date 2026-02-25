from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import feature_engineer as fe
from src.features import momentum as mom


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2010-01-01", periods=300, freq="B")
        # Simple random walk for returns
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, size=len(dates)), index=dates, name="r")
        prices = 100 * np.exp(returns.cumsum())
        self.returns = returns
        self.prices = prices

    def test_shannon_entropy_has_expected_length(self) -> None:
        ent = fe.shannon_entropy_binary(self.returns, window=66)
        self.assertEqual(len(ent), len(self.returns))
        # Initial window should be NaN
        self.assertTrue(ent.iloc[:65].isna().all())

    def test_vol_scaled_returns_has_nans_in_warmup(self) -> None:
        scaled = fe.vol_scaled_returns(self.returns, vol_window=126)
        self.assertTrue(scaled.iloc[:125].isna().all())

    def test_momentum_functions(self) -> None:
        m3 = mom.momentum_3m(self.prices)
        m12 = mom.momentum_12m(self.prices)
        self.assertEqual(len(m3), len(self.prices))
        self.assertEqual(len(m12), len(self.prices))


if __name__ == "__main__":
    unittest.main()

