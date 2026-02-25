from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from src.data import fetch as fetch_mod


class TestDataFetch(unittest.TestCase):
    @mock.patch.object(fetch_mod.yf, "download")
    def test_fetch_index_prices_raises_on_empty(self, mock_download) -> None:
        mock_download.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            fetch_mod.fetch_index_prices("^GDAXI", start="2010-01-01", end="2010-12-31")

    @mock.patch.object(fetch_mod.yf, "download")
    def test_fetch_index_prices_returns_close_series(self, mock_download) -> None:
        dates = pd.date_range("2010-01-01", periods=5, freq="B")
        mock_download.return_value = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104]},
            index=dates,
        )

        df = fetch_mod.fetch_index_prices("^GDAXI", start="2010-01-01", end="2010-12-31", auto_save=False)
        self.assertIn("close", df.columns)
        self.assertEqual(len(df), 5)


if __name__ == "__main__":
    unittest.main()

