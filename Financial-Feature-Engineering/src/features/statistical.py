from __future__ import annotations

"""
Statistical feature definitions used in the regime-detection project.

This module exposes the core statistical features described in Matuozzo (2023):
- Rolling Shannon entropy on the sign of returns.
- Volatility-scaled returns.
- Rolling skewness of returns.

Implementations are imported from ``feature_engineer`` so there is a
single source of truth.
"""

from .feature_engineer import (  # re-export for convenience
    rolling_skewness,
    shannon_entropy_binary,
    vol_scaled_returns,
)

__all__ = [
    "shannon_entropy_binary",
    "vol_scaled_returns",
    "rolling_skewness",
]

