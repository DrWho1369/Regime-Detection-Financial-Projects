"""
Thin wrapper so `python -m main` still runs the GMM pipeline.
The canonical entry point is main_gmm.py.
"""
from __future__ import annotations

from main_gmm import main

if __name__ == "__main__":
    main()
