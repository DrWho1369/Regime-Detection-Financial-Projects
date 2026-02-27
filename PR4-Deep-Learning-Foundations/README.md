## Project 4 – Deep Learning Foundations (CNN + DBE)

This project builds a deep-learning baseline on top of the existing regime
analysis stack (Projects 1–3). It:

- Stacks the **14 canonical features** from Project 1/1a with **GMM / HMM regime
  probabilities** using `src/data/feature_stacker.py`.
- Converts daily panels into **2D image-like tensors** of shape
  `(lookback, n_features)` suitable for a CNN using
  `src/data/tensor_builder.py`.
- Trains a **PyTorch 2D CNN** with a **Directional Big Error (DBE)** loss that
  heavily penalises wrong-direction forecasts (see
  `src/models/cnn_baseline.py`).
- Supports **cross-asset training** for both equity (Project 1) and European
  gas (Project 1a) via `scripts/train_deep_regime.py`.

### DBE loss (Directional Big Error)

The DBE loss emphasises getting the **direction** of returns correct:

\\[
L_{\\text{DBE}}(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^{N}
\\left( (y_i - \\hat{y}_i)^2 \\times [1 + \\alpha \\cdot
\\mathbf{1}(\\operatorname{sign}(y_i) \\neq \\operatorname{sign}(\\hat{y}_i))] \\right)
\\]

where \\(\\alpha > 0\\) controls how strongly to penalise wrong-direction
predictions. The PyTorch implementation lives in `DirectionalBigErrorLoss`
inside `src/models/cnn_baseline.py`.

### How to run

From the repository root, after running PR1/PR1a/PR2/PR3 pipelines:

```bash
cd PR4-Deep-Learning-Foundations

# Equity CNN (Project 1 features + equity GMM/HMM regimes)
python -m scripts.train_deep_regime --asset equity --alpha 2.0

# Gas CNN (Project 1a features + gas HMM regimes)
python -m scripts.train_deep_regime --asset gas --alpha 2.0
```

Models are saved under `models/` as:

- `cnn_v1_equity.pth`
- `cnn_v1_gas.pth`

