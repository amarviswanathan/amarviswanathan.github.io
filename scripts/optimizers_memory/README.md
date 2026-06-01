# Optimizers Memory Figures

This folder contains the code used to generate figures for:

`Optimizers Are Memory: A Visual Guide from SGD to AdamW`

## Files

- `optimizers.py`: from-scratch SGD, Momentum, RMSProp, Adam, AdamW
- `losses.py`: toy losses + synthetic linear-regression data
- `plotting.py`: shared Matplotlib helpers
- `experiments_toy.py`: bowl/ravine/scale/noisy/bias-correction figures
- `experiments_linear.py`: Adam-vs-AdamW norm plot + linear-loss plot + state counts
- `experiments_moons.py`: two-moons decision boundaries + accuracies
- `generate_all.py`: one command to generate all required figures

## Run

From repo root:

```bash
python scripts/optimizers_memory/generate_all.py
```

Figures are written to:

`assets/images/optimizers-memory/`

## Dependencies

- Python 3.10+
- `numpy`
- `matplotlib`
- `torch`
- `scikit-learn` (optional; local two-moons fallback is included)
