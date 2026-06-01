"""Loss surfaces and small synthetic datasets used across experiments."""

from __future__ import annotations

import numpy as np


def bowl_loss(point: np.ndarray) -> float:
    x, y = point
    return 0.5 * (x**2 + y**2)


def bowl_grad(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([x, y], dtype=np.float64)


def ravine_loss(point: np.ndarray) -> float:
    x, y = point
    return 0.5 * (x**2 + 50.0 * y**2)


def ravine_grad(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([x, 50.0 * y], dtype=np.float64)


def scale_loss(point: np.ndarray) -> float:
    x, y = point
    return 0.5 * (0.1 * x**2 + 50.0 * y**2)


def scale_grad(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([0.1 * x, 50.0 * y], dtype=np.float64)


def noisy_quadratic_loss(point: np.ndarray) -> float:
    x, y = point
    return 0.5 * (x**2 + 10.0 * y**2)


def noisy_quadratic_grad(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([x, 10.0 * y], dtype=np.float64)


def make_synthetic_linear_regression(
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: int,
):
    """Return torch tensors (X, y) for y = Xw + b + noise."""
    import torch

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w_true = rng.normal(size=(n_features, 1)).astype(np.float32)
    b_true = np.float32(rng.normal())
    y = x @ w_true + b_true + rng.normal(scale=noise_std, size=(n_samples, 1)).astype(np.float32)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor

