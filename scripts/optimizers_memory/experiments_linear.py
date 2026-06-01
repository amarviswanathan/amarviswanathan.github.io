"""Linear-regression experiments for optimizer sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from losses import make_synthetic_linear_regression
from optimizers import Adam, AdamW, Momentum, RMSProp, SGD, optimizer_state_report
from plotting import plot_line_series


def _init_linear_params(n_features: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w = 0.15 * torch.randn((n_features, 1), generator=generator, dtype=torch.float32)
    b = 0.15 * torch.randn((1,), generator=generator, dtype=torch.float32)
    return w, b


def _train_linear(
    optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    init_w: torch.Tensor,
    init_b: torch.Tensor,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    w = init_w.clone().requires_grad_(True)
    b = init_b.clone().requires_grad_(True)
    losses = np.zeros(steps, dtype=np.float64)
    norms = np.zeros(steps, dtype=np.float64)

    for step in range(steps):
        preds = x @ w + b
        loss = torch.mean((preds - y) ** 2)
        loss.backward()

        with torch.no_grad():
            optimizer.step([w, b], [w.grad, b.grad])
            norms[step] = torch.linalg.vector_norm(w).item()

        w.grad.zero_()
        b.grad.zero_()
        losses[step] = float(loss.item())

    return losses, norms


def generate_adamw_weight_decay(output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "adamw_weight_decay.png"
    x, y = make_synthetic_linear_regression(
        n_samples=512,
        n_features=20,
        noise_std=0.1,
        seed=13,
    )
    init_w, init_b = _init_linear_params(n_features=20, seed=13)
    steps = 300

    adam_l2 = Adam(lr=0.03, weight_decay=0.1)
    adamw = AdamW(lr=0.03, weight_decay=0.1)

    _, adam_l2_norms = _train_linear(adam_l2, x, y, init_w, init_b, steps)
    _, adamw_norms = _train_linear(adamw, x, y, init_w, init_b, steps)

    series = {
        "Adam + L2 mixed into grad": adam_l2_norms,
        "AdamW decoupled weight decay": adamw_norms,
    }
    return plot_line_series(
        series=series,
        output_path=output_path,
        title="Weight Norm Under Adam (L2) vs AdamW",
        ylabel="||w||2",
    )


def generate_linear_regression_loss(output_dir: str | Path) -> tuple[str, dict[str, int]]:
    output_path = Path(output_dir) / "linear_regression_loss.png"
    x, y = make_synthetic_linear_regression(
        n_samples=512,
        n_features=10,
        noise_std=0.1,
        seed=42,
    )
    init_w, init_b = _init_linear_params(n_features=10, seed=42)
    steps = 250

    experiments = {
        "SGD": SGD(lr=0.05),
        "Momentum": Momentum(lr=0.05, beta=0.9),
        "RMSProp": RMSProp(lr=0.03, beta=0.9),
        "Adam": Adam(lr=0.03),
        "AdamW": AdamW(lr=0.03, weight_decay=0.01),
    }

    loss_curves: dict[str, np.ndarray] = {}
    for name, optimizer in experiments.items():
        losses, _ = _train_linear(optimizer, x, y, init_w, init_b, steps)
        loss_curves[name] = losses

    state_report = optimizer_state_report([init_w, init_b])
    print(f"Model parameters: {state_report['Model parameters']}")
    print(f"SGD state values: {state_report['SGD state values']}")
    print(f"Momentum state values: {state_report['Momentum state values']}")
    print(f"RMSProp state values: {state_report['RMSProp state values']}")
    print(f"Adam state values: {state_report['Adam state values']}")
    print(f"AdamW state values: {state_report['AdamW state values']}")

    path = plot_line_series(
        series=loss_curves,
        output_path=output_path,
        title="Linear Regression Training Loss",
        ylabel="MSE Loss",
    )
    return path, state_report

