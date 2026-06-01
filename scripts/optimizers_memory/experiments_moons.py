"""Two-moons MLP experiment for optimizer behavior on a nonlinear task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from optimizers import AdamW, Momentum, SGD
from plotting import plot_decision_boundary_panels


def _make_moons_local(n_samples: int, noise: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, np.pi, n_outer)
    theta_inner = rng.uniform(0.0, np.pi, n_inner)

    outer = np.c_[np.cos(theta_outer), np.sin(theta_outer)]
    inner = np.c_[1.0 - np.cos(theta_inner), 1.0 - np.sin(theta_inner) - 0.5]

    x = np.vstack([outer, inner]).astype(np.float32)
    y = np.hstack([np.zeros(n_outer, dtype=np.int64), np.ones(n_inner, dtype=np.int64)])
    x += rng.normal(scale=noise, size=x.shape).astype(np.float32)

    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


def make_moons_data(n_samples: int, noise: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.datasets import make_moons

        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return x.astype(np.float32), y.astype(np.int64)
    except Exception:
        return _make_moons_local(n_samples=n_samples, noise=noise, random_state=random_state)


def _build_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _train_one(
    name: str,
    optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_indices: list[np.ndarray],
    seed: int,
) -> tuple[nn.Module, float, float]:
    model = _build_model(seed=seed)
    params = list(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for idx in batch_indices:
        xb = x_train[idx]
        yb = y_train[idx]

        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()

        grads = [p.grad for p in params]
        with torch.no_grad():
            optimizer.step(params, grads)

        for p in params:
            p.grad.zero_()

    with torch.no_grad():
        train_preds = (torch.sigmoid(model(x_train).squeeze(1)) >= 0.5).float()
        test_preds = (torch.sigmoid(model(x_test).squeeze(1)) >= 0.5).float()
        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

    print(f"{name} final train accuracy: {train_acc:.4f}")
    print(f"{name} final test accuracy: {test_acc:.4f}")
    return model, train_acc, test_acc


def generate_moons_decision_boundary(output_dir: str | Path) -> tuple[str, dict[str, tuple[float, float]]]:
    output_path = Path(output_dir) / "moons_decision_boundary.png"
    x_np, y_np = make_moons_data(n_samples=1000, noise=0.2, random_state=0)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_np))
    split = int(0.8 * len(x_np))
    train_idx = perm[:split]
    test_idx = perm[split:]

    x_train = torch.tensor(x_np[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_np[train_idx], dtype=torch.float32)
    x_test = torch.tensor(x_np[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y_np[test_idx], dtype=torch.float32)

    batch_size = 128
    steps = 1000
    batch_rng = np.random.default_rng(0)
    batches = [
        batch_rng.choice(len(train_idx), size=batch_size, replace=False)
        for _ in range(steps)
    ]

    experiments = {
        "SGD": SGD(lr=0.1),
        "Momentum": Momentum(lr=0.1, beta=0.9),
        "AdamW": AdamW(lr=0.01, weight_decay=0.01),
    }

    metrics: dict[str, tuple[float, float]] = {}
    models: dict[str, nn.Module] = {}
    for name, optimizer in experiments.items():
        model, train_acc, test_acc = _train_one(
            name=name,
            optimizer=optimizer,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            batch_indices=batches,
            seed=0,
        )
        models[name] = model
        metrics[name] = (train_acc, test_acc)

    param_count = sum(p.numel() for p in models["SGD"].parameters())
    print(f"MLP parameter count: {param_count}")

    x_min, x_max = x_np[:, 0].min() - 0.6, x_np[:, 0].max() + 0.6
    y_min, y_max = x_np[:, 1].min() - 0.6, x_np[:, 1].max() + 0.6
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    panel_outputs: list[tuple[str, np.ndarray]] = []
    for name in ["SGD", "Momentum", "AdamW"]:
        with torch.no_grad():
            probs = torch.sigmoid(models[name](grid_tensor).squeeze(1)).cpu().numpy()
        panel_outputs.append((name, probs.reshape(xx.shape)))

    path = plot_decision_boundary_panels(
        outputs=panel_outputs,
        xx=xx,
        yy=yy,
        x_data=x_np,
        y_data=y_np,
        output_path=output_path,
    )
    return path, metrics

