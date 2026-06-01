"""Toy 2D optimizer experiments for visual intuition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from losses import (
    bowl_grad,
    bowl_loss,
    noisy_quadratic_grad,
    noisy_quadratic_loss,
    ravine_grad,
    ravine_loss,
    scale_grad,
    scale_loss,
)
from optimizers import Adam, Momentum, RMSProp, SGD
from plotting import plot_contour_trajectories, plot_line_series, save_figure


def _run_trajectory(optimizer, grad_fn, start_point, steps: int) -> np.ndarray:
    point = np.array(start_point, dtype=np.float64)
    trajectory = [point.copy()]
    for _ in range(steps):
        grad = grad_fn(point)
        optimizer.step([point], [grad])
        trajectory.append(point.copy())
    return np.array(trajectory)


def generate_sgd_bowl(output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "sgd_bowl.png"
    traj = _run_trajectory(
        optimizer=SGD(lr=0.18),
        grad_fn=bowl_grad,
        start_point=[4.0, 3.0],
        steps=30,
    )
    return plot_contour_trajectories(
        loss_fn=bowl_loss,
        trajectories={"SGD": traj},
        output_path=output_path,
        xlim=(-4.5, 4.5),
        ylim=(-3.5, 3.5),
        title="SGD on f(x, y) = 0.5(x² + y²)",
    )


def generate_momentum_ravine(output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "momentum_ravine.png"
    trajectories = {
        "SGD": _run_trajectory(SGD(lr=0.035), ravine_grad, [4.0, 1.0], steps=45),
        "Momentum": _run_trajectory(
            Momentum(lr=0.035, beta=0.9),
            ravine_grad,
            [4.0, 1.0],
            steps=45,
        ),
    }
    return plot_contour_trajectories(
        loss_fn=ravine_loss,
        trajectories=trajectories,
        output_path=output_path,
        xlim=(-4.5, 4.5),
        ylim=(-1.5, 1.5),
        title="SGD vs Momentum in a Narrow Ravine",
    )


def generate_rmsprop_scale(output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "rmsprop_scale.png"
    trajectories = {
        "SGD": _run_trajectory(SGD(lr=0.035), scale_grad, [6.0, 1.0], steps=60),
        "RMSProp": _run_trajectory(
            RMSProp(lr=0.12, beta=0.9),
            scale_grad,
            [6.0, 1.0],
            steps=60,
        ),
    }
    return plot_contour_trajectories(
        loss_fn=scale_loss,
        trajectories=trajectories,
        output_path=output_path,
        xlim=(-6.5, 6.5),
        ylim=(-1.5, 1.5),
        title="SGD vs RMSProp with Different Coordinate Scales",
    )


def generate_adam_noisy(output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "adam_noisy.png"
    steps = 80
    rng = np.random.default_rng(7)
    noise = rng.normal(loc=0.0, scale=0.8, size=(steps, 2))

    optimizers = {
        "SGD": SGD(lr=0.04),
        "Momentum": Momentum(lr=0.04, beta=0.9),
        "RMSProp": RMSProp(lr=0.08, beta=0.9),
        "Adam": Adam(lr=0.08),
    }

    trajectories = {}
    for name, optimizer in optimizers.items():
        point = np.array([4.0, 2.0], dtype=np.float64)
        traj = [point.copy()]
        for t in range(steps):
            grad = noisy_quadratic_grad(point) + noise[t]
            optimizer.step([point], [grad])
            traj.append(point.copy())
        trajectories[name] = np.array(traj)

    return plot_contour_trajectories(
        loss_fn=noisy_quadratic_loss,
        trajectories=trajectories,
        output_path=output_path,
        xlim=(-4.5, 4.5),
        ylim=(-2.5, 2.5),
        title="Noisy-Gradient Trajectories: SGD, Momentum, RMSProp, Adam",
    )


def generate_bias_correction(output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "bias_correction.png"
    steps = 50

    series = {}
    for beta in (0.9, 0.99):
        m = 0.0
        raw = np.zeros(steps, dtype=np.float64)
        corrected = np.zeros(steps, dtype=np.float64)
        for t in range(1, steps + 1):
            m = beta * m + (1.0 - beta)
            raw[t - 1] = m
            corrected[t - 1] = m / (1.0 - beta**t)
        series[f"raw EMA (beta={beta})"] = raw
        series[f"bias-corrected (beta={beta})"] = corrected

    return plot_line_series(
        series=series,
        output_path=output_path,
        title="Bias Correction for Exponential Moving Averages",
        ylabel="Value",
    )


def generate_toy_figures(output_dir: str | Path) -> list[str]:
    return [
        generate_sgd_bowl(output_dir),
        generate_momentum_ravine(output_dir),
        generate_rmsprop_scale(output_dir),
        generate_adam_noisy(output_dir),
        generate_bias_correction(output_dir),
    ]

