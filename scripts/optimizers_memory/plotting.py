"""Plot helpers shared by optimizer-memory experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: plt.Figure, output_path: str | Path, use_tight_layout: bool = True) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_contour_trajectories(
    loss_fn,
    trajectories: dict[str, np.ndarray],
    output_path: str | Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(xlim[0], xlim[1], 260)
    y = np.linspace(ylim[0], ylim[1], 260)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = loss_fn(np.array([xx[i, j], yy[i, j]], dtype=np.float64))

    ax.contour(xx, yy, zz, levels=28, linewidths=0.8, colors="0.7")

    color_cycle = {
        "SGD": "#1f77b4",
        "Momentum": "#ff7f0e",
        "RMSProp": "#2ca02c",
        "Adam": "#d62728",
        "AdamW": "#9467bd",
    }
    for name, traj in trajectories.items():
        color = color_cycle.get(name, None)
        ax.plot(traj[:, 0], traj[:, 1], label=name, linewidth=2.2, color=color)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=45, marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=55, marker="X")

    ax.scatter([0.0], [0.0], c="black", s=65, marker="*", label="Minimum")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(frameon=True)
    return save_figure(fig, output_path)


def plot_line_series(
    series: dict[str, np.ndarray],
    output_path: str | Path,
    title: str,
    ylabel: str,
    xlabel: str = "Step",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in series.items():
        ax.plot(values, linewidth=2.0, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, output_path)


def plot_decision_boundary_panels(
    outputs: list[tuple[str, np.ndarray]],
    xx: np.ndarray,
    yy: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_path: str | Path,
) -> str:
    fig, axes = plt.subplots(1, len(outputs), figsize=(15, 4.8), sharex=True, sharey=True)
    if len(outputs) == 1:
        axes = [axes]

    for ax, (name, probs) in zip(axes, outputs):
        contour = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.7, vmin=0.0, vmax=1.0)
        ax.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.2)
        ax.scatter(
            x_data[:, 0],
            x_data[:, 1],
            c=y_data,
            cmap="bwr",
            edgecolors="k",
            s=18,
            alpha=0.75,
            linewidth=0.35,
        )
        ax.set_title(name)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    cbar = fig.colorbar(contour, ax=axes, shrink=0.85)
    cbar.set_label("Predicted P(class=1)")
    fig.subplots_adjust(wspace=0.18)
    return save_figure(fig, output_path, use_tight_layout=False)
