"""Generate all figures for the 'Optimizers Are Memory' post."""

from __future__ import annotations

from pathlib import Path

from experiments_linear import generate_adamw_weight_decay, generate_linear_regression_loss
from experiments_moons import generate_moons_decision_boundary
from experiments_toy import generate_toy_figures
from plotting import ensure_output_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = ensure_output_dir(repo_root / "assets" / "images" / "optimizers-memory")

    generated_paths: list[str] = []

    generated_paths.extend(generate_toy_figures(output_dir))
    generated_paths.append(generate_adamw_weight_decay(output_dir))

    linear_path, state_report = generate_linear_regression_loss(output_dir)
    generated_paths.append(linear_path)

    moons_path, moon_metrics = generate_moons_decision_boundary(output_dir)
    generated_paths.append(moons_path)

    print("\nGenerated figure files:")
    for path in generated_paths:
        print(f"- {path}")

    print("\nState-size report:")
    print(f"Model parameters: {state_report['Model parameters']}")
    print(f"SGD state values: {state_report['SGD state values']}")
    print(f"Momentum state values: {state_report['Momentum state values']}")
    print(f"RMSProp state values: {state_report['RMSProp state values']}")
    print(f"Adam state values: {state_report['Adam state values']}")
    print(f"AdamW state values: {state_report['AdamW state values']}")

    print("\nFinal two-moons accuracies:")
    for name in ["SGD", "Momentum", "AdamW"]:
        train_acc, test_acc = moon_metrics[name]
        print(f"- {name}: train={train_acc:.4f}, test={test_acc:.4f}")


if __name__ == "__main__":
    main()

