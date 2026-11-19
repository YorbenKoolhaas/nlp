"""
Creates a history plot from experiment JSON files.

By default, the script reads JSON files from ./results/experiment.json and writes:
    ./results/history_plots/experiment.png

Examples
--------
    python history.py results/experiment.json
    python history.py results/
    python history.py results/ --show-runs
    python history.py results/experiment.json --show-runs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("results")
PLOT_DIR = RESULT_DIR / "history_plots"

METRICS = {
    "training_loss": {
        "title": "Training Loss",
        "ylabel": "Loss",
        "color": "tab:blue",
    },
    "validation_accuracy": {
        "title": "Validation Accuracy",
        "ylabel": "Accuracy",
        "color": "tab:orange",
    },
    "validation_macro_f1": {
        "title": "Validation Macro F1",
        "ylabel": "Macro F1",
        "color": "tab:green",
    },
}


def load_results(json_path: Path) -> dict:
    """Load experiment results from a JSON file."""
    with json_path.open("r") as f:
        return json.load(f)


def plot_history(json_path: Path, show_runs: bool = False) -> None:
    """Plot the average training history across all runs."""
    data = load_results(json_path)
    runs = data["results"]

    if not runs:
        print(f"Skipping {json_path}: no runs found.")
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
    )

    for ax, (metric, settings) in zip(axes, METRICS.items()):
        histories = [
            run["history"][metric]
            for run in runs
            if metric in run.get("history", {})
        ]

        if not histories:
            ax.set_title(settings["title"])
            ax.text(
                0.5, 0.5, "No data available",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            continue

        # Ensure all runs have the same number of epochs.
        min_epochs = min(len(history) for history in histories)
        values = np.array([
            history[:min_epochs]
            for history in histories
        ])

        epochs = np.arange(1, min_epochs + 1)

        # Plot individual runs if requested.
        if show_runs:
            for history in values:
                ax.plot(
                    epochs,
                    history,
                    color=settings["color"],
                    alpha=0.2,
                    linewidth=1,
                )

        # Calculate mean and standard deviation.
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        ax.errorbar(
            epochs,
            mean,
            yerr=std,
            label="Mean ± Std",
            color=settings["color"],
            linewidth=2,
            marker="o",
            markersize=4,
            capsize=3,
        )

        ax.set_title(settings["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(settings["ylabel"])
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Add experiment configuration as the figure title.
    config = data.get("config", {})
    model = config.get("model", "Unknown")
    embedding = config.get("embedding", "Unknown")

    fig.suptitle(
        f"{model.upper()} | {embedding} | "
        f"{len(runs)} runs",
        fontsize=14,
    )

    fig.tight_layout()

    # Save using the original JSON filename.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{json_path.stem}.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training histories from experiment JSON files."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to a JSON file or a folder containing JSON files.",
    )

    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Show individual runs as transparent lines.",
    )

    args = parser.parse_args()

    if args.input.is_file():
        if args.input.suffix.lower() != ".json":
            parser.error("Input file must be a JSON file.")

        json_files = [args.input]

    elif args.input.is_dir():
        json_files = sorted(args.input.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {args.input}")
            return

    else:
        parser.error(f"Input path does not exist: {args.input}")

    for json_path in json_files:
        try:
            plot_history(json_path, show_runs=args.show_runs)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing {json_path}: {e}")


if __name__ == "__main__":
    main()
