"""
Create an interactive parallel-coordinates plot from experiment JSON files.

By default, the script reads JSON files from ./results and writes:
    ./results/analysis_parameters/parameter_analysis.html

Examples
--------
    python analyse_parameters.py
    python analyse_parameters.py --input-dir results
    python analyse_parameters.py --input-dir results --exclude-config seed num_runs
    python analyse_parameters.py --color-by accuracy_mean
    python analyse_parameters.py --show-files
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px


RESULT_DIR = Path("results")
PLOT_DIR = RESULT_DIR / "analysis_parameters"

SUMMARY_METRICS = ("accuracy_mean", "macro_f1_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive parallel-coordinates plot from experiment JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULT_DIR,
        help=f"Folder containing experiment JSON files (default: {RESULT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output HTML file (default: {PLOT_DIR / 'parameter_analysis.html'}).",
    )
    parser.add_argument(
        "--exclude-config",
        nargs="+",
        default=[],
        metavar="PARAMETER",
        help="Configuration parameter(s) to exclude, in addition to automatically excluded constants.",
    )
    parser.add_argument(
        "--color-by",
        choices=SUMMARY_METRICS,
        default="macro_f1_mean",
        help="Summary metric used to color the lines (default: macro_f1_mean).",
    )
    return parser.parse_args()


def load_experiments(input_dir: Path) -> list[dict[str, Any]]:
    """Load JSON files directly inside input_dir."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    # Avoid accidentally treating generated JSON files in the output folder as inputs.
    files = [path for path in files if path.parent.resolve() != PLOT_DIR.resolve()]

    if not files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    experiments = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            print(f"Warning: skipping {path.name}: top-level JSON value is not an object.", file=sys.stderr)
            continue

        config = data.get("config", {})
        summary = data.get("summary", {})
        if not isinstance(config, dict) or not isinstance(summary, dict):
            print(f"Warning: skipping {path.name}: 'config' or 'summary' is not an object.", file=sys.stderr)
            continue

        row = dict(config)
        for metric in SUMMARY_METRICS:
            value = summary.get(metric)
            if value is None:
                raise ValueError(f"{path.name} is missing summary.{metric}")
            try:
                row[metric] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name} has a non-numeric summary.{metric}: {value!r}"
                ) from exc

        row["__filename"] = path.name
        experiments.append(row)

    if not experiments:
        raise ValueError(f"No valid experiment JSON files found in: {input_dir}")

    return experiments


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def values_equal(a: Any, b: Any) -> bool:
    """Compare JSON values while treating equivalent numeric types as equal."""
    if is_missing(a) and is_missing(b):
        return True
    if is_missing(a) or is_missing(b):
        return False
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        if isinstance(b, (int, float)) and not isinstance(b, bool):
            return float(a) == float(b)
    return type(a) is type(b) and a == b


def is_constant(values: list[Any]) -> bool:
    """Return True if every experiment has the same value for this parameter."""
    return all(values_equal(values[0], value) for value in values[1:])


def is_numeric_column(values: list[Any]) -> bool:
    """A column is numeric if every non-missing value is numeric (excluding bool)."""
    present = [value for value in values if not is_missing(value)]
    return bool(present) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in present
    )


def prepare_dimensions(
    experiments: list[dict[str, Any]],
    excluded: set[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[int, str]]]:
    """
    Convert config values to Plotly-compatible numeric dimensions.

    Categorical dimensions are encoded as integers and get explicit tick labels.
    Missing categorical values get their own category; missing numeric values
    are replaced with the median of that dimension.
    """
    all_config_keys = sorted(
        {key for experiment in experiments for key in experiment if not key.startswith("__")}
        - set(SUMMARY_METRICS)
    )

    rows = [dict(experiment) for experiment in experiments]
    dimensions: list[str] = []
    category_labels: dict[str, dict[int, str]] = {}

    for key in all_config_keys:
        if key in excluded:
            continue

        values = [row.get(key) for row in rows]
        if is_constant(values):
            print(f"Automatically excluding constant parameter: {key} = {values[0]!r}")
            continue

        dimensions.append(key)

        if is_numeric_column(values):
            numeric_values = [
                float(value) for value in values if not is_missing(value)
            ]
            median = float(pd.Series(numeric_values).median())
            for row in rows:
                value = row.get(key)
                row[key] = median if is_missing(value) else float(value)
        else:
            # Use stable, readable labels rather than relying on arbitrary object reprs.
            labels = sorted(
                {"(missing)" if is_missing(value) else str(value) for value in values}
            )
            mapping = {label: index for index, label in enumerate(labels)}
            category_labels[key] = {index: label for label, index in mapping.items()}
            for row in rows:
                value = row.get(key)
                label = "(missing)" if is_missing(value) else str(value)
                row[key] = mapping[label]

    return rows, dimensions, category_labels


def main() -> None:
    args = parse_args()
    output_path = args.output or (PLOT_DIR / "parameter_analysis.html")

    experiments = load_experiments(args.input_dir)
    explicitly_excluded = set(args.exclude_config)

    # Warn about misspelled or absent exclusions.
    available_config = {
        key
        for experiment in experiments
        for key in experiment
        if not key.startswith("__") and key not in SUMMARY_METRICS
    }
    unknown = explicitly_excluded - available_config
    if unknown:
        print(
            "Warning: requested exclusions were not found: "
            + ", ".join(sorted(unknown)),
            file=sys.stderr,
        )

    rows, config_dimensions, category_labels = prepare_dimensions(
        experiments, explicitly_excluded
    )

    # Summary metrics are always included as axes.
    dimensions = config_dimensions + list(SUMMARY_METRICS)
    if not config_dimensions:
        print("Warning: all configuration parameters were excluded; plotting summary metrics only.")

    frame = pd.DataFrame(rows)
    plot_dimensions = []

    for key in dimensions:
        dimension: dict[str, Any] = {
            "label": key,
            "values": frame[key].tolist(),
        }
        if key in category_labels:
            labels = category_labels[key]
            tickvals = sorted(labels)
            dimension["tickvals"] = tickvals
            dimension["ticktext"] = [labels[value] for value in tickvals]
        plot_dimensions.append(dimension)

    fig = px.parallel_coordinates(
        frame,
        dimensions=dimensions,
        color=args.color_by,
        color_continuous_scale="Viridis",
        labels={
            "accuracy_mean": "Accuracy mean",
            "macro_f1_mean": "Macro F1 mean",
        },
    )

    # Plotly Express does not consistently expose categorical tick labels for
    # parallel coordinates, so set the dimension metadata explicitly.
    fig.update_traces(dimensions=plot_dimensions)

    fig.update_layout(
        title=f"Experiment parameter analysis (colored by {args.color_by})",
        coloraxis_colorbar_title=args.color_by.replace("_", " ").title(),
        autosize=True,
        height=max(650, 100 + 35 * len(dimensions)),
        margin=dict(l=80, r=100, t=90, b=60),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"\nLoaded {len(experiments)} experiments.")
    print(f"Included dimensions: {', '.join(dimensions)}")
    if explicitly_excluded:
        print(f"Manually excluded: {', '.join(sorted(explicitly_excluded))}")
    print(f"Saved interactive plot to: {output_path}")


if __name__ == "__main__":
    main()
