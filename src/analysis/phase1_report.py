"""Aggregate Phase 1 results and test the registered hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.metrics import compute_sample_efficiency
from src.analysis.plot_results import load_run_artifacts, save_phase_comparison_plot
from src.analysis.statistics import holm_bonferroni, mann_whitney_u, rank_biserial_effect_size


def _records_with_nulls(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to records while mapping NaN values to None."""

    cleaned = frame.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


def collect_run_summaries(results_root: str | Path) -> pd.DataFrame:
    """Collect all run-level `summary.json` files under a results directory."""

    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for summary_path in root.glob("phase1*/**/summary.json"):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_frame, _ = load_run_artifacts(summary_path.parent)
        tau_clip = float(row.get("tau_clip", 0.0) or 0.0)
        row["condition_label"] = (
            f"clipped_dense_tau_{tau_clip:.2f}"
            if row["reward_type"] == "clipped_dense"
            else row["reward_type"]
        )
        success_curve = metrics_frame["metrics_success_rate"].tolist()
        row["sample_efficiency_iteration"] = compute_sample_efficiency(success_curve, target_threshold=0.8)
        row["best_training_success_rate"] = float(metrics_frame["metrics_success_rate"].max())
        row["mean_training_negative_ratio"] = float(metrics_frame["reward_negative_ratio"].mean())
        row["run_dir"] = str(summary_path.parent)
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_stats(env_frame: pd.DataFrame, metric_name: str) -> list[dict[str, Any]]:
    available_conditions = sorted(env_frame["condition_label"].unique().tolist())
    comparisons: list[tuple[str, str]] = []
    if "dense" in available_conditions and "binary" in available_conditions:
        comparisons.append(("dense", "binary"))
    for condition in available_conditions:
        if condition.startswith("clipped_dense_tau_") and "binary" in available_conditions:
            comparisons.append((condition, "binary"))
        if condition.startswith("clipped_dense_tau_") and "dense" in available_conditions:
            comparisons.append((condition, "dense"))
    raw_p_values: list[float] = []
    pending: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_values = env_frame.loc[env_frame["condition_label"] == left, metric_name].dropna().tolist()
        right_values = env_frame.loc[env_frame["condition_label"] == right, metric_name].dropna().tolist()
        if not left_values or not right_values:
            continue
        mw = mann_whitney_u(left_values, right_values)
        effect = rank_biserial_effect_size(left_values, right_values)
        raw_p_values.append(mw.p_value)
        pending.append(
            {
                "metric": metric_name,
                "comparison": f"{left}_vs_{right}",
                "u_statistic": mw.u_statistic,
                "raw_p_value": mw.p_value,
                "effect_size_rank_biserial": effect,
            }
        )
    corrected = holm_bonferroni(raw_p_values) if raw_p_values else []
    for item, corrected_p in zip(pending, corrected):
        item["holm_corrected_p_value"] = corrected_p
    return pending


def build_phase1_report(results_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build an aggregate Phase 1 report with pairwise tests and plot outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = collect_run_summaries(results_root)
    if frame.empty:
        raise FileNotFoundError(f"No summary.json files found under {results_root}.")

    env_reports: dict[str, Any] = {}
    for env_name, env_frame in frame.groupby("env_name"):
        env_reports[env_name] = {
            "reward_means": _records_with_nulls(
                env_frame.groupby("condition_label", as_index=False)
                .agg(
                    final_success_rate_mean=("final_success_rate", "mean"),
                    final_success_rate_std=("final_success_rate", "std"),
                    best_training_success_rate_mean=("best_training_success_rate", "mean"),
                    mean_cumulative_reward_mean=("mean_cumulative_reward", "mean"),
                    gradient_variance_mean=("mean_gradient_variance_proxy", "mean"),
                    negative_reward_ratio_mean=("mean_training_negative_ratio", "mean"),
                    sample_efficiency_iteration_mean=("sample_efficiency_iteration", "mean"),
                )
            ),
            "pairwise_statistics": _pairwise_stats(env_frame, "final_success_rate")
            + _pairwise_stats(env_frame, "sample_efficiency_iteration")
            + _pairwise_stats(env_frame, "mean_gradient_variance_proxy"),
        }

    plot_paths = [str(path) for path in save_phase_comparison_plot(frame, output_path / "plots")]
    report = {
        "results_root": str(results_root),
        "run_count": int(len(frame)),
        "environments": env_reports,
        "generated_plots": plot_paths,
    }
    (output_path / "phase1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    frame.to_csv(output_path / "phase1_runs.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Directory containing run outputs.")
    parser.add_argument(
        "--output-dir",
        default="results/phase1_report",
        help="Directory for aggregate CSV/JSON reports and plots.",
    )
    args = parser.parse_args()
    report = build_phase1_report(args.results_root, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
