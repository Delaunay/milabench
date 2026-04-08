#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
DEFAULT_BATCH_ROOT = Path("/tmp/milabench/artifacts/benchmarks/batchsize_multiseed_seeds0to4")
DEFAULT_PRIOR_ROOT = Path("/tmp/milabench/artifacts/benchmarks/multiseed_retest_seeds012")

BATCH_ORDER = ["control", "b32768", "b16384", "b8192"]
BATCH_LABELS = {
    "control": "Control 65536",
    "b32768": "Batch 32768",
    "b16384": "Batch 16384",
    "b8192": "Batch 8192",
}
BATCH_COLORS = {
    "control": "#111111",
    "b32768": "#0b7285",
    "b16384": "#d9480f",
    "b8192": "#7b2cbf",
}

PRIOR_ORDER = ["control", "target_fold", "bf16", "fp16", "trainint20"]
PRIOR_LABELS = {
    "control": "Control",
    "target_fold": "Target Fold",
    "bf16": "BF16",
    "fp16": "FP16",
    "trainint20": "Train Int 20",
}
PRIOR_COLORS = {
    "control": "#111111",
    "target_fold": "#0b7285",
    "bf16": "#2b6cb0",
    "fp16": "#e67e22",
    "trainint20": "#7f8c8d",
}

SUMMARY_LABELS = {
    "prior:target_fold": "Target Fold",
    "prior:bf16": "BF16",
    "prior:fp16": "FP16",
    "prior:trainint20": "Train Int 20",
    "batch:b32768": "Batch 32768",
    "batch:b16384": "Batch 16384",
    "batch:b8192": "Batch 8192",
}
SUMMARY_FAMILIES = {
    "prior:target_fold": "Earlier Candidate Family",
    "prior:bf16": "Earlier Candidate Family",
    "prior:fp16": "Earlier Candidate Family",
    "prior:trainint20": "Earlier Candidate Family",
    "batch:b32768": "Batch-Size Family",
    "batch:b16384": "Batch-Size Family",
    "batch:b8192": "Batch-Size Family",
}
SUMMARY_COLORS = {
    "prior:target_fold": "#0b7285",
    "prior:bf16": "#2b6cb0",
    "prior:fp16": "#e67e22",
    "prior:trainint20": "#7f8c8d",
    "batch:b32768": "#0b7285",
    "batch:b16384": "#d9480f",
    "batch:b8192": "#7b2cbf",
}


def load_record(raw_line: str) -> dict | None:
    cleaned = ANSI_ESCAPE_RE.sub("", raw_line).strip()
    if "{" not in cleaned:
        return None

    payload = cleaned[cleaned.find("{") :]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def parse_trace(path: Path, experiment: str, seed: int, label_map: dict[str, str]) -> list[dict]:
    returns = []
    rates = []
    progress = []

    with path.open("r", errors="ignore") as handle:
        for raw_line in handle:
            record = load_record(raw_line)
            if record is None:
                continue

            if "returns" in record:
                returns.append(record["returns"])
            elif "rate" in record:
                rates.append(record["rate"])
            elif record.get("task") == "early_stop" and "progress" in record:
                marker = record["progress"]
                if isinstance(marker, list) and marker:
                    progress.append(marker[0])

    count = min(len(returns), len(rates), len(progress) if progress else 10**9)
    rows = []
    for idx in range(count):
        rows.append(
            {
                "experiment": experiment,
                "label": label_map.get(experiment, experiment),
                "seed": seed,
                "snapshot": progress[idx] if progress else idx,
                "returns": returns[idx],
                "rate": rates[idx],
            }
        )
    return rows


def collect_traces(root: Path, experiment_order: list[str], label_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment in experiment_order:
        experiment_dir = root / experiment
        if not experiment_dir.is_dir():
            continue
        for seed_dir in sorted(experiment_dir.glob("seed*")):
            stdout_path = seed_dir / "stdout.txt"
            if not stdout_path.exists():
                continue
            seed = int(seed_dir.name.removeprefix("seed"))
            rows.extend(parse_trace(stdout_path, experiment, seed, label_map))

    if not rows:
        raise RuntimeError(f"No trace rows were parsed from {root}")

    return pd.DataFrame(rows)


def aggregate_traces(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["experiment", "label", "snapshot"], observed=True)
        .agg(
            reward_mean=("returns", "mean"),
            reward_std=("returns", lambda s: s.std(ddof=0)),
            rate_mean=("rate", "mean"),
            rate_std=("rate", lambda s: s.std(ddof=0)),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    grouped["reward_std"] = grouped["reward_std"].fillna(0.0)
    grouped["rate_std"] = grouped["rate_std"].fillna(0.0)
    grouped["reward_lower"] = grouped["reward_mean"] - grouped["reward_std"]
    grouped["reward_upper"] = grouped["reward_mean"] + grouped["reward_std"]
    grouped["rate_lower"] = grouped["rate_mean"] - grouped["rate_std"]
    grouped["rate_upper"] = grouped["rate_mean"] + grouped["rate_std"]
    return grouped


def summarize_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["experiment", "seed"], observed=True)
        .agg(
            returns_last=("returns", "last"),
            returns_mean=("returns", "mean"),
            returns_max=("returns", "max"),
            rate_mean=("rate", "mean"),
            rate_last=("rate", "last"),
        )
        .reset_index()
    )


def make_metric_chart(
    data: pd.DataFrame,
    metric_prefix: str,
    y_title: str,
    title: str,
    label_order: list[str],
    color_scale: alt.Scale,
    width: int = 880,
    height: int = 280,
) -> alt.Chart:
    base = alt.Chart(data).encode(
        x=alt.X("snapshot:Q", title="Logging Snapshot", axis=alt.Axis(tickMinStep=1)),
    )

    band = base.mark_area(opacity=0.10).encode(
        y=alt.Y(f"{metric_prefix}_lower:Q", title=y_title),
        y2=f"{metric_prefix}_upper:Q",
        color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
    )

    rules = (
        alt.Chart(data.loc[data["snapshot"] % 10 == 0].copy())
        .mark_rule(opacity=0.30)
        .encode(
            x=alt.X("snapshot:Q", title="Logging Snapshot", axis=alt.Axis(tickMinStep=1)),
            y=f"{metric_prefix}_lower:Q",
            y2=f"{metric_prefix}_upper:Q",
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
        )
    )

    line = base.mark_line(strokeWidth=2.5).encode(
        y=alt.Y(f"{metric_prefix}_mean:Q", title=y_title),
        color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
    )

    return (band + rules + line).properties(width=width, height=height, title=title)


def build_manual_legend(label_order: list[str], color_scale: alt.Scale, title: str) -> alt.Chart:
    legend_df = pd.DataFrame(
        {
            "label": label_order,
            "x_start": [0.0] * len(label_order),
            "x_end": [24.0] * len(label_order),
            "text_x": [30.0] * len(label_order),
        }
    )

    swatches = (
        alt.Chart(legend_df)
        .mark_rule(strokeWidth=4)
        .encode(
            x=alt.X("x_start:Q", axis=None, scale=alt.Scale(domain=[0, 160], nice=False)),
            x2="x_end:Q",
            y=alt.Y("label:N", sort=label_order, axis=None),
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
        )
    )

    labels_chart = (
        alt.Chart(legend_df)
        .mark_text(align="left", baseline="middle", fontSize=12)
        .encode(
            x=alt.X("text_x:Q", axis=None, scale=alt.Scale(domain=[0, 160], nice=False)),
            y=alt.Y("label:N", sort=label_order, axis=None),
            text="label:N",
            color=alt.value("#222222"),
        )
    )

    return (swatches + labels_chart).properties(width=240, height=26 * len(label_order), title=title)


def build_batch_figure(aggregate_df: pd.DataFrame) -> alt.Chart:
    label_order = [BATCH_LABELS[key] for key in BATCH_ORDER if key in aggregate_df["experiment"].unique()]
    color_scale = alt.Scale(
        domain=label_order,
        range=[BATCH_COLORS[key] for key in BATCH_ORDER if key in aggregate_df["experiment"].unique()],
    )

    reward_chart = make_metric_chart(
        aggregate_df,
        "reward",
        "Mean Return (mean ± 1 sd across seeds)",
        "Batch Sweep Reward Evolution",
        label_order,
        color_scale,
    )
    rate_chart = make_metric_chart(
        aggregate_df,
        "rate",
        "Throughput (items/s, mean ± 1 sd across seeds)",
        "Batch Sweep Throughput Evolution",
        label_order,
        color_scale,
    )
    legend = build_manual_legend(label_order, color_scale, "Batch Sweep")
    return alt.hconcat(alt.vconcat(reward_chart, rate_chart), legend, spacing=20).properties(
        title="Multi-Seed Batch-Size Sweep"
    )


def build_summary_dataframe(
    prior_seed_summary: pd.DataFrame,
    batch_seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    prior_control = prior_seed_summary.loc[prior_seed_summary["experiment"] == "control"].copy()
    batch_control = batch_seed_summary.loc[batch_seed_summary["experiment"] == "control"].copy()

    for experiment in PRIOR_ORDER:
        if experiment == "control":
            continue
        candidate = prior_seed_summary.loc[prior_seed_summary["experiment"] == experiment].copy()
        merged = candidate.merge(prior_control, on="seed", suffixes=("_cand", "_ctrl"))
        rows.append(
            {
                "key": f"prior:{experiment}",
                "label": SUMMARY_LABELS[f"prior:{experiment}"],
                "family": SUMMARY_FAMILIES[f"prior:{experiment}"],
                "rate_delta_pct_mean": ((merged["rate_mean_cand"] / merged["rate_mean_ctrl"]) - 1.0).mean() * 100.0,
                "rate_delta_pct_std": ((merged["rate_mean_cand"] / merged["rate_mean_ctrl"]) - 1.0).std(ddof=0) * 100.0,
                "reward_delta_mean": (merged["returns_mean_cand"] - merged["returns_mean_ctrl"]).mean(),
                "reward_delta_std": (merged["returns_mean_cand"] - merged["returns_mean_ctrl"]).std(ddof=0),
                "seed_count": len(merged),
            }
        )

    for experiment in BATCH_ORDER:
        if experiment == "control":
            continue
        candidate = batch_seed_summary.loc[batch_seed_summary["experiment"] == experiment].copy()
        merged = candidate.merge(batch_control, on="seed", suffixes=("_cand", "_ctrl"))
        rows.append(
            {
                "key": f"batch:{experiment}",
                "label": SUMMARY_LABELS[f"batch:{experiment}"],
                "family": SUMMARY_FAMILIES[f"batch:{experiment}"],
                "rate_delta_pct_mean": ((merged["rate_mean_cand"] / merged["rate_mean_ctrl"]) - 1.0).mean() * 100.0,
                "rate_delta_pct_std": ((merged["rate_mean_cand"] / merged["rate_mean_ctrl"]) - 1.0).std(ddof=0) * 100.0,
                "reward_delta_mean": (merged["returns_mean_cand"] - merged["returns_mean_ctrl"]).mean(),
                "reward_delta_std": (merged["returns_mean_cand"] - merged["returns_mean_ctrl"]).std(ddof=0),
                "seed_count": len(merged),
            }
        )

    return pd.DataFrame(rows)


def build_summary_points_dataframe(
    prior_seed_summary: pd.DataFrame,
    batch_seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    prior_control = prior_seed_summary.loc[prior_seed_summary["experiment"] == "control"].copy()
    batch_control = batch_seed_summary.loc[batch_seed_summary["experiment"] == "control"].copy()

    for experiment in PRIOR_ORDER:
        if experiment == "control":
            continue
        candidate = prior_seed_summary.loc[prior_seed_summary["experiment"] == experiment].copy()
        merged = candidate.merge(prior_control, on="seed", suffixes=("_cand", "_ctrl"))
        for _, row in merged.iterrows():
            rows.append(
                {
                    "key": f"prior:{experiment}",
                    "label": SUMMARY_LABELS[f"prior:{experiment}"],
                    "family": SUMMARY_FAMILIES[f"prior:{experiment}"],
                    "seed": int(row["seed"]),
                    "rate_delta_pct": ((row["rate_mean_cand"] / row["rate_mean_ctrl"]) - 1.0) * 100.0,
                    "reward_delta": row["returns_mean_cand"] - row["returns_mean_ctrl"],
                    "returns_last_delta": row["returns_last_cand"] - row["returns_last_ctrl"],
                }
            )

    for experiment in BATCH_ORDER:
        if experiment == "control":
            continue
        candidate = batch_seed_summary.loc[batch_seed_summary["experiment"] == experiment].copy()
        merged = candidate.merge(batch_control, on="seed", suffixes=("_cand", "_ctrl"))
        for _, row in merged.iterrows():
            rows.append(
                {
                    "key": f"batch:{experiment}",
                    "label": SUMMARY_LABELS[f"batch:{experiment}"],
                    "family": SUMMARY_FAMILIES[f"batch:{experiment}"],
                    "seed": int(row["seed"]),
                    "rate_delta_pct": ((row["rate_mean_cand"] / row["rate_mean_ctrl"]) - 1.0) * 100.0,
                    "reward_delta": row["returns_mean_cand"] - row["returns_mean_ctrl"],
                    "returns_last_delta": row["returns_last_cand"] - row["returns_last_ctrl"],
                }
            )

    return pd.DataFrame(rows)


def build_day_summary_chart(summary_df: pd.DataFrame, summary_points_df: pd.DataFrame) -> alt.Chart:
    label_order = [SUMMARY_LABELS[key] for key in SUMMARY_LABELS if key in summary_df["key"].tolist()]
    color_scale = alt.Scale(
        domain=label_order,
        range=[SUMMARY_COLORS[key] for key in SUMMARY_LABELS if key in summary_df["key"].tolist()],
    )

    seed_scatter = (
        alt.Chart(summary_points_df)
        .mark_circle(size=70, opacity=0.35)
        .encode(
            x=alt.X("rate_delta_pct:Q", title="Average Throughput Delta vs Family Control (%)"),
            y=alt.Y("reward_delta:Q", title="Average Return-Mean Delta vs Family Control"),
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
            shape=alt.Shape("family:N", title="Experiment Family"),
            tooltip=[
                alt.Tooltip("label:N"),
                alt.Tooltip("family:N"),
                alt.Tooltip("seed:Q"),
                alt.Tooltip("rate_delta_pct:Q", format=".2f"),
                alt.Tooltip("reward_delta:Q", format=".3f"),
                alt.Tooltip("returns_last_delta:Q", format=".3f"),
            ],
        )
    )

    average_scatter = (
        alt.Chart(summary_df)
        .mark_circle(size=220, opacity=0.95, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("rate_delta_pct_mean:Q", title="Average Throughput Delta vs Family Control (%)"),
            y=alt.Y("reward_delta_mean:Q", title="Average Return-Mean Delta vs Family Control"),
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
            shape=alt.Shape("family:N", title="Experiment Family"),
            tooltip=[
                alt.Tooltip("label:N"),
                alt.Tooltip("family:N"),
                alt.Tooltip("seed_count:Q"),
                alt.Tooltip("rate_delta_pct_mean:Q", format=".2f"),
                alt.Tooltip("rate_delta_pct_std:Q", format=".2f"),
                alt.Tooltip("reward_delta_mean:Q", format=".3f"),
                alt.Tooltip("reward_delta_std:Q", format=".3f"),
            ],
        )
        .properties(width=760, height=460, title="Candidate Summary Across Multi-Seed Families")
    )

    x_error = (
        alt.Chart(summary_df)
        .mark_rule(opacity=0.30)
        .encode(
            x="rate_delta_pct_mean:Q",
            x2="rate_delta_pct_upper:Q",
            y="reward_delta_mean:Q",
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
        )
    )

    x_error_left = (
        alt.Chart(summary_df)
        .mark_rule(opacity=0.30)
        .encode(
            x="rate_delta_pct_lower:Q",
            x2="rate_delta_pct_mean:Q",
            y="reward_delta_mean:Q",
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
        )
    )

    y_error = (
        alt.Chart(summary_df)
        .mark_rule(opacity=0.30)
        .encode(
            x="rate_delta_pct_mean:Q",
            y="reward_delta_lower:Q",
            y2="reward_delta_upper:Q",
            color=alt.Color("label:N", scale=color_scale, sort=label_order, legend=None),
        )
    )

    legend = build_manual_legend(label_order, color_scale, "Candidates")
    return alt.hconcat(
        (seed_scatter + x_error_left + x_error + y_error + average_scatter),
        legend,
        spacing=20,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot batch multi-seed sweep results and a day-end candidate summary.")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    args = parser.parse_args()

    batch_df = collect_traces(args.batch_root, BATCH_ORDER, BATCH_LABELS)
    batch_agg = aggregate_traces(batch_df)
    batch_seed_summary = summarize_per_seed(batch_df)

    prior_df = collect_traces(args.prior_root, PRIOR_ORDER, PRIOR_LABELS)
    prior_seed_summary = summarize_per_seed(prior_df)

    summary_df = build_summary_dataframe(prior_seed_summary, batch_seed_summary)
    summary_points_df = build_summary_points_dataframe(prior_seed_summary, batch_seed_summary)
    summary_df["rate_delta_pct_lower"] = summary_df["rate_delta_pct_mean"] - summary_df["rate_delta_pct_std"]
    summary_df["rate_delta_pct_upper"] = summary_df["rate_delta_pct_mean"] + summary_df["rate_delta_pct_std"]
    summary_df["reward_delta_lower"] = summary_df["reward_delta_mean"] - summary_df["reward_delta_std"]
    summary_df["reward_delta_upper"] = summary_df["reward_delta_mean"] + summary_df["reward_delta_std"]

    batch_prefix = args.batch_root / "batchsize_multiseed"
    batch_df.to_csv(batch_prefix.with_suffix(".raw.csv"), index=False)
    batch_agg.to_csv(batch_prefix.with_suffix(".agg.csv"), index=False)
    batch_chart = build_batch_figure(batch_agg)
    batch_chart.save(batch_prefix.with_suffix(".html"))
    batch_chart.save(batch_prefix.with_suffix(".png"))

    summary_prefix = args.batch_root / "day_end_summary"
    summary_df.to_csv(summary_prefix.with_suffix(".csv"), index=False)
    summary_points_df.to_csv(args.batch_root / "day_end_summary.seed_points.csv", index=False)
    summary_chart = build_day_summary_chart(summary_df, summary_points_df)
    summary_chart.save(summary_prefix.with_suffix(".html"))
    summary_chart.save(summary_prefix.with_suffix(".png"))

    summary_md = args.batch_root / "SUMMARY.md"
    top_batch = summary_df.loc[summary_df["family"] == "Batch-Size Family"].sort_values(
        ["reward_delta_mean", "rate_delta_pct_mean"], ascending=[False, False]
    )
    top_batch_label = top_batch.iloc[0]["label"] if not top_batch.empty else "n/a"
    top_batch_rate = top_batch.iloc[0]["rate_delta_pct_mean"] if not top_batch.empty else 0.0
    top_batch_reward = top_batch.iloc[0]["reward_delta_mean"] if not top_batch.empty else 0.0
    summary_md.write_text(
        "\n".join(
            [
                "# Batch-Size Multi-Seed Sweep",
                "",
                f"Seeds: `0-4`",
                "",
                f"Best batch candidate on this sweep: `{top_batch_label}`",
                f"- mean throughput delta vs control: `{top_batch_rate:.2f}%`",
                f"- mean return delta vs control: `{top_batch_reward:.4f}`",
                "",
                "Artifacts:",
                f"- `batchsize_multiseed.png`",
                f"- `day_end_summary.png`",
                f"- `batchsize_multiseed.raw.csv`",
                f"- `day_end_summary.csv`",
                "",
            ]
        )
        + "\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
