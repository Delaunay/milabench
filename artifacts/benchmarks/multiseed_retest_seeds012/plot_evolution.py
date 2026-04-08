#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

EXPERIMENT_ORDER = [
    "control",
    "target_fold",
    "fp16",
    "bf16",
    "trainint20",
]

EXPERIMENT_LABELS = {
    "control": "Control",
    "target_fold": "Target Fold",
    "fp16": "FP16",
    "bf16": "BF16",
    "trainint20": "Train Int 20",
}

EXPERIMENT_COLORS = {
    "control": "#111111",
    "target_fold": "#0b7285",
    "fp16": "#e67e22",
    "bf16": "#2b6cb0",
    "trainint20": "#7f8c8d",
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


def parse_trace(path: Path, experiment: str, seed: int) -> list[dict]:
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
                "label": EXPERIMENT_LABELS.get(experiment, experiment),
                "seed": seed,
                "snapshot": progress[idx] if progress else idx,
                "returns": returns[idx],
                "rate": rates[idx],
            }
        )
    return rows


def collect_traces(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment in EXPERIMENT_ORDER:
        experiment_dir = root / experiment
        if not experiment_dir.is_dir():
            continue
        for seed_dir in sorted(experiment_dir.glob("seed*")):
            stdout_path = seed_dir / "stdout.txt"
            if not stdout_path.exists():
                continue
            seed = int(seed_dir.name.removeprefix("seed"))
            rows.extend(parse_trace(stdout_path, experiment, seed))

    if not rows:
        raise RuntimeError(f"No trace rows were parsed from {root}")

    df = pd.DataFrame(rows)
    df["label"] = pd.Categorical(
        df["label"],
        categories=[EXPERIMENT_LABELS[key] for key in EXPERIMENT_ORDER if key in df["experiment"].unique()],
        ordered=True,
    )
    return df


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


def make_metric_chart(
    data: pd.DataFrame,
    metric_prefix: str,
    y_title: str,
    title: str,
    color_scale: alt.Scale,
) -> alt.Chart:
    label_order = [EXPERIMENT_LABELS[key] for key in EXPERIMENT_ORDER if key in data["experiment"].unique()]

    x_encoding = alt.X("snapshot:Q", title="Logging Snapshot", axis=alt.Axis(tickMinStep=1))
    line_color = alt.Color("label:N", scale=color_scale, sort=label_order, legend=None)
    band_color = alt.Color("label:N", scale=color_scale, sort=label_order, legend=None)

    base = alt.Chart(data).encode(
        x=x_encoding,
    )

    band = base.mark_area(opacity=0.10).encode(
        y=alt.Y(f"{metric_prefix}_lower:Q", title=y_title),
        y2=f"{metric_prefix}_upper:Q",
        color=band_color,
    )

    line = base.mark_line(strokeWidth=2.5).encode(
        y=alt.Y(f"{metric_prefix}_mean:Q", title=y_title),
        color=line_color,
    )

    error_points = data.loc[data["snapshot"] % 10 == 0].copy()
    rules = alt.Chart(error_points).encode(
        x=x_encoding,
        color=band_color,
    ).mark_rule(opacity=0.30).encode(
        y=f"{metric_prefix}_lower:Q",
        y2=f"{metric_prefix}_upper:Q",
    )

    return (
        band + rules + line
    ).properties(
        width=980,
        height=320,
        title=title,
    )


def build_manual_legend(color_scale: alt.Scale, labels: list[str]) -> alt.Chart:
    legend_df = pd.DataFrame(
        {
            "label": labels,
            "x_start": [0.0] * len(labels),
            "x_end": [24.0] * len(labels),
            "text_x": [30.0] * len(labels),
        }
    )

    y_encoding = alt.Y(
        "label:N",
        sort=labels,
        axis=alt.Axis(title=None, domain=False, ticks=False, labels=False),
    )

    swatches = (
        alt.Chart(legend_df)
        .mark_rule(strokeWidth=4)
        .encode(
            x=alt.X(
                "x_start:Q",
                axis=None,
                scale=alt.Scale(domain=[0, 120], nice=False),
            ),
            x2="x_end:Q",
            y=y_encoding,
            color=alt.Color("label:N", scale=color_scale, legend=None),
        )
    )

    labels_chart = (
        alt.Chart(legend_df)
        .mark_text(align="left", baseline="middle", fontSize=12)
        .encode(
            x=alt.X(
                "text_x:Q",
                axis=None,
                scale=alt.Scale(domain=[0, 120], nice=False),
            ),
            y=alt.Y("label:N", sort=labels, axis=None),
            text="label:N",
            color=alt.value("#222222"),
        )
    )

    return (swatches + labels_chart).properties(
        width=220,
        height=26 * len(labels),
        title="Experiment",
    )


def build_chart(aggregate_df: pd.DataFrame) -> alt.Chart:
    label_order = [EXPERIMENT_LABELS[key] for key in EXPERIMENT_ORDER if key in aggregate_df["experiment"].unique()]
    color_scale = alt.Scale(
        domain=label_order,
        range=[EXPERIMENT_COLORS[key] for key in EXPERIMENT_ORDER if key in aggregate_df["experiment"].unique()],
    )

    reward_chart = make_metric_chart(
        aggregate_df,
        metric_prefix="reward",
        y_title="Mean Return (mean ± 1 sd across seeds)",
        title="Reward Evolution",
        color_scale=color_scale,
    )

    rate_chart = make_metric_chart(
        aggregate_df,
        metric_prefix="rate",
        y_title="Throughput (items/s, mean ± 1 sd across seeds)",
        title="Throughput Evolution",
        color_scale=color_scale,
    )

    legend_chart = build_manual_legend(color_scale=color_scale, labels=label_order)

    plots = alt.vconcat(reward_chart, rate_chart).properties(
        title="Multi-Seed Reward and Throughput Evolution"
    )
    return alt.hconcat(plots, legend_chart, spacing=20)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot reward/throughput evolution from grouped multi-seed retests.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/tmp/milabench/artifacts/benchmarks/multiseed_retest_seeds012"),
        help="Root directory containing <experiment>/seed<seed>/stdout.txt traces.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/tmp/milabench/artifacts/benchmarks/multiseed_retest_seeds012/evolution_comparison"),
        help="Output prefix for .png/.html/.csv files.",
    )
    args = parser.parse_args()

    df = collect_traces(args.root)
    aggregate_df = aggregate_traces(df)
    chart = build_chart(aggregate_df)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_prefix.with_suffix(".raw.csv"), index=False)
    aggregate_df.to_csv(args.output_prefix.with_suffix(".agg.csv"), index=False)
    chart.save(args.output_prefix.with_suffix(".html"))
    chart.save(args.output_prefix.with_suffix(".png"), scale_factor=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
