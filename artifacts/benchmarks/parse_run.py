#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path


def summarize(run_dir: Path) -> dict:
    data_file = run_dir / "dqn.D0.data"
    rates = []
    losses = []
    returns = []
    memory_peaks = []
    gpu_mems = []
    harness_progress = None
    step_timer_progress = None

    with data_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("event") != "data":
                continue

            payload = record.get("data", {})

            if "rate" in payload:
                rates.append(payload["rate"])
            if "loss" in payload:
                losses.append(payload["loss"])
            if "returns" in payload:
                returns.append(payload["returns"])
            if "memory_peak" in payload:
                memory_peaks.append(payload["memory_peak"])
            if payload.get("task") == "main" and "gpudata" in payload:
                for gpu in payload["gpudata"].values():
                    memory = gpu.get("memory")
                    if memory:
                        gpu_mems.append(memory[0])
            if payload.get("task") == "early_stop" and "progress" in payload:
                progress = payload["progress"]
                if len(progress) == 2 and progress[1] == 20:
                    harness_progress = progress
                else:
                    step_timer_progress = progress

    if not rates:
        raise RuntimeError(f"No rate samples found in {data_file}")

    return {
        "run_dir": str(run_dir),
        "rate_count": len(rates),
        "rate_mean": statistics.mean(rates),
        "rate_median": statistics.median(rates),
        "rate_min": min(rates),
        "rate_max": max(rates),
        "loss_last": losses[-1] if losses else None,
        "returns_last": returns[-1] if returns else None,
        "returns_max": max(returns) if returns else None,
        "memory_peak_mib": max(memory_peaks) if memory_peaks else None,
        "gpu_memory_used_mib": max(gpu_mems) if gpu_mems else None,
        "harness_progress": harness_progress,
        "step_timer_progress": step_timer_progress,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_run.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1])
    print(json.dumps(summarize(run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
