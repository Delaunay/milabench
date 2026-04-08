#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from milabench.common import get_base_defaults
from milabench.config import build_config


ROOT = Path("/tmp/milabench")
CONFIG_PATH = ROOT / "benchmarks/purejaxrl/dev.yaml"
MAIN_PATH = ROOT / "benchmarks/purejaxrl/main.py"
PYTHON_BIN = Path("/tmp/results/venv/torch/bin/python")
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/benchmarks/batchsize_multiseed_seeds0to4"
DEFAULT_BASE = "/tmp/results"
AUTO_FALLBACK_RE = re.compile(r"auto\([^,]+,\s*([0-9]+)\s*\)")

EXPERIMENTS = {
    "control": "dqn-fp16",
    "b32768": "dqn-fp16-b32768",
    "b16384": "dqn-fp16-b16384",
    "b8192": "dqn-fp16-b8192",
}


def resolve_auto_value(value: object) -> str:
    if not isinstance(value, str):
        return str(value)

    match = AUTO_FALLBACK_RE.fullmatch(value.strip())
    if match:
        return match.group(1)
    return value


def load_config() -> dict:
    return build_config(get_base_defaults(DEFAULT_BASE), str(CONFIG_PATH))


def build_command(config: dict, seed: int) -> list[str]:
    argv = dict(config["argv"])

    if argv.pop("dqn", False):
        mode = "dqn"
    elif argv.pop("ppo", False):
        mode = "ppo"
    else:
        raise ValueError(f"Unsupported benchmark argv keys: {sorted(argv)}")

    argv["--seed"] = seed

    command = [str(PYTHON_BIN), str(MAIN_PATH), mode]
    for key, value in argv.items():
        command.extend([key, resolve_auto_value(value)])
    return command


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed batch-size reward sweeps from merged PureJaxRL config names.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    configs = load_config()
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    manifest = {
        "config_path": str(CONFIG_PATH),
        "python_bin": str(PYTHON_BIN),
        "main_path": str(MAIN_PATH),
        "experiments": EXPERIMENTS,
        "seeds": seeds,
        "env": {
            "PUREJAXRL_DQN_SCAN_UNROLL": "4",
        },
    }
    write_json(output_root / "manifest.json", manifest)

    env = os.environ.copy()
    env["PUREJAXRL_DQN_SCAN_UNROLL"] = "4"

    for experiment, config_name in EXPERIMENTS.items():
        config = configs[config_name]
        experiment_dir = output_root / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            run_dir = experiment_dir / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"

            if not args.force and stdout_path.exists() and stderr_path.exists():
                continue

            command = build_command(config, seed)
            (run_dir / "command.txt").write_text(" ".join(command) + "\n")
            write_json(
                run_dir / "metadata.json",
                {
                    "experiment": experiment,
                    "config_name": config_name,
                    "seed": seed,
                    "command": command,
                    "env": {
                        "PUREJAXRL_DQN_SCAN_UNROLL": env["PUREJAXRL_DQN_SCAN_UNROLL"],
                    },
                },
            )

            with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    check=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
