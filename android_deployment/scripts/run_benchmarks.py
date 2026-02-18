#!/usr/bin/env python3
"""
Run the on-device CLI benchmark across models and execution providers.

This script assumes:
- slm_bench binary + libs exist under --device-workdir
- model artifacts live under --device-model-root/<model_name>
- Each run emits JSON to <model_dir>/bench_<ep>.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "android_deployment" / "config" / "models.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="models.yml registry")
    parser.add_argument("--adb", default="adb", help="adb binary")
    parser.add_argument(
        "--device-workdir",
        default="/data/local/tmp/slm_bench",
        help="Directory containing slm_bench binary on device",
    )
    parser.add_argument(
        "--device-binary",
        default="slm_bench",
        help="Binary name relative to device workdir",
    )
    parser.add_argument(
        "--device-lib-dir",
        default="/data/local/tmp/slm_bench",
        help="Path added to LD_LIBRARY_PATH before running the CLI",
    )
    parser.add_argument(
        "--device-model-root",
        default="/sdcard/Models",
        help="Base directory storing model artifacts on device",
    )
    parser.add_argument(
        "--device-json-template",
        default="bench_{ep}.json",
        help="Filename produced per run (within each model dir).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "android_deployment" / "results",
        help="Host directory where pulled JSON metrics are stored.",
    )
    parser.add_argument("--models", nargs="*", help="Subset of model names to benchmark.")
    parser.add_argument("--eps", nargs="+", default=["cpu", "vulkan", "nnapi"], help="Execution providers.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of runs per EP.")
    parser.add_argument("--prompt-file", default="/sdcard/Models/prompts.txt", help="Prompt file on device.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampler temperature.")
    parser.add_argument("--extra-args", default="", help="Additional args forwarded to slm_bench.")
    parser.add_argument("--dry-run", action="store_true", help="Only print adb commands.")
    return parser.parse_args()


def load_registry(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_adb(cmd: Sequence[str], *, dry_run: bool) -> None:
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    print(f"[adb] {cmd_str}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def pull_file(adb: str, remote: str, local: Path, *, dry_run: bool) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    run_adb([adb, "pull", remote, str(local)], dry_run=dry_run)


def build_shell_command(
    device_workdir: str,
    device_binary: str,
    *,
    model_dir: str,
    ep: str,
    prompt_file: str,
    max_new_tokens: int,
    temperature: float,
    repeats: int,
    device_lib_dir: str,
    extra_args: str,
) -> str:
    ld_path = f"LD_LIBRARY_PATH={device_lib_dir} " if device_lib_dir else ""
    args = [
        f"--model_dir {shlex.quote(model_dir)}",
        f"--ep {ep}",
        f"--prompt_file {shlex.quote(prompt_file)}",
        f"--max_new_tokens {max_new_tokens}",
        f"--temperature {temperature}",
        f"--repeat {repeats}",
    ]
    if extra_args:
        args.append(extra_args)
    joined_args = " ".join(args)
    return f"cd {device_workdir} && {ld_path}./{device_binary} {joined_args}"


def main() -> None:
    args = parse_args()
    registry = load_registry(args.config)
    entries = registry.get("models", [])
    selected = set(args.models) if args.models else None
    args.results_root.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        if selected and entry["name"] not in selected:
            continue
        model_name = entry["name"]
        model_dir = f"{args.device_model_root}/{model_name}"
        model_results_dir = args.results_root / model_name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        for ep in args.eps:
            for rep_idx in range(args.repeats):
                shell_cmd = build_shell_command(
                    args.device_workdir,
                    args.device_binary,
                    model_dir=model_dir,
                    ep=ep,
                    prompt_file=args.prompt_file,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repeats=1,
                    device_lib_dir=args.device_lib_dir,
                    extra_args=args.extra_args,
                )
                adb_cmd = [args.adb, "shell", shell_cmd]
                print(f"[run] model={model_name} ep={ep} repeat={rep_idx + 1}")
                run_adb(adb_cmd, dry_run=args.dry_run)

                device_json = f"{model_dir}/{args.device_json_template.format(ep=ep)}"
                host_json = model_results_dir / f"bench_{ep}_r{rep_idx + 1}.json"
                pull_file(args.adb, device_json, host_json, dry_run=args.dry_run)

        metadata = {
            "model": model_name,
            "eps": args.eps,
            "repeats": args.repeats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_file": args.prompt_file,
            "max_new_tokens": args.max_new_tokens,
        }
        metadata_path = model_results_dir / "run_metadata.json"
        if not args.dry_run:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
