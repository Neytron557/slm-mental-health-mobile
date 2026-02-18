#!/usr/bin/env python3
"""
Convert release checkpoints to ONNX Runtime Mobile artifacts.

Primary path: ORT GenAI conversion.
Fallback path: Optimum ONNX export + ORT dynamic quantization.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "android_deployment" / "config" / "models.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to models.yml registry.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        help="Override artifacts root (default: from config).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ort-genai", "onnx"],
        default="auto",
        help="Conversion backend to use. 'auto' reads per-model preference.",
    )
    parser.add_argument(
        "--fallback-backend",
        choices=["ort-genai", "onnx", "none"],
        default="onnx",
        help="Backend to attempt if the primary conversion fails.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model names to process.",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="Precision flag passed to the converters (overrides config).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override context length passed to converters.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task passed to the converter; defaults to config value.",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python interpreter used to invoke ORT/Optimum modules.",
    )
    parser.add_argument(
        "--ort-genai-extra-args",
        default="",
        help="Extra CLI flags appended to the ORT GenAI command.",
    )
    parser.add_argument(
        "--optimum-extra-args",
        default="",
        help="Extra CLI flags appended to the Optimum export command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artifacts/manifests.",
    )
    return parser.parse_args()


def load_registry(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def run_command(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> None:
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    heading = f"[cmd] {cmd_str}"
    if cwd:
        heading += f" (cwd={cwd})"
    print(heading)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def resolve_backend(entry: Dict, args_backend: str, defaults_backend: str) -> str:
    if args_backend != "auto":
        return args_backend
    return entry.get("preferred_backend", defaults_backend)


def resolve_precision(entry: Dict, cli_precision: Optional[str], defaults_precision: str) -> str:
    if cli_precision:
        return cli_precision
    return entry.get("preferred_precision") or entry.get("precision") or defaults_precision


def resolve_max_seq_len(entry: Dict, cli_max_seq_len: Optional[int], defaults_max_seq_len: int) -> int:
    if cli_max_seq_len:
        return cli_max_seq_len
    return entry.get("max_seq_len", defaults_max_seq_len)


def resolve_task(entry: Dict, cli_task: Optional[str], defaults_task: str) -> str:
    if cli_task:
        return cli_task
    return entry.get("task", defaults_task)


def copy_tokenizer_assets(source: Path, destination: Path, dry_run: bool) -> List[str]:
    wanted = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    copied = []
    for filename in wanted:
        src_file = source / filename
        if src_file.exists():
            dest_file = destination / filename
            if dry_run:
                print(f"[copy] {src_file} -> {dest_file}")
            else:
                ensure_dir(dest_file.parent)
                dest_file.write_bytes(src_file.read_bytes())
            copied.append(filename)
    return copied


def find_decoder_graph(onnx_dir: Path) -> Path:
    candidates = [
        "decoder_with_past_model.onnx",
        "decoder_with_past.onnx",
        "decoder_model.onnx",
        "model.onnx",
    ]
    for cand in candidates:
        path = onnx_dir / cand
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate decoder graph in {onnx_dir}")


def ort_genai_command(
    python_exec: str,
    source: Path,
    output: Path,
    task: str,
    precision: str,
    max_seq_len: int,
    extra_args: str,
) -> List[str]:
    cmd = [
        python_exec,
        "-m",
        "onnxruntime_genai.models.convert",
        "--model",
        str(source),
        "--output",
        str(output),
        "--task",
        task,
        "--precision",
        precision,
        "--max_seq_len",
        str(max_seq_len),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def optimum_command(
    source: Path,
    output: Path,
    task: str,
    extra_args: str,
) -> List[str]:
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--task",
        task,
        "--model",
        str(source),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    cmd.append(str(output))
    return cmd


def onnx_checker_command(python_exec: str, graph: Path) -> List[str]:
    return [
        python_exec,
        "-m",
        "onnxruntime.tools.onnxruntime_test.checkers",
        str(graph),
    ]


def ort_dynamic_quant_command(
    python_exec: str,
    input_graph: Path,
    output_graph: Path,
    weight_type: str = "QInt8",
) -> List[str]:
    return [
        python_exec,
        "-m",
        "onnxruntime.quantization.quantize_dynamic",
        "--weight_type",
        weight_type,
        "--input",
        str(input_graph),
        "--output",
        str(output_graph),
    ]


def write_manifest(
    manifest_path: Path,
    *,
    entry: Dict,
    backend: str,
    precision: str,
    max_seq_len: int,
    commands: List[Sequence[str]],
    tokenizer_files: List[str],
    dry_run: bool,
) -> None:
    manifest = {
        "model_name": entry["name"],
        "source": entry["source"],
        "backend": backend,
        "precision": precision,
        "max_seq_len": max_seq_len,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commands": [" ".join(shlex.quote(part) for part in cmd) for cmd in commands],
        "tokenizer_files": tokenizer_files,
        "notes": entry.get("notes", ""),
    }
    if dry_run:
        print(f"[manifest] {manifest_path} (dry-run)")
        return
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def convert_entry(
    entry: Dict,
    *,
    models_root: Path,
    artifacts_root: Path,
    defaults: Dict,
    args: argparse.Namespace,
) -> None:
    source_dir = (models_root / entry["source"]).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    backend = resolve_backend(entry, args.backend, defaults.get("backend", "ort-genai"))
    precision = resolve_precision(entry, args.precision, defaults.get("precision", "int4"))
    max_seq_len = resolve_max_seq_len(entry, args.max_seq_len, defaults.get("max_seq_len", 4096))
    task = resolve_task(entry, args.task, defaults.get("task", "text-generation"))

    output_dir = artifacts_root / entry["name"] / backend
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[skip] {entry['name']} already converted ({manifest_path})")
        return

    ensure_dir(output_dir)
    tokenizer_files = copy_tokenizer_assets(source_dir, output_dir, args.dry_run)
    commands_run: List[Sequence[str]] = []

    def run_with_handling(cmd: Sequence[str]) -> None:
        run_command(cmd, dry_run=args.dry_run)
        commands_run.append(cmd)

    try:
        if backend == "ort-genai":
            cmd = ort_genai_command(
                args.python_exec,
                source=source_dir,
                output=output_dir,
                task=task,
                precision=precision,
                max_seq_len=max_seq_len,
                extra_args=args.ort_genai_extra_args,
            )
            run_with_handling(cmd)
        elif backend == "onnx":
            cmd = optimum_command(
                source=source_dir,
                output=output_dir,
                task="text-generation-with-past",
                extra_args=args.optimum_extra_args,
            )
            run_with_handling(cmd)

            decoder_graph = find_decoder_graph(output_dir)
            run_with_handling(onnx_checker_command(args.python_exec, decoder_graph))

            quantized_out = decoder_graph.with_suffix(".int8.onnx")
            run_with_handling(
                ort_dynamic_quant_command(args.python_exec, decoder_graph, quantized_out)
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    except subprocess.CalledProcessError as exc:
        print(f"[error] Conversion failed for {entry['name']} with backend {backend}: {exc}")
        fallback = args.fallback_backend
        if fallback == "none" or fallback == backend:
            raise
        print(f"[retry] Attempting fallback backend '{fallback}'")
        entry_with_override = dict(entry)
        entry_with_override["preferred_backend"] = fallback
        convert_entry(
            entry_with_override,
            models_root=models_root,
            artifacts_root=artifacts_root,
            defaults=defaults,
            args=args,
        )
        return

    write_manifest(
        manifest_path,
        entry=entry,
        backend=backend,
        precision=precision,
        max_seq_len=max_seq_len,
        commands=commands_run,
        tokenizer_files=tokenizer_files,
        dry_run=args.dry_run,
    )
    print(f"[done] {entry['name']} -> {output_dir}")


def main() -> None:
    args = parse_args()
    registry = load_registry(args.config)

    models_root = (REPO_ROOT / registry["models_root"]).resolve()
    artifacts_root = (
        args.artifacts_root or (REPO_ROOT / registry.get("artifacts_root", "android_deployment/artifacts"))
    ).resolve()
    ensure_dir(artifacts_root)

    defaults = registry.get("defaults", {})
    entries = registry.get("models", [])
    selected = {name for name in args.models} if args.models else None

    for entry in entries:
        if selected and entry["name"] not in selected:
            continue
        convert_entry(
            entry,
            models_root=models_root,
            artifacts_root=artifacts_root,
            defaults=defaults,
            args=args,
        )


if __name__ == "__main__":
    main()
