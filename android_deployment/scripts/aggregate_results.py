#!/usr/bin/env python3
"""
Aggregate benchmark JSON outputs into CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="android_deployment/results/**/*.json",
        help="Glob for JSON metric files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("android_deployment/results/benchmarks.csv"),
        help="CSV file to write.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional Markdown summary output.",
    )
    return parser.parse_args()


def iter_json(glob_pattern: str) -> Iterable[Path]:
    return Path(".").glob(glob_pattern)


def flatten_record(data: Dict[str, Any], source: Path) -> Dict[str, Any]:
    record = {
        "model": data.get("model"),
        "ep": data.get("ep"),
        "vocab_size": data.get("vocab_size"),
        "max_seq_len": data.get("max_seq_len"),
        "prompt_tokens": data.get("prompt_tokens"),
        "generated_tokens": data.get("generated_tokens"),
        "load_ms": data.get("load_ms"),
        "ttft_ms": data.get("ttft_ms"),
        "decode_ms": data.get("decode_ms"),
        "tokens_per_second": data.get("tokens_per_second") or data.get("tps"),
        "peak_rss_mb": data.get("peak_rss_mb"),
        "timestamp": data.get("timestamp"),
        "source": str(source),
    }
    return record


def write_csv(records: List[Dict[str, Any]], output: Path) -> None:
    if not records:
        print("[warn] No benchmark records found.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[done] Wrote {len(records)} rows -> {output}")


def write_markdown(records: List[Dict[str, Any]], output: Path) -> None:
    if not records:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Model | EP | TTFT (ms) | Decode (ms) | Tokens/s | Peak RSS (MB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        lines.append(
            f"| {rec.get('model')} | {rec.get('ep')} | {rec.get('ttft_ms')} | "
            f"{rec.get('decode_ms')} | {rec.get('tokens_per_second')} | {rec.get('peak_rss_mb')} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] Markdown summary -> {output}")


def main() -> None:
    args = parse_args()
    records: List[Dict[str, Any]] = []
    for path in iter_json(args.input_glob):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] Skipping {path}: {exc}")
            continue
        if not isinstance(data, dict) or "ep" not in data:
            continue
        records.append(flatten_record(data, path))

    if not records:
        print("[warn] No usable benchmark JSON files found.")
        return

    write_csv(records, args.output)
    if args.markdown:
        write_markdown(records, args.markdown)


if __name__ == "__main__":
    main()
