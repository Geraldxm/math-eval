#!/usr/bin/env python3
"""Replay parser and metrics from durable raw rollout shards."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from io_utils import atomic_json, read_jsonl, utc_now, write_jsonl
from metrics import metrics_file
from parser import (
    DUAL_PARSER_CONFIG_HASH,
    DUAL_PARSER_ID,
    PARSER_CONFIG_HASH,
    PARSER_ID,
    V5_DUAL_PARSER_CONFIG_HASH,
    V5_DUAL_PARSER_ID,
    parse_and_verify,
    parse_dual_and_verify,
    parse_v5_dual_and_verify,
)


def replay(
    run_dir: Path,
    k_values: list[int],
    parser_id: str = V5_DUAL_PARSER_ID,
) -> tuple[Path, Path]:
    if parser_id not in {PARSER_ID, DUAL_PARSER_ID, V5_DUAL_PARSER_ID}:
        raise ValueError(f"unsupported parser_id: {parser_id}")
    raw_paths = sorted(
        {
            path
            for pattern in ("raw/**/*.jsonl", "raw/**/*.jsonl.gz")
            for path in run_dir.glob(pattern)
            if not path.name.endswith(".inprogress")
        }
    )
    if not raw_paths:
        raise ValueError(f"{run_dir}: no sealed raw JSONL shards")
    parsed_path = run_dir / "parsed" / parser_id / "parsed.jsonl"
    parser_config_hash = {
        PARSER_ID: PARSER_CONFIG_HASH,
        DUAL_PARSER_ID: DUAL_PARSER_CONFIG_HASH,
        V5_DUAL_PARSER_ID: V5_DUAL_PARSER_CONFIG_HASH,
    }[parser_id]

    def parsed_rows():
        for raw_path in raw_paths:
            for row in read_jsonl(raw_path):
                required = {"final_text", "gold_answer"}
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"{raw_path}: missing fields {sorted(missing)}")
                raw_row = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "answer_contract",
                        "allow_last_number",
                        "parser_id",
                        "parser_config_hash",
                        "soft",
                    }
                }
                arguments = (
                    str(row["final_text"]),
                    str(row["gold_answer"]),
                    bool(row.get("truncated", False)),
                )
                if parser_id in {DUAL_PARSER_ID, V5_DUAL_PARSER_ID}:
                    parser = (
                        parse_v5_dual_and_verify
                        if parser_id == V5_DUAL_PARSER_ID
                        else parse_dual_and_verify
                    )
                    result = parser(*arguments)
                    yield {
                        **raw_row,
                        **asdict(result.strict),
                        "soft": asdict(result.soft),
                        "parser_id": parser_id,
                        "parser_config_hash": parser_config_hash,
                    }
                else:
                    result = parse_and_verify(*arguments)
                    yield {
                        **raw_row,
                        **asdict(result),
                        "parser_config_hash": parser_config_hash,
                    }

    count = write_jsonl(parsed_path, parsed_rows())
    metrics_path = run_dir / "metrics" / parser_id / "metrics.json"
    metrics_file([parsed_path], metrics_path, k_values)
    manifest_path = run_dir / "manifests/run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "evaluation_status": "completed",
            "evaluated_at": utc_now(),
            "parser_id": parser_id,
            "parser_config_hash": parser_config_hash,
            "parsed_rows": count,
            "parsed_path": str(parsed_path.relative_to(run_dir)),
            "metrics_path": str(metrics_path.relative_to(run_dir)),
        }
    )
    atomic_json(manifest_path, manifest)
    return parsed_path, metrics_path


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--run-dir", type=Path, required=True)
    argument_parser.add_argument("--k", type=int, nargs="+", default=[1])
    argument_parser.add_argument(
        "--parser-id",
        choices=(PARSER_ID, DUAL_PARSER_ID, V5_DUAL_PARSER_ID),
        default=V5_DUAL_PARSER_ID,
    )
    args = argument_parser.parse_args()
    parsed, metrics = replay(args.run_dir, args.k, args.parser_id)
    print(json.dumps({"parsed": str(parsed), "metrics": str(metrics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
