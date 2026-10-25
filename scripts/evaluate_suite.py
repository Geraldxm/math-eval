#!/usr/bin/env python3
"""Evaluate compatible dataset configs sequentially with one lazy vLLM backend."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from evaluate import StopState, execute_run, load_config, prepare_run, safe_name, stop_signals
from inference import create_backend, vllm_backend_signature
from io_utils import atomic_json
from resume import RawShardWriter


def load_suite(path: Path) -> list[tuple[Path, str, Path]]:
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or set(suite) != {"schema_version", "cases"}:
        raise ValueError("suite requires exactly schema_version and cases")
    if type(suite["schema_version"]) is not int or suite["schema_version"] != 1:
        raise ValueError("suite.schema_version must be 1")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("suite.cases must be a non-empty list")
    cases = []
    run_ids = set()
    reference = None
    for index, case in enumerate(suite["cases"]):
        label = f"case {index + 1}"
        if not isinstance(case, dict) or set(case) != {"config", "run_id"}:
            raise ValueError(f"{label}: requires exactly config and run_id")
        run_id = case["run_id"]
        if (
            not isinstance(run_id, str) or not run_id.strip()
            or Path(run_id).name != run_id or run_id in {".", ".."}
        ):
            raise ValueError(f"{label}: run_id must be a non-empty directory name")
        if run_id in run_ids:
            raise ValueError(f"{label}: duplicate run_id {run_id!r}")
        run_ids.add(run_id)
        if not isinstance(case["config"], str) or not case["config"].strip():
            raise ValueError(f"{label} ({run_id}): config must be a path string")
        config_path = (path.parent / Path(case["config"]).expanduser()).resolve()
        config = load_config(config_path)
        if config["model"]["backend"] != "vllm":
            raise ValueError(f"{label} ({run_id}): suite supports vllm only")
        signature = vllm_backend_signature(config["model"])
        if signature["request_batch_size"] < 1:
            raise ValueError(f"{label} ({run_id}): request_batch_size must be >= 1")
        if reference is None:
            reference = signature
        else:
            mismatches = [
                f"engine.{key}"
                for key in reference["engine"].keys() | signature["engine"].keys()
                if reference["engine"].get(key) != signature["engine"].get(key)
            ]
            mismatches += [
                key for key in ("input_mode", "thinking", "request_batch_size")
                if reference[key] != signature[key]
            ]
            if mismatches:
                raise ValueError(
                    f"{label} ({run_id}): incompatible backend fields: {sorted(mismatches)}"
                )
        run_dir = (Path(config["output"]["root"]).expanduser() / run_id).resolve()
        for _, previous_id, previous_dir in cases:
            if (
                run_dir == previous_dir or run_dir in previous_dir.parents
                or previous_dir in run_dir.parents
            ):
                raise ValueError(f"{label} ({run_id}): output path conflicts with {previous_id}")
        cases.append((config_path, run_id, run_dir))
    return cases


def run_suite(suite_path: Path, resume: bool = False) -> int:
    stop = StopState()
    with stop_signals(stop, "vllm"):
        cases = load_suite(suite_path)
        prepared = []
        for config_path, run_id, run_dir in cases:
            if stop.requested:
                return 130
            manifest_exists = (run_dir / "manifests/run.json").exists()
            if not manifest_exists and run_dir.exists() and any(run_dir.rglob("*")):
                raise FileExistsError(f"{run_dir}: refusing to reuse output without a manifest")
            try:
                case = prepare_run(config_path, run_id, resume and manifest_exists)
                if case.manifest.get("run_id") != run_id:
                    raise ValueError("run_id does not match existing run")
                snapshot = yaml.safe_load((run_dir / "manifests/config.yaml").read_text())
                snapshot_dir = (Path(snapshot["output"]["root"]).expanduser() / run_id).resolve()
                if snapshot_dir != run_dir:
                    raise ValueError("output path does not match existing run")
                if not manifest_exists:
                    case.manifest["status"] = "pending"
                    atomic_json(run_dir / "manifests/run.json", case.manifest)
                # Probe actual output mounts, including the eventual raw writer directory.
                raw_dir = (
                    run_dir / "raw"
                    / safe_name(str(case.config["model"]["name"]))
                    / safe_name(str(case.config["dataset"]["name"]))
                )
                for directory in (run_dir / "manifests", raw_dir, run_dir / "parsed", run_dir / "metrics"):
                    directory.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryFile(dir=directory) as handle:
                        handle.write(b"output preflight")
                        handle.flush()
                RawShardWriter(
                    raw_dir,
                    shard_size=int(case.config["output"].get("shard_size", 100)),
                    compression=case.config["output"].get("compression", "none"),
                    fsync_every=case.fsync_every,
                ).abort()
            except Exception as exc:
                raise RuntimeError(f"case {run_id}: {exc}") from exc
            prepared.append(case)

        backend = None

        def shared_backend(model, preflight):
            nonlocal backend
            if backend is None:
                backend = create_backend(model, preflight)
            return backend

        for case in prepared:
            if stop.requested:
                return 130
            run_dir = execute_run(case, shared_backend, stop)
            status = json.loads((run_dir / "manifests/run.json").read_text())["status"]
            print(json.dumps({"run_id": case.run_id, "run_dir": str(run_dir), "status": status}), flush=True)
            if stop.requested or status == "stopped":
                return 130
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run_suite(args.suite, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
