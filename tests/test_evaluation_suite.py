import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_suite
from evaluate import run as evaluate_run
from evaluate_suite import load_suite, run_suite
from inference import InferenceResult, sampling_request
from io_utils import read_jsonl
from replay_evaluation import replay
from resume import scan_raw


class FakeBackend:
    request_batch_size = 1

    def __init__(self, fail_on=None, stop_on=None):
        self.fail_on = fail_on
        self.stop_on = stop_on
        self.calls = []

    def generate(self, prompt, decode, sample_idx):
        user = prompt[-1]["content"]
        self.calls.append((user, decode, sample_idx))
        if self.fail_on == (user, sample_idx):
            self.fail_on = None
            raise RuntimeError("fake failure")
        if self.stop_on == (user, sample_idx):
            signal.raise_signal(signal.SIGTERM)
        return InferenceResult(
            raw_text=r"\boxed{2}", reasoning_text=None, final_text=r"\boxed{2}",
            thinking_status="non_thinking_ok", finish_reason="stop", prompt_tokens=3,
            output_tokens=2, response_id=f"{user}-{sample_idx}",
            resolved_request={"messages": prompt, "sampling": sampling_request(decode, decode["seed"] + sample_idx)},
        )

    def generate_batch(self, requests):
        return [self.generate(**request) for request in requests]


class EvaluationSuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "runs"
        for name in ("a", "b"):
            prompt = self.root / f"prompt-{name}"
            prompt.mkdir()
            (prompt / "00-system.txt").write_text(f"{name} rules")
            (prompt / "01-user.txt").write_text(f"{name}: {{{{problem}}}}")
            data = self.root / f"{name}.jsonl"
            data.write_text(json.dumps({"id": "p", "problem": "1+1", "answer": "2"}) + "\n")
            config = {
                "dataset": {"name": name, "path": str(data)},
                "model": {"name": "fake", "backend": "vllm", "path": "fake/model", "input_mode": "chat", "thinking": False},
                "prompt": str(prompt),
                "decode": {"temperature": 0.5 if name == "a" else 0.8, "top_p": 0.9, "seed": 3 if name == "a" else 17, "max_output_tokens": 12 if name == "a" else 19, "samples_per_problem": 4, "stop": [f"{name}-end"]},
                "output": {"root": str(self.output), "shard_size": 3, "compression": "none"},
            }
            (self.root / f"{name}.yaml").write_text(yaml.safe_dump(config))
        self.suite = self.root / "suite.yaml"
        self._suite()

    def tearDown(self):
        self.tmp.cleanup()

    def _suite(self, cases=None):
        if cases is None:
            cases = [{"config": "a.yaml", "run_id": "run-a"}, {"config": "b.yaml", "run_id": "run-b"}]
        self.suite.write_text(yaml.safe_dump({"schema_version": 1, "cases": cases}))

    def _runtime_patches(self, backend):
        return (
            patch("evaluate.preflight_model", return_value={"applicable": True}),
            patch("evaluate.environment_snapshot", return_value={"packages": {}}),
            patch("evaluate.code_revision", return_value={"commit": "test", "dirty": True}),
            patch("evaluate_suite.create_backend", return_value=backend),
        )

    def test_shared_backend_case_isolation_and_replay(self):
        backend = FakeBackend()
        *runtime, factory_patch = self._runtime_patches(backend)
        config_bytes = [(self.root / f"{name}.yaml").read_bytes() for name in ("a", "b")]
        with runtime[0], runtime[1], runtime[2], factory_patch as factory, patch(
            "evaluate_suite.execute_run", wraps=evaluate_suite.execute_run
        ) as execute:
            self.assertEqual(run_suite(self.suite), 0)
            self.assertEqual(factory.call_count, 1)
            self.assertIs(execute.call_args_list[0].args[1], execute.call_args_list[1].args[1])
        self.assertEqual(config_bytes, [(self.root / f"{name}.yaml").read_bytes() for name in ("a", "b")])
        for name, run_id, seed in (("a", "run-a", 3), ("b", "run-b", 17)):
            calls = [row for row in backend.calls if row[0].startswith(f"{name}:")]
            self.assertEqual({row[2] for row in calls}, set(range(4)))
            config = yaml.safe_load((self.root / f"{name}.yaml").read_text())
            self.assertTrue(all(row[1] == config["decode"] for row in calls))
            run_dir = self.output / run_id
            rows = scan_raw(run_dir / "raw")
            self.assertEqual(len(rows), 4)
            self.assertEqual({key[1] for key in rows}, {name})
            for key, indexed in rows.items():
                self.assertEqual(indexed.row["resolved_request"]["sampling"], sampling_request(calls[0][1], seed + key[3]))
                self.assertTrue(indexed.row["resolved_request"]["messages"][-1]["content"].startswith(f"{name}:"))
                self.assertEqual(indexed.row["resolved_request"]["messages"][0]["content"], f"{name} rules")
            parsed, metrics = replay(run_dir, [1, 2, 4])
            self.assertEqual(len(list(read_jsonl(parsed))), 4)
            self.assertEqual(json.loads(metrics.read_text())["strict"]["k"]["4"]["pass_at_k"], 1.0)
            self.assertEqual(json.loads((run_dir / "manifests/run.json").read_text())["status"], "completed")
            single_config = dict(config)
            single_config["output"] = dict(config["output"], root=str(self.root / f"single-{name}"))
            single_path = self.root / f"single-{name}.yaml"
            single_path.write_text(yaml.safe_dump(single_config))
            single_backend = FakeBackend()
            with (
                patch("evaluate.preflight_model", return_value={"applicable": True}),
                patch("evaluate.environment_snapshot", return_value={"packages": {}}),
                patch("evaluate.code_revision", return_value={"commit": "test", "dirty": True}),
                patch("evaluate.create_backend", return_value=single_backend),
            ):
                single_dir = evaluate_run(single_path, "single")
            self.assertEqual(set(scan_raw(run_dir / "raw")), set(scan_raw(single_dir / "raw")))

    def test_partial_failure_resume_and_completed_resume(self):
        backend = FakeBackend(fail_on=("b: 1+1", 1))
        *runtime, factory_patch = self._runtime_patches(backend)
        with runtime[0], runtime[1], runtime[2], factory_patch as factory:
            with self.assertRaisesRegex(RuntimeError, "fake failure"):
                run_suite(self.suite)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(len(scan_raw(self.output / "run-a/raw")), 4)
            self.assertEqual(len(scan_raw(self.output / "run-b/raw")), 1)
            resumed = FakeBackend()
            with patch("evaluate_suite.create_backend", return_value=resumed) as create:
                self.assertEqual(run_suite(self.suite, resume=True), 0)
                self.assertEqual(create.call_count, 1)
            self.assertEqual(len(resumed.calls), 3)
            self.assertEqual({call[2] for call in resumed.calls if call[0] == "b: 1+1"}, {1, 2, 3})
            self.assertEqual(len(scan_raw(self.output / "run-a/raw")), 4)
            self.assertEqual(len(scan_raw(self.output / "run-b/raw")), 4)
            metrics = replay(self.output / "run-a", [1, 2])[1]
            before = metrics.read_bytes()
            with patch("evaluate_suite.create_backend") as create:
                self.assertEqual(run_suite(self.suite, resume=True), 0)
                create.assert_not_called()
            self.assertEqual(metrics.read_bytes(), before)

    def test_stop_seals_current_run_and_restores_signals(self):
        backend = FakeBackend(stop_on=("a: 1+1", 0))
        old = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        *runtime, factory_patch = self._runtime_patches(backend)
        with runtime[0], runtime[1], runtime[2], factory_patch as factory:
            self.assertEqual(run_suite(self.suite), 130)
            self.assertEqual(factory.call_count, 1)
        self.assertEqual((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)), old)
        self.assertEqual(json.loads((self.output / "run-a/manifests/run.json").read_text())["status"], "stopped")
        self.assertFalse(list((self.output / "run-a/raw").glob("**/*.inprogress")))
        self.assertFalse(list((self.output / "run-b/raw").glob("**/*.jsonl*")))
        self.assertEqual(json.loads((self.output / "run-b/manifests/run.json").read_text())["status"], "pending")

    def test_stop_resume_and_stop_between_cases(self):
        backend = FakeBackend(stop_on=("a: 1+1", 0))
        *runtime, factory_patch = self._runtime_patches(backend)
        with runtime[0], runtime[1], runtime[2], factory_patch:
            self.assertEqual(run_suite(self.suite), 130)
            resumed = FakeBackend()
            with patch("evaluate_suite.create_backend", return_value=resumed) as create:
                self.assertEqual(run_suite(self.suite, resume=True), 0)
                self.assertEqual(create.call_count, 1)
        self.assertEqual({call[0] for call in resumed.calls}, {"a: 1+1", "b: 1+1"})
        self.assertEqual({call[2] for call in resumed.calls if call[0] == "a: 1+1"}, {1, 2, 3})
        self.assertEqual({call[2] for call in resumed.calls if call[0] == "b: 1+1"}, {0, 1, 2, 3})

        self.tearDown()
        self.setUp()
        boundary_backend = FakeBackend()
        *runtime, factory_patch = self._runtime_patches(boundary_backend)
        original = evaluate_suite.execute_run
        calls = 0

        def stop_after_first_case(*args):
            nonlocal calls
            result = original(*args)
            calls += 1
            if calls == 1:
                signal.raise_signal(signal.SIGTERM)
            return result

        with runtime[0], runtime[1], runtime[2], factory_patch, patch(
            "evaluate_suite.execute_run", side_effect=stop_after_first_case
        ):
            self.assertEqual(run_suite(self.suite), 130)
        self.assertEqual(calls, 1)
        self.assertEqual(len(boundary_backend.calls), 4)
        self.assertEqual(json.loads((self.output / "run-a/manifests/run.json").read_text())["status"], "completed")
        self.assertEqual(json.loads((self.output / "run-b/manifests/run.json").read_text())["status"], "pending")

    def test_preflight_rejects_incompatibility_duplicate_ids_and_bad_schema(self):
        b_path = self.root / "b.yaml"
        config = yaml.safe_load(b_path.read_text())
        for field, value, expected in (
            ("path", "fake/other-model", "engine.model"),
            ("tokenizer_path", "fake/other-tokenizer", "engine.tokenizer"),
            ("max_num_seqs", 99, "max_num_seqs"),
            ("tensor_parallel_size", 2, "tensor_parallel_size"),
            ("dtype", "float16", "dtype"),
            ("gpu_memory_utilization", 0.7, "gpu_memory_utilization"),
            ("max_context_tokens", 2048, "max_model_len"),
            ("enforce_eager", True, "enforce_eager"),
            ("disable_custom_all_reduce", True, "disable_custom_all_reduce"),
            ("language_model_only", True, "language_model_only"),
            ("gdn_prefill_backend", "triton", "gdn_prefill_backend"),
            ("input_mode", "completion", "input_mode"),
            ("thinking", True, "thinking"),
            ("request_batch_size", 2, "request_batch_size"),
        ):
            changed = json.loads(json.dumps(config))
            if field in {"path", "tokenizer_path"}:
                changed["model"][field] = value
            elif field in {"input_mode", "thinking"}:
                changed["model"][field] = value
                if field == "input_mode":
                    changed["model"].pop("thinking", None)
                    completion = self.root / "completion.txt"
                    completion.write_text("{{problem}}")
                    changed["prompt"] = str(completion)
            else:
                changed["model"]["runtime"] = {field: value}
            b_path.write_text(yaml.safe_dump(changed))
            with self.subTest(field=field), patch(
                "evaluate.preflight_model", return_value={"applicable": True}
            ), patch("evaluate_suite.create_backend") as create:
                with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                    run_suite(self.suite)
                create.assert_not_called()
        b_path.write_text(yaml.safe_dump(config))
        self._suite([{"config": "a.yaml", "run_id": "same"}, {"config": "b.yaml", "run_id": "same"}])
        with self.assertRaisesRegex(ValueError, "duplicate run_id"):
            load_suite(self.suite)
        alias = self.root / "runs-alias"
        alias.symlink_to(self.output, target_is_directory=True)
        config["output"]["root"] = str(alias / "path-a")
        b_path.write_text(yaml.safe_dump(config))
        self._suite([{"config": "a.yaml", "run_id": "path-a"}, {"config": "b.yaml", "run_id": "child"}])
        with self.assertRaisesRegex(ValueError, "output path conflicts"):
            load_suite(self.suite)
        self.suite.write_text(yaml.safe_dump({"schema_version": 2, "cases": []}))
        with self.assertRaisesRegex(ValueError, "schema_version"):
            load_suite(self.suite)

    def test_bad_data_prompt_and_output_fail_before_backend_and_changed_resume_is_rejected(self):
        bad_data = yaml.safe_load((self.root / "b.yaml").read_text())
        Path(bad_data["dataset"]["path"]).write_text('{"id":"p","problem":"","answer":"2"}\n')
        (self.root / "b.yaml").write_text(yaml.safe_dump(bad_data))
        *runtime, factory_patch = self._runtime_patches(FakeBackend())
        with runtime[0], runtime[1], runtime[2], factory_patch as factory:
            with self.assertRaisesRegex(RuntimeError, "problem and answer"):
                run_suite(self.suite)
            factory.assert_not_called()

        # Invalid prompt is rejected by the all-case config pass, before outputs or engine creation.
        bad_data["dataset"]["path"] = str(self.root / "b.jsonl")
        bad_data["prompt"] = str(self.root / "missing-prompt")
        (self.root / "b.yaml").write_text(yaml.safe_dump(bad_data))
        with patch("evaluate_suite.create_backend") as factory:
            with self.assertRaisesRegex(ValueError, "prompt"):
                load_suite(self.suite)
            factory.assert_not_called()

        self.tearDown()
        self.setUp()
        denied = yaml.safe_load((self.root / "b.yaml").read_text())
        denied["output"]["root"] = "/proc/math-eval-suite-test"
        (self.root / "b.yaml").write_text(yaml.safe_dump(denied))
        *runtime, factory_patch = self._runtime_patches(FakeBackend())
        with runtime[0], runtime[1], runtime[2], factory_patch as factory:
            with self.assertRaisesRegex(RuntimeError, "case run-b"):
                run_suite(self.suite)
            factory.assert_not_called()

        self.tearDown()
        self.setUp()
        backend = FakeBackend()
        *runtime, factory_patch = self._runtime_patches(backend)
        with runtime[0], runtime[1], runtime[2], factory_patch:
            self.assertEqual(run_suite(self.suite), 0)
            Path(yaml.safe_load((self.root / "b.yaml").read_text())["dataset"]["path"]).write_text(
                json.dumps({"id": "p", "problem": "1+1", "answer": "3"}) + "\n"
            )
            with patch("evaluate_suite.create_backend") as factory:
                with self.assertRaisesRegex(RuntimeError, "generation spec"):
                    run_suite(self.suite, resume=True)
                factory.assert_not_called()

    def test_invalid_raw_writer_state_fails_before_backend(self):
        *runtime, factory_patch = self._runtime_patches(FakeBackend())
        with runtime[0], runtime[1], runtime[2], factory_patch:
            self.assertEqual(run_suite(self.suite), 0)
        for path in (self.output / "run-b/raw").glob("**/*.jsonl"):
            path.rename(path.with_suffix(".jsonl.inprogress"))
        *runtime, factory_patch = self._runtime_patches(FakeBackend())
        with runtime[0], runtime[1], runtime[2], factory_patch as create:
            with self.assertRaisesRegex(RuntimeError, "multiple active shards"):
                run_suite(self.suite, resume=True)
            create.assert_not_called()

    def test_defaults_normalize_and_sample_counts_stay_per_case(self):
        config_path = self.root / "b.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["model"]["tokenizer_path"] = config["model"]["path"]
        config["model"]["runtime"] = {
            "tensor_parallel_size": "1", "gpu_memory_utilization": "0.9",
            "dtype": "auto", "request_batch_size": 1,
            "enforce_eager": False, "disable_custom_all_reduce": False,
        }
        config["decode"]["samples_per_problem"] = 2
        config_path.write_text(yaml.safe_dump(config))
        backend = FakeBackend()
        *runtime, factory_patch = self._runtime_patches(backend)
        with runtime[0], runtime[1], runtime[2], factory_patch as create:
            self.assertEqual(run_suite(self.suite), 0)
            self.assertEqual(create.call_count, 1)
        self.assertEqual(len(scan_raw(self.output / "run-a/raw")), 4)
        self.assertEqual(len(scan_raw(self.output / "run-b/raw")), 2)

    def test_resume_rejects_changed_output_identity_before_backend(self):
        backend = FakeBackend()
        *runtime, factory_patch = self._runtime_patches(backend)
        with runtime[0], runtime[1], runtime[2], factory_patch:
            self.assertEqual(run_suite(self.suite), 0)
        (self.output / "renamed").symlink_to(self.output / "run-a", target_is_directory=True)
        self._suite([{"config": "a.yaml", "run_id": "renamed"}])
        *runtime, factory_patch = self._runtime_patches(FakeBackend())
        with runtime[0], runtime[1], runtime[2], factory_patch as create:
            with self.assertRaisesRegex(RuntimeError, "run_id does not match"):
                run_suite(self.suite, resume=True)
            create.assert_not_called()
        moved_root = self.root / "moved"
        moved_root.mkdir()
        (self.output / "run-a").rename(moved_root / "run-a")
        config_path = self.root / "a.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["output"]["root"] = str(moved_root)
        config_path.write_text(yaml.safe_dump(config))
        self._suite([{"config": "a.yaml", "run_id": "run-a"}])
        *runtime, factory_patch = self._runtime_patches(FakeBackend())
        with runtime[0], runtime[1], runtime[2], factory_patch as create:
            with self.assertRaisesRegex(RuntimeError, "output path does not match"):
                run_suite(self.suite, resume=True)
            create.assert_not_called()

    def test_suite_config_reference_and_legacy_config_paths_keep_their_base(self):
        old_cwd = Path.cwd()
        try:
            work = self.root / "cwd"
            suite_dir = self.root / "suite-dir"
            config_dir = suite_dir / "configs"
            config_dir.mkdir(parents=True)
            work.mkdir()
            prompt = work / "prompt"
            prompt.mkdir()
            (prompt / "00-user.txt").write_text("cwd: {{problem}}")
            dataset = work / "data.jsonl"
            dataset.write_text(json.dumps({"id": "p", "problem": "1+1", "answer": "2"}) + "\n")
            config = yaml.safe_load((self.root / "a.yaml").read_text())
            config["dataset"]["path"] = "data.jsonl"
            config["prompt"] = "prompt"
            cfg_path = config_dir / "case.yaml"
            original = yaml.safe_dump(config)
            cfg_path.write_text(original)
            suite = suite_dir / "suite.yaml"
            suite.write_text(yaml.safe_dump({"schema_version": 1, "cases": [{"config": "configs/case.yaml", "run_id": "cwd-case"}]}))
            backend = FakeBackend()
            os.chdir(work)
            *runtime, factory_patch = self._runtime_patches(backend)
            with runtime[0], runtime[1], runtime[2], factory_patch:
                self.assertEqual(run_suite(suite), 0)
            self.assertEqual(cfg_path.read_text(), original)
            self.assertEqual(backend.calls[0][0], "cwd: 1+1")
            self.assertEqual(load_suite(suite)[0][0], cfg_path)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
