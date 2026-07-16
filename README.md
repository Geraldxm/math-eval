# math-eval

可续传、可重放的数学推理生成与评测工具。它读取 canonical JSONL，通过本地 vLLM 或 OpenAI-compatible Chat Completions 生成答案，再独立解析、统计和比较。

```text
canonical JSONL -> evaluate.py -> raw shards -> replay_evaluation.py -> parsed + metrics
                                                        |
                                                        +-> compare_runs.py
```

生成与判分刻意分离：更换 parser 或指标时不需要重新调用模型，所有指标都能从 raw/parsed 产物重算。

## 安装

完整 GPU 环境要求 Linux x86_64、NVIDIA driver 和 [uv](https://docs.astral.sh/uv/)。当前 lock 验证基线为 Python 3.12.11、vLLM 0.25.1+cu129 与 PyTorch 2.11.0+cu129：

```bash
./install.sh .venv
```

`requirements-gpu.lock` 是已验证环境的完整 package 快照，不是跨平台通用 lock。下载代理可通过单次命令的 `DOWNLOAD_PROXY` 环境变量显式提供。

只调用 OpenAI-compatible API 时不需要安装 vLLM、PyTorch 或 CUDA：

```bash
uv python install 3.12.11
uv venv .venv-api --python 3.12.11
uv pip install --python .venv-api/bin/python pyyaml==6.0.3 math-verify==0.9.0
```

## 数据

canonical JSONL 每行至少包含：

```json
{"id":"stable-id","problem":"...","answer":"..."}
```

仓库提供 10 条原创 smoke 数据：

```text
data/example_math.jsonl
```

完整数据与转换来源由 [Geraldxm/math-vault](https://github.com/Geraldxm/math-vault) 维护；其中 `canonical/` 下的 JSONL 是本仓库直接适配的输入。本仓库不复制数据转换管线或完整 benchmark。

## Prompt

Prompt 正文保存在 `prompts/`，YAML 只引用路径：

- completion：单个 UTF-8 文本文件，例如 `prompts/math-completion.txt`。
- chat：按 `<序号>-<role>.txt` 排序的目录，例如 `prompts/math-chat/`。

整个 prompt 必须恰好包含一次 `{{problem}}`。文件内容按 UTF-8 原样读取，不自动删除或补充空白。

## 配置

两个最短入口：

- `configs/qwen35_4b_thinking.yaml`：本地 Qwen3.5-4B thinking 示例。
- `configs/deepseek_api.yaml`：DeepSeek API 单题示例。

其他公开模型示例：

```text
configs/llama32_3b_instruct.yaml
configs/llama33_70b_instruct.yaml
configs/qwen25_3b_base.yaml
configs/qwen25_3b_instruct.yaml
configs/qwen3_4b_base.yaml
configs/qwen3_4b_instruct_2507.yaml
configs/qwen3_next_80b_a3b_instruct.yaml
configs/qwen35_4b_base.yaml
```

这些配置使用公开 model ID；`model.path` 也可以改成本地 checkpoint。Llama 模型仍受其上游访问条件约束。全部已实现字段和注释见：

- `configs/reference_vllm.yaml`
- `configs/reference_openai.yaml`

常改字段为 `dataset`、`prompt`、`model`、`decode` 和 `output`。`decode.temperature`、`top_p`、`seed`、`max_output_tokens`、`samples_per_problem` 必须显式填写；每题多样本时应使用非零 temperature，避免重复 greedy 输出。

本地 chat 使用 checkpoint 自带 tokenizer template。只有模板真实支持 `enable_thinking` 时才能设置 `model.thinking`；启动 preflight 会在加载 GPU 模型前验证该开关。

## 生成与续跑

本地 vLLM：

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/qwen25_3b_instruct.yaml \
  --run-id example-vllm
```

中断后使用相同 config 和 run ID：

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/qwen25_3b_instruct.yaml \
  --run-id example-vllm --resume
```

DeepSeek API：先通过环境变量提供 `DEEPSEEK_API_KEY`，再运行：

```bash
.venv-api/bin/python scripts/evaluate.py \
  --config configs/deepseek_api.yaml \
  --run-id example-deepseek
```

API key 只从环境变量读取，不写入配置或产物。

## 重放评测

默认使用 `math-v5-dual`：boxed-only strict 是正式分数，无完整 `\boxed{}` 时的全文 soft 结果只作诊断。

```bash
.venv/bin/python scripts/replay_evaluation.py \
  --run-dir outputs/runs/example-vllm --k 1 2
```

状态严格区分 `correct`、`incorrect`、`no_candidate`、`parse_error` 和 `verification_error`。parser 只判定 `final_text`；截断事实单独记录，不自动改变 candidate 或 verdict。

比较两个具有相同数据、gold、parser 和 decode 语义的 run：

```bash
.venv/bin/python scripts/compare_runs.py \
  --base outputs/runs/run-a/parsed/math-v5-dual/parsed.jsonl \
  --target outputs/runs/run-b/parsed/math-v5-dual/parsed.jsonl \
  --k 1 --output outputs/comparison.json
```

metrics 的 `pass@k` 使用每题全部样本计算无偏估计；compare 的 solved 集合检查按 `sample_idx` 排序后的前 k 个样本，两者用途不同。

## 产物

每个 run 位于 `outputs/runs/<run-id>/`：

- `manifests/`：不可变 config/prompt 快照、hash、环境和运行状态。
- `raw/`：sample 级 JSONL 或 gzip shards；active shard 为可恢复的 `.jsonl.inprogress`。
- `parsed/<parser-id>/`：逐 sample verdict。
- `metrics/<parser-id>/`：可从 parsed 重算的聚合指标。

## Notebook

- `quickstart.ipynb`：安装、vLLM/API 生成与 replay 主线。
- `math_verify_walkthrough.ipynb`：当前 strict/soft parser 语义、错误分类和已知边界。

两个 notebook 均不在打开时调用 GPU 或 API。

## 检查

```bash
.venv/bin/python -m unittest tests.test_evaluation_core -v
.venv/bin/python -m py_compile scripts/*.py
```

## 引用与许可证

如果本项目对你的研究有帮助，请使用 [`CITATION.cff`](CITATION.cff) 或 [`CITATION.bib`](CITATION.bib) 引用。

Copyright 2026 Xinmu Ge. Licensed under the [Apache License 2.0](LICENSE)；再分发时须保留许可证、版权和适用的署名声明，详见 [`NOTICE`](NOTICE)。
