# math-eval

可续传、可重放的数学推理生成与评测工具。它读取 canonical JSONL，通过本地 vLLM 或 OpenAI-compatible Chat Completions 生成答案，再独立解析、统计和比较。

```text
canonical JSONL -> evaluate.py -> raw shards -> replay_evaluation.py -> parsed + metrics
                                                        |
                                                        +-> compare_runs.py
```

生成与判分刻意分离：更换 parser 或指标时不需要重新调用模型，所有指标都能从 raw/parsed 产物重算。

## 与其他评测框架的区别

大多数评测框架在 CLI 层面统一了多 benchmark 调用，但每个 benchmark 背后使用不同的数据格式、答案解析器和正确性判断逻辑。math-eval 在更深的层级做统一：**数据格式（canonical JSONL）+ 验证语义（统一 Math-Verify pipeline）**。

| | lm-eval-harness / LightEval | math-eval + math-vault |
|---|---:|---|
| 统一层级 | CLI（`--tasks gsm8k,math`） | 数据层（canonical JSONL）+ 验证层（统一 Math-Verify） |
| 新增 benchmark | 编写新 task 模块（数据加载 + 自定义解析 + 聚合） | 将数据转为 `id/problem/answer` JSONL |
| 更换 parser/指标 | 需重新调用模型 | `replay_evaluation.py` 从 raw shards 直接重算 |
| 中断恢复 | 通常从头开始 | `--resume` 从断点继续 |
| 数据 provenance | 不追踪 | math-vault 逐数据集记录上游来源与许可证 |

math-eval 不是 lm-eval-harness 的替代品。需要快速覆盖 50+ benchmark 出 leaderboard 时，lm-eval-harness 或 LightEval 更合适。需要对同一批数据反复做严谨评测——换 parser、换指标、对比不同配置——时，math-eval 的生成与判分分离架构是更趁手的工具。

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

### 统一验证语义

本仓库统一使用 [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify) 0.9.0 解析和比较数学答案，不为不同 benchmark 分别维护 integer、number、expression grader，也不提供 last-number fallback 或题目级等价特判。

Math-Verify 上游负责从给定文本中提取数学表达式、转换为公共数学表示，并执行数值或符号等价判断。本仓库在它之上增加一层可重放的评测协议：

| 层 | 职责 |
|---|---|
| canonical data | 所有数据集统一提供字符串形式的 `answer` |
| math-eval 本地适配 | 选择 candidate，定义 strict/soft 与截断语义，分类失败状态，记录 parser ID 和配置 hash |
| Math-Verify 0.9.0 | 使用统一 extraction config 解析 gold/prediction，并执行数学等价判断 |
| metrics | 从冻结的 parsed verdict 重算 accuracy、pass@k、failure rate 和 truncation rate |

gold 与 prediction 最终共用 `LatexExtractionConfig(boxed_match_priority=0)` 和 `ExprExtractionConfig()`。本仓库没有重新实现分数、小数、代数式、tuple、复数、集合或矩阵的等价规则；这些能力、容差与已知边界继承自固定版本的 Math-Verify。具体适配实现在 [`scripts/parser.py`](scripts/parser.py)，可执行案例与边界见 [`math_verify_walkthrough.ipynb`](math_verify_walkthrough.ipynb)。

### Strict、soft 与截断

`math-v5-dual` 的 strict 与 soft 只在 candidate selection 上不同，之后调用完全相同的 Math-Verify `parse()` / `verify()`：

- **strict**：只接受最后一个完整的 `\boxed{}`，用于正式指标。
- **soft**：仅当不存在完整 box 时，才将非空 `final_text` 全文交给 Math-Verify，用于诊断"答案可能正确但未遵守 boxed 输出协议"的样本。
- 如果存在完整 box，strict 与 soft 使用同一个 candidate 和 verdict；即使 box 内答案错误，soft 也不会回退到正文寻找另一个答案。

`truncated` 是与 strict/soft 正交的生成状态。当前 parser 不会因为输出被截断就自动判错或切换提取策略；真正影响 candidate 的是截断后是否仍存在完整 box：

| `final_text` 状态 | `truncated` | strict | soft |
|---|---:|---|---|
| 存在完整 `\boxed{}` | `false` | 验证最后一个完整 box | 与 strict 相同 |
| 存在完整 box，但之后被截断 | `true` | 仍验证最后一个完整 box | 与 strict 相同 |
| 末尾 box 未闭合，但此前存在完整 box | `true` | 忽略未闭合部分，验证此前最后一个完整 box | 与 strict 相同 |
| 没有完整 box，文本非空 | `false` 或 `true` | `no_candidate` | 尝试从已有全文提取并验证 |
| 文本为空 | `false` 或 `true` | `no_candidate` | `no_candidate` |

因此，截断率单独进入 metrics，但不自动改变 candidate 或 verdict。若希望把"截断后一律判错"作为另一种评测协议，应创建新的 parser ID，而不是在同一 parser 版本下改变历史结果。

状态严格区分 `correct`、`incorrect`、`no_candidate`、`parse_error` 和 `verification_error`：`incorrect` 表示解析成功但数学上不等价，后三者分别表示没有候选、候选解析失败和等价验证异常。它们在正式指标中都按未答对计入，同时保留独立计数，避免把格式失败或 verifier 异常混成普通数学错误。

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

如果本项目对你的研究有帮助，请使用 [`CITATION.cff`](CITATION.cff) 引用或直接复制以下 BibTeX：

```bibtex
@software{ge_math_eval_2026,
  author  = {Ge, Xinmu},
  title   = {math-eval: Reproducible Mathematical Reasoning Generation and Evaluation},
  year    = {2026},
  url     = {https://github.com/Geraldxm/math-eval},
  license = {Apache-2.0}
}
```

Copyright 2026 Xinmu Ge. Licensed under the [Apache License 2.0](LICENSE)；再分发时须保留许可证、版权和适用的署名声明，详见 [`NOTICE`](NOTICE)。
