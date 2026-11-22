#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$ROOT/.venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.11}"
# PyTorch CUDA backend for uv. "auto" detects the local NVIDIA driver; on a machine
# without a GPU (e.g. a CPU instance preparing an env for GPU jobs) set TORCH_BACKEND=cu129.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"
# Optional PyPI mirror: uv reads UV_DEFAULT_INDEX directly. It only affects regular
# PyPI packages; torch* come from the PyTorch index and vLLM from its direct wheel URL in the lock.

if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
  echo "requirements-gpu.lock supports Linux x86_64 only" >&2
  exit 1
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [[ -n "${DOWNLOAD_PROXY:-}" ]]; then
  export HTTP_PROXY="$DOWNLOAD_PROXY" HTTPS_PROXY="$DOWNLOAD_PROXY"
  export http_proxy="$DOWNLOAD_PROXY" https_proxy="$DOWNLOAD_PROXY"
fi

echo "torch backend: $TORCH_BACKEND; default index: ${UV_DEFAULT_INDEX:-https://pypi.org/simple (uv default)}"
"$UV_BIN" python install "$PYTHON_VERSION"
"$UV_BIN" venv "$ENV_DIR" --python "$PYTHON_VERSION" --seed
"$UV_BIN" pip sync "$ROOT/requirements-gpu.lock" \
  --python "$ENV_DIR/bin/python" --torch-backend="$TORCH_BACKEND"
"$UV_BIN" pip check --python "$ENV_DIR/bin/python"
"$ENV_DIR/bin/python" -c 'import torch, vllm; print("python env ok", torch.__version__, vllm.__version__, torch.cuda.is_available())'
