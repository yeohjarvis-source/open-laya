#!/bin/zsh
set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
models_root="${LAYA_OMNI_MODELS_PATH:-$repo_root/models}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is missing. Install open-laya with the omni extra first:" >&2
  echo "  uv pip install -e '.[mlx,omni]'" >&2
  exit 1
fi

echo "Downloading third-party model weights to $models_root"
echo "Laya Omni weights are for research and non-commercial use."
echo "Terms: https://huggingface.co/zheqiushui/laya-omni"
mkdir -p "$models_root"

hf download zheqiushui/laya-omni \
  --local-dir "$models_root/laya-omni"
hf download convaiinnovations/laya-multilingual \
  --local-dir "$models_root/laya-multilingual"
hf download google/siglip2-base-patch16-256 \
  --local-dir "$models_root/siglip2-base-patch16-256"

echo "Models downloaded. Set LAYA_OMNI_MODELS_PATH=$models_root"
