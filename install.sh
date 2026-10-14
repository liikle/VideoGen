#!/usr/bin/env bash
# One-time setup for Linux and macOS: creates .venv and installs VideoGen with the right PyTorch build.
# Override the CUDA build with TORCH_CUDA=cu126 (older NVIDIA drivers) or cu132.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu130}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.10+ is required (set PYTHON=/path/to/python if it is not on PATH)." >&2
  exit 1
fi
"$PYTHON" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' || {
  echo "Python 3.10 - 3.13 is required; found $("$PYTHON" --version)." >&2
  exit 1
}

echo "==> Creating virtual environment in .venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

if [[ "$(uname)" == "Darwin" ]]; then
  echo "==> Installing PyTorch for Apple Silicon (MPS)"
  pip install torch torchvision
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Installing PyTorch with CUDA ($TORCH_CUDA)"
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
else
  echo "==> No NVIDIA GPU found: installing CPU PyTorch (only the test engine will be usable)"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing VideoGen"
pip install -e .

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "    (system ffmpeg not found; using the bundled imageio-ffmpeg binary)"
fi

cat <<'EOF'

VideoGen is installed.

Next steps:
  1. LTX-2.5 (best model) needs a free Hugging Face license click:
       open https://huggingface.co/Lightricks/LTX-2.5-Diffusers, click "Agree and access repository",
       then run:  .venv/bin/hf auth login
  2. Optional: pre-download weights now instead of on first use:
       .venv/bin/videogen download ltx        # or wan-5b, wan-14b, all
  3. Start the app:  ./run.sh   and open http://localhost:7860
EOF
