#!/bin/bash
# setup.sh — One-time project setup
# Run from the olmo-hybrid directory:
#   cd ~/Development/MLX/olmo-hybrid
#   bash setup.sh

set -e

echo "=== OLMo Hybrid MLX — Setup ==="

# Use python3.14 > 3.13 > 3.12 > 3.11 > 3.10, bail if < 3.10
PYTHON=$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "Using: $($PYTHON --version)"

PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
  echo ""
  echo "❌ MLX requires Python 3.10+. You have $($PYTHON --version)."
  echo "   Run: brew install python@3.14"
  echo "   Then re-run this script."
  exit 1
fi

# Create venv
echo ""
echo "Creating venv..."
$PYTHON -m venv .venv
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q

# Core deps
pip install \
    mlx \
    transformers \
    huggingface_hub \
    safetensors \
    numpy

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. python inspect_weights.py          # check HF weight names (fast)"
echo "  2. python convert.py --download --out ./weights   # download + convert (~14GB)"
echo "  3. python generate.py --weights ./weights/weights.npz --prompt 'Hello'"
