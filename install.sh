#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Install uv, create .venv with Python 3.12.12, install dependencies and test.
# Usage: bash install.sh   (Linux/macOS, from project root)
# Windows: install uv then run: uv venv --python 3.12.12 --clear ; uv sync
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- 1. Ensure uv is installed and on PATH ---
if ! command -v uv &>/dev/null; then
  echo ">>> Installing uv..."
  INSTALLER_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
  if command -v curl &>/dev/null; then
    curl -LsSf "$INSTALLER_URL" | sh
  elif command -v wget &>/dev/null; then
    wget -qO- "$INSTALLER_URL" | sh
  else
    echo "Error: curl or wget is required. Install one of them and try again."
    exit 1
  fi
  export PATH="${HOME}/.local/bin:${PATH:-}"
  if ! command -v uv &>/dev/null; then
    echo "uv was installed to ~/.local/bin. Run: export PATH=\"\${HOME}/.local/bin:\$PATH\" and run this script again."
    exit 1
  fi
fi

echo ">>> uv: $(uv --version)"

# --- 2. Create .venv (replace if it exists) and install dependencies ---
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo ">>> Creating .venv with Python 3.12.12..."
uv venv --python 3.12.12 --clear

echo ">>> Installing dependencies..."
uv sync

# --- 3. Quick test ---
echo ">>> Testing environment..."
uv run python -c "
import sys
import numpy as np
import pandas as pd
import sklearn
print('Python:', sys.version.split()[0])
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('sklearn:', sklearn.__version__)
print('OK — environment is working.')
"

echo ""
echo "Done. Use: source .venv/bin/activate   or   uv run python src/Complete_trainning.py"
