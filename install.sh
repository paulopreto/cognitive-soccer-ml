#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Instala uv, cria .venv com Python 3.12.12, instala dependências e testa.
# Uso: bash install.sh   (Linux/macOS, na raiz do projeto)
# Windows: instale uv depois rode: uv venv --python 3.12.12 --clear ; uv sync
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- 1. Garantir que uv está instalado e no PATH ---
if ! command -v uv &>/dev/null; then
  echo ">>> Instalando uv..."
  INSTALLER_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
  if command -v curl &>/dev/null; then
    curl -LsSf "$INSTALLER_URL" | sh
  elif command -v wget &>/dev/null; then
    wget -qO- "$INSTALLER_URL" | sh
  else
    echo "Erro: é necessário curl ou wget. Instale um deles e tente novamente."
    exit 1
  fi
  export PATH="${HOME}/.local/bin:${PATH:-}"
  if ! command -v uv &>/dev/null; then
    echo "uv instalado em ~/.local/bin. Rode: export PATH=\"\${HOME}/.local/bin:\$PATH\" e execute este script de novo."
    exit 1
  fi
fi

echo ">>> uv: $(uv --version)"

# --- 2. Criar .venv (substituir se já existir) e instalar dependências ---
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo ">>> Criando .venv com Python 3.12.12..."
uv venv --python 3.12.12 --clear

echo ">>> Instalando dependências..."
uv sync

# --- 3. Teste rápido ---
echo ">>> Testando ambiente..."
uv run python -c "
import sys
import numpy as np
import pandas as pd
import sklearn
print('Python:', sys.version.split()[0])
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('sklearn:', sklearn.__version__)
print('OK — ambiente funcionando.')
"

echo ""
echo "Pronto. Use: source .venv/bin/activate   ou   uv run python src/Complete_trainning.py"
