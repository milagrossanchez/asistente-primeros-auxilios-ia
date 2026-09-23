#!/bin/bash
# Prepara el entorno para fine-tuning local en Apple Silicon (MLX).
# Requiere Homebrew ya instalado.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "== 1. Instalando Python 3.12 via Homebrew =="
brew install python@3.12

echo "== 2. Creando entorno virtual =="
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate

echo "== 3. Instalando dependencias (mlx-lm, mlx, huggingface_hub) =="
pip install --upgrade pip
pip install mlx-lm mlx huggingface_hub

echo "== Entorno listo =="
echo "Activa con: source $DIR/venv/bin/activate"
