#!/bin/bash
# Entorno para fine-tuning LoRA de Stable Diffusion en Apple Silicon.
# NOTA: mlx-lm es solo para modelos de texto. Para imagenes usamos diffusers
# de Hugging Face con backend MPS (Metal), que si soporta entrenamiento LoRA.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d "venv" ]; then
  echo "Creando entorno virtual (usa el mismo Python 3.12 del setup de texto)..."
  /opt/homebrew/bin/python3.12 -m venv venv
fi

source venv/bin/activate

echo "== Instalando dependencias para LoRA de imagen (diffusers + torch con MPS) =="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate peft datasets pillow

echo "== Descargando script oficial de entrenamiento LoRA SDXL de diffusers =="
curl -fsSL -o train_lora_sdxl.py \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py

echo "== Entorno de imagen listo =="
