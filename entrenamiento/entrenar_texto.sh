

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

source venv/bin/activate

echo "== 1. Preparando dataset en formato MLX =="
python3 preparar_dataset_mlx.py

echo "== 2. Entrenando LoRA (Qwen2.5-7B-Instruct) =="
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --train \
  --data ./datos_mlx \
  --iters 300 \
  --batch-size 2 \
  --num-layers 8 \
  --adapter-path ./adaptador_lora_texto \
  --save-every 100

echo "== Entrenamiento de texto completo =="
echo "Adaptador guardado en: $DIR/adaptador_lora_texto"
