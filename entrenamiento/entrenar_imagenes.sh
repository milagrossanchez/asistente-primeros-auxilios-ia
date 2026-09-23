

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

source venv/bin/activate

echo "== 1. Preparando dataset de imagenes =="
python3 preparar_dataset_imagenes.py

echo "== 2. Entrenando LoRA de SDXL (esto puede tardar horas en Mac M4) =="
export PYTORCH_ENABLE_MPS_FALLBACK=1

accelerate launch train_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/sdxl-turbo" \
  --train_data_dir="./datos_mlx_imagenes" \
  --resolution=384 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=100 \
  --learning_rate=1e-4 \
  --rank=4 \
  --output_dir="./adaptador_lora_imagenes" \
  --mixed_precision="no"

echo "== Entrenamiento de imagen completo =="
echo "Adaptador guardado en: $DIR/adaptador_lora_imagenes"
