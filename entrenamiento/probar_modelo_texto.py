#!/usr/bin/env python3
"""Prueba rapida del modelo Qwen2.5-7B con el adaptador LoRA de primeros auxilios ya cargado."""

import sys
from pathlib import Path
from mlx_lm import load, generate

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = BASE_DIR / "adaptador_lora_texto"
MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SYSTEM_PROMPT = (
    "Eres un asistente educativo de primeros auxilios en Peru, basado en la Cartilla "
    "Educativa del MINSA. Responde de forma breve, clara y en pasos si aplica. "
    "Recuerda que ante una emergencia real se debe llamar al 106 (SAMU) o 116 (bomberos)."
)


def main():
    pregunta = sys.argv[1] if len(sys.argv) > 1 else "Alguien se esta atragantando, que hago?"

    print(f"Cargando modelo + adaptador LoRA desde {ADAPTER_PATH} ...")
    model, tokenizer = load(str(MODEL), adapter_path=str(ADAPTER_PATH))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pregunta},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print(f"\nPregunta: {pregunta}\n")
    print("Respuesta del modelo entrenado:")
    respuesta = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=True)


if __name__ == "__main__":
    main()
