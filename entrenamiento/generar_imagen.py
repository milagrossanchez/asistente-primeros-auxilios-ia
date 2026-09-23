#!/usr/bin/env python3
"""Genera una imagen nueva con SDXL-Turbo + el adaptador LoRA entrenado localmente
sobre las 10 imagenes de primeros auxilios. Uso:

    python3 generar_imagen.py "PAUX_STYLE educational first-aid illustration of ..." salida.png
"""

import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = BASE_DIR / "adaptador_lora_imagenes" / "pytorch_lora_weights.safetensors"
MODEL_NAME = "stabilityai/sdxl-turbo"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "PAUX_STYLE educational first-aid illustration of an adult responder kneeling beside an adult training mannequin and performing chest compressions"
    salida = sys.argv[2] if len(sys.argv) > 2 else "prueba_generada.png"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    print(f"Cargando modelo base: {MODEL_NAME}")
    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(device)
    print(f"Modelo base cargado en {time.time() - t0:.1f}s")

    if ADAPTER_PATH.exists():
        print(f"Cargando adaptador LoRA entrenado: {ADAPTER_PATH}")
        pipe.load_lora_weights(str(ADAPTER_PATH.parent))
    else:
        print(f"AVISO: no se encontro el adaptador en {ADAPTER_PATH}, se usa el modelo base sin LoRA.")

    # El decode del VAE de SDXL se cuelga en MPS con ciertos tamaños de tensor
    # (bug conocido de PyTorch en Apple Silicon). Se ejecuta en CPU como workaround.
    print("Moviendo VAE a CPU para evitar cuelgue conocido de MPS...")
    pipe.vae.to("cpu")

    print(f"\nPrompt: {prompt}\n")
    print("Generando imagen...")
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=384,
        width=384,
        output_type="latent",
    ).images
    print(f"Denoise completo en {time.time() - t0:.1f}s, decodificando en CPU...")

    t1 = time.time()
    with torch.no_grad():
        image = image.to("cpu").to(torch.float32) / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(image).sample
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    print(f"Decode completo en {time.time() - t1:.1f}s")

    elapsed = time.time() - t0
    print(f"Generada en {elapsed:.1f}s (total)")

    image.save(salida)
    print(f"Guardada en: {salida}")


if __name__ == "__main__":
    main()
