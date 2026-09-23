#!/usr/bin/env python3
"""Prepara las 10 imagenes de referencia con sus captions para el fine-tuning LoRA de SDXL (mlx)."""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGES_SRC = BASE_DIR.parent.parent / "imagenes"
OUT_DIR = BASE_DIR / "datos_mlx_imagenes"

# Mismos captions usados en el proyecto original (activador PAUX_STYLE)
CAPTIONS = {
    "01_rcp_adulto.png": "PAUX_STYLE educational first-aid illustration of an adult responder kneeling beside an adult training mannequin and performing chest compressions with interlocked hands, straight arms and shoulders above the center of the chest",
    "02_atragantamiento_adulto.png": "PAUX_STYLE educational first-aid illustration of a conscious choking adult while a responder stands behind and performs abdominal thrusts with one fist above the navel and the other hand grasping it",
    "03_presion_hemorragia.png": "PAUX_STYLE educational first-aid illustration of a responder controlling bleeding from an adult forearm by applying firm direct pressure with clean folded gauze while the arm rests supported on a table",
    "04_enfriar_quemadura.png": "PAUX_STYLE educational first-aid illustration of a responder cooling a burn on an adult hand under gentle running water",
    "05_inmovilizar_fractura.png": "PAUX_STYLE educational first-aid illustration of a responder immobilizing a fractured leg with a rigid splint and padded bandages",
    "06_esguince_tobillo.png": "PAUX_STYLE educational first-aid illustration of a responder applying cold compress to a sprained ankle",
    "07_posicion_recuperacion.png": "PAUX_STYLE educational first-aid illustration of an unconscious but breathing adult placed in the recovery position on their side",
    "08_proteger_convulsion.png": "PAUX_STYLE educational first-aid illustration of a responder protecting the head of a person having a seizure, clearing nearby objects",
    "09_sangrado_nasal.png": "PAUX_STYLE educational first-aid illustration of a responder helping a person tilt their head forward while pinching the nose to stop a nosebleed",
    "10_cabestrillo_brazo.png": "PAUX_STYLE educational first-aid illustration of a responder placing an injured arm in a sling for immobilization",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT_DIR / "metadata.jsonl"

    count = 0
    with open(metadata_path, "w", encoding="utf-8") as f:
        for filename, caption in CAPTIONS.items():
            src = IMAGES_SRC / filename
            if not src.exists():
                print(f"AVISO: no se encontro {src}, se omite.")
                continue
            dst = OUT_DIR / filename
            shutil.copy(src, dst)
            f.write(json.dumps({"file_name": filename, "text": caption}, ensure_ascii=False) + "\n")
            count += 1

    print(f"{count} imagenes copiadas a {OUT_DIR}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
