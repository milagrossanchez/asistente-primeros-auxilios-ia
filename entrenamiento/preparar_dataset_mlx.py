#!/usr/bin/env python3
"""Convierte dataset_primeros_auxilios.csv al formato JSONL de chat que espera mlx-lm para fine-tuning LoRA."""

import csv
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "dataset" / "dataset_primeros_auxilios.csv"
OUT_DIR = BASE_DIR / "datos_mlx"

SYSTEM_PROMPT = (
    "Eres un asistente educativo de primeros auxilios en Peru, basado en la Cartilla "
    "Educativa del MINSA. Responde de forma breve, clara y en pasos si aplica. "
    "Recuerda que ante una emergencia real se debe llamar al 106 (SAMU) o 116 (bomberos)."
)

random.seed(42)


def main():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Total de ejemplos: {len(rows)}")

    random.shuffle(rows)
    n_valid = max(1, int(len(rows) * 0.1))
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def to_chat_example(row):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ]
        }

    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(to_chat_example(row), ensure_ascii=False) + "\n")

    with open(OUT_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for row in valid_rows:
            f.write(json.dumps(to_chat_example(row), ensure_ascii=False) + "\n")

    print(f"train.jsonl: {len(train_rows)} ejemplos")
    print(f"valid.jsonl: {len(valid_rows)} ejemplos")
    print(f"Guardado en: {OUT_DIR}")


if __name__ == "__main__":
    main()
