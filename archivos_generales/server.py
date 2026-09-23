#!/usr/bin/env python3
"""Servidor local: chat de primeros auxilios con RAG sobre el dataset del MINSA + Ollama (qwen3.5:9b)."""

import csv
import json
import re
import unicodedata
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset_primeros_auxilios.csv"
IMAGES_DIR = BASE_DIR.parent / "imagenes"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
PORT = 8765
TOP_K = 4

# Palabras clave -> imagen ilustrativa asociada
IMAGE_RULES = [
    ("01_rcp_adulto.png", ["rcp", "reanimacion", "reanimación", "paro cardiaco", "compresiones toracicas", "compresiones torácicas", "no respira", "sin pulso"]),
    ("02_atragantamiento_adulto.png", ["atragant", "asfixia", "obstruccion", "obstrucción", "maniobra de heimlich", "no puede respirar", "se ahoga"]),
    ("03_presion_hemorragia.png", ["hemorragia", "sangrado abundante", "sangra mucho", "herida sangrante", "presion directa", "presión directa"]),
    ("04_enfriar_quemadura.png", ["quemadura", "quemad", "se quemo", "se quemó"]),
    ("05_inmovilizar_fractura.png", ["fractura", "hueso roto", "se rompio el hueso", "se rompió el hueso"]),
    ("06_esguince_tobillo.png", ["esguince", "torcio el tobillo", "torció el tobillo", "torcedura"]),
    ("07_posicion_recuperacion.png", ["posicion de recuperacion", "posición de recuperación", "posicion lateral de seguridad", "inconsciente pero respira"]),
    ("08_proteger_convulsion.png", ["convulsion", "convulsión", "ataque epileptico", "ataque epiléptico"]),
    ("09_sangrado_nasal.png", ["sangrado nasal", "hemorragia nasal", "sangra la nariz", "le sangra la nariz"]),
    ("10_cabestrillo_brazo.png", ["cabestrillo", "luxacion", "luxación", "inmovilizar el brazo", "hombro dislocado"]),
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "y", "a",
    "que", "es", "para", "con", "se", "su", "sus", "por", "como", "si", "no", "mi", "me",
    "lo", "le", "les", "al", "o", "u", "esta", "este", "estos", "estas", "hay", "debe",
    "puede", "cual", "cuales", "que", "porque", "cuando", "donde",
}


def tokenize(text: str) -> set:
    return {w for w in normalize(text).split() if w not in STOPWORDS and len(w) > 2}


class KnowledgeBase:
    def __init__(self, csv_path: Path):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_tokens"] = tokenize(row["instruction"] + " " + row.get("category", ""))
                self.rows.append(row)
        print(f"[RAG] Cargadas {len(self.rows)} entradas del dataset.")

    def search(self, query: str, top_k: int = TOP_K):
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored = []
        for row in self.rows:
            overlap = q_tokens & row["_tokens"]
            if not overlap:
                continue
            score = len(overlap) / len(q_tokens | row["_tokens"])
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for score, row in scored[:top_k] if score > 0]


def split_steps(answer: str):
    """Divide la respuesta en pasos (numerados o por linea) para buscar una imagen por cada uno."""
    steps = re.split(r"\n\s*\d+[.)]\s*|\n+", answer)
    steps = [s.strip() for s in steps if s.strip()]
    return steps if steps else [answer]


def find_images(query: str, answer: str = ""):
    """Devuelve una lista de imagenes (sin repetir), una por cada paso/tema detectado,
    para simular una infografia paso a paso con las ilustraciones ya existentes."""
    images = []
    seen = set()

    for text_chunk in [query] + split_steps(answer):
        norm_chunk = normalize(text_chunk)
        for filename, keywords in IMAGE_RULES:
            if filename in seen:
                continue
            for kw in keywords:
                if normalize(kw) in norm_chunk:
                    images.append(filename)
                    seen.add(filename)
                    break

    return images


def build_prompt(query: str, matches: list) -> list:
    if matches:
        contexto = "\n\n".join(
            f"P: {m['instruction']}\nR: {m['output']}" for m in matches
        )
        system = (
            "Eres un asistente educativo de primeros auxilios en Perú, basado en la Cartilla "
            "Educativa del MINSA. Responde SIEMPRE apoyándote en el CONTEXTO proporcionado, que "
            "proviene de una fuente oficial validada. No inventes procedimientos que no estén en "
            "el contexto. Responde en español, de forma clara, completa y detallada, en pasos si aplica. "
            "Si la pregunta no tiene relación con primeros auxilios, dilo con amabilidad. "
            "Termina siempre recordando que ante una emergencia real en Perú se debe llamar al "
            "106 (SAMU) o 116 (bomberos), y que esto no sustituye la atención médica profesional.\n\n"
            f"CONTEXTO:\n{contexto}"
        )
    else:
        system = (
            "Eres un asistente educativo de primeros auxilios en Perú. No encontraste contexto "
            "relevante en la base de conocimiento del MINSA para esta pregunta. Indícalo con "
            "honestidad, da una orientación general muy breve y recomienda llamar al 106 (SAMU) "
            "o 116 (bomberos) ante cualquier emergencia real."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def call_ollama(messages: list):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    texto = data.get("message", {}).get("content", "").strip()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    total_duration_ns = data.get("total_duration", 0)
    eval_seconds = eval_duration_ns / 1e9 if eval_duration_ns else 0
    tokens_por_seg = eval_count / eval_seconds if eval_seconds > 0 else 0

    metrics = {
        "tiempo_segundos": round(total_duration_ns / 1e9, 2),
        "tokens_generados": eval_count,
        "tokens_por_segundo": round(tokens_por_seg, 1),
    }
    return texto, metrics


kb = KnowledgeBase(CSV_PATH)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(BASE_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path.startswith("/imagenes/"):
            filename = self.path.replace("/imagenes/", "", 1)
            self._serve_file(IMAGES_DIR / filename, "image/png")
        else:
            self._send_json({"error": "not found"}, 404)

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            query = payload["message"].strip()
        except Exception:
            self._send_json({"error": "payload invalido"}, 400)
            return

        matches = kb.search(query)
        messages = build_prompt(query, matches)
        try:
            answer, metrics = call_ollama(messages)
        except Exception as e:
            self._send_json({"error": f"No se pudo contactar a Ollama: {e}"}, 500)
            return

        images = find_images(query, answer)
        self._send_json({
            "answer": answer,
            "images": [f"/imagenes/{img}" for img in images],
            "sources": [m["instruction"] for m in matches],
            "metrics": metrics,
        })


def main():
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"Servidor corriendo en http://localhost:{PORT}")
    print(f"Modelo Ollama: {MODEL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
