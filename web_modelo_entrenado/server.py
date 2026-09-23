#!/usr/bin/env python3
"""Servidor local: chat de primeros auxilios usando el modelo Qwen2.5-7B-Instruct
con el adaptador LoRA entrenado localmente (MLX), sin RAG -- respuestas 100% del
conocimiento aprendido durante el fine-tuning."""

import json
import re
import unicodedata
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import time
from mlx_lm import load, generate

BASE_DIR = Path(__file__).resolve().parent
ENTRENAMIENTO_DIR = BASE_DIR.parent / "entrenamiento"
ADAPTER_PATH = ENTRENAMIENTO_DIR / "adaptador_lora_texto"
IMAGE_ADAPTER_PATH = ENTRENAMIENTO_DIR / "adaptador_lora_imagenes" / "pytorch_lora_weights.safetensors"
IMAGES_DIR = BASE_DIR.parent / "imagenes"
GENERADAS_DIR = BASE_DIR / "imagenes_generadas"
GENERADAS_DIR.mkdir(exist_ok=True)
MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"
SD_MODEL_NAME = "stabilityai/sdxl-turbo"
PORT = 8766

# Traduccion breve de temas detectados a prompts en ingles, en el estilo con el
# que se entreno el LoRA de imagen (activador PAUX_STYLE).
IMAGE_PROMPTS = {
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

sd_pipe = None
GENERACION_IMAGEN_DISPONIBLE = IMAGE_ADAPTER_PATH.exists()


def cargar_pipeline_imagen():
    """Carga SDXL-Turbo + el LoRA de imagen entrenado, solo la primera vez que se necesita."""
    global sd_pipe
    if sd_pipe is not None:
        return sd_pipe

    import torch
    from diffusers import AutoPipelineForText2Image

    print(f"[imagen] Cargando modelo base: {SD_MODEL_NAME}")
    pipe = AutoPipelineForText2Image.from_pretrained(SD_MODEL_NAME, torch_dtype=torch.float32).to("mps")
    print(f"[imagen] Cargando adaptador LoRA entrenado: {IMAGE_ADAPTER_PATH}")
    pipe.load_lora_weights(str(IMAGE_ADAPTER_PATH.parent))

    # El decode del VAE de SDXL se cuelga en MPS con ciertos tamaños de tensor
    # (bug conocido de PyTorch en Apple Silicon). Se ejecuta en CPU como workaround.
    pipe.vae.to("cpu")

    sd_pipe = pipe
    print("[imagen] Pipeline de generacion listo.")
    return sd_pipe


def generar_imagen(filename_base: str) -> str:
    """Genera una imagen nueva con el LoRA de imagen entrenado, usando el mismo
    caption con el que se entreno esa ilustracion. Devuelve la ruta relativa
    del archivo generado."""
    import torch

    prompt = IMAGE_PROMPTS.get(filename_base)
    if prompt is None:
        return None

    pipe = cargar_pipeline_imagen()

    latents = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=384,
        width=384,
        output_type="latent",
    ).images

    with torch.no_grad():
        decoded = latents.to("cpu").to(torch.float32) / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(decoded).sample
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    out_name = f"{Path(filename_base).stem}_{uuid.uuid4().hex[:8]}.png"
    out_path = GENERADAS_DIR / out_name
    image.save(out_path)
    return out_name

SYSTEM_PROMPT = (
    "Eres un asistente educativo de primeros auxilios en Peru, basado en la Cartilla "
    "Educativa del MINSA. Responde de forma clara, completa y detallada, explicando cada "
    "paso a seguir cuando aplique. "
    "Recuerda que ante una emergencia real se debe llamar al 106 (SAMU) o 116 (bomberos)."
)

IMAGE_RULES = [
    ("01_rcp_adulto.png", ["rcp", "reanimacion", "reanimación", "paro cardiaco", "compresiones toracicas", "compresiones torácicas", "no respira", "sin pulso"]),
    ("02_atragantamiento_adulto.png", ["atragant", "asfixia", "obstruccion", "obstrucción", "maniobra de heimlich", "no puede respirar", "se ahoga"]),
    ("03_presion_hemorragia.png", ["hemorragia", "sangrado abundante", "sangra mucho", "herida sangrante", "presion directa", "presión directa"]),
    ("04_enfriar_quemadura.png", ["quemadura", "quemad", "se quemo", "se quemó"]),
    ("05_inmovilizar_fractura.png", ["fractura", "hueso roto", "se rompio el hueso", "se rompió el hueso"]),
    ("06_esguince_tobillo.png", ["esguince", "torcio el tobillo", "torció el tobillo", "torcedura"]),
    ("07_posicion_recuperacion.png", ["posicion de recuperacion", "posición de recuperación", "posicion lateral de seguridad", "inconsciente pero respira"]),
    ("08_proteger_convulsion.png", ["convulsion", "convulsión", "ataque epileptico", "ataque epiléptico"]),
    ("09_sangrado_nasal.png", ["sangrado nasal", "hemorragia nasal", "sangra la nariz", "le sangra la nariz", "sangra por la nariz"]),
    ("10_cabestrillo_brazo.png", ["cabestrillo", "luxacion", "luxación", "inmovilizar el brazo", "hombro dislocado"]),
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


print(f"Cargando modelo base: {MODEL_NAME}")
if ADAPTER_PATH.exists():
    print(f"Cargando adaptador LoRA entrenado: {ADAPTER_PATH}")
    model, tokenizer = load(MODEL_NAME, adapter_path=str(ADAPTER_PATH))
    MODELO_ENTRENADO = True
else:
    print(f"AVISO: no se encontro el adaptador en {ADAPTER_PATH}. Se usara el modelo base SIN fine-tuning.")
    model, tokenizer = load(MODEL_NAME)
    MODELO_ENTRENADO = False
print("Modelo listo.")


def generar_respuesta(pregunta: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pregunta},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    start = time.time()
    texto = generate(model, tokenizer, prompt=prompt, max_tokens=700, verbose=False).strip()
    elapsed = time.time() - start

    num_tokens = len(tokenizer.encode(texto))
    tokens_por_seg = num_tokens / elapsed if elapsed > 0 else 0

    metrics = {
        "tiempo_segundos": round(elapsed, 2),
        "tokens_generados": num_tokens,
        "tokens_por_segundo": round(tokens_por_seg, 1),
    }
    return texto, metrics


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
        elif self.path == "/api/status":
            self._send_json({
                "modelo_entrenado": MODELO_ENTRENADO,
                "modelo": MODEL_NAME,
                "generacion_imagen_disponible": GENERACION_IMAGEN_DISPONIBLE,
            })
        elif self.path.startswith("/imagenes/"):
            filename = self.path.replace("/imagenes/", "", 1)
            self._serve_file(IMAGES_DIR / filename, "image/png")
        elif self.path.startswith("/imagenes_generadas/"):
            filename = self.path.replace("/imagenes_generadas/", "", 1)
            self._serve_file(GENERADAS_DIR / filename, "image/png")
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
        if self.path == "/api/chat":
            self._handle_chat()
        elif self.path == "/api/generar-imagen":
            self._handle_generar_imagen()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            query = payload["message"].strip()
        except Exception:
            self._send_json({"error": "payload invalido"}, 400)
            return

        try:
            answer, metrics = generar_respuesta(query)
        except Exception as e:
            self._send_json({"error": f"Error generando respuesta: {e}"}, 500)
            return

        temas = find_images(query, answer)
        self._send_json({
            "answer": answer,
            "images": [f"/imagenes/{img}" for img in temas],
            "temas": temas,
            "modelo_entrenado": MODELO_ENTRENADO,
            "metrics": metrics,
        })

    def _handle_generar_imagen(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            tema = payload["tema"]
        except Exception:
            self._send_json({"error": "payload invalido"}, 400)
            return

        if not GENERACION_IMAGEN_DISPONIBLE:
            self._send_json({"error": "El adaptador LoRA de imagen no esta disponible."}, 400)
            return

        try:
            t0 = time.time()
            generado = generar_imagen(tema)
            self._send_json({
                "imagen_generada": f"/imagenes_generadas/{generado}" if generado else None,
                "segundos": round(time.time() - t0, 1),
            })
        except Exception as e:
            self._send_json({"error": f"Error generando imagen: {e}"}, 500)


def main():
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Servidor corriendo en http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
