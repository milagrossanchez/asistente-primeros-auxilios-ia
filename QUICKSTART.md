# Guía rápida (local, Mac Apple Silicon)

Todo corre en tu Mac, sin Colab ni GPU NVIDIA. Hay dos chats independientes que puedes usar por separado.

## Opción A — Chat RAG (el más rápido de arrancar)

No requiere entrenamiento. Usa el dataset del MINSA como contexto para el modelo `qwen3.5:9b` corriendo en **Ollama**.

Requisitos: tener [Ollama](https://ollama.com) instalado y corriendo, con el modelo descargado (`ollama pull qwen3.5:9b`).

```bash
cd archivos_generales
python3 server.py
```

Abre **http://localhost:8765**

## Opción B — Chat con el modelo entrenado (fine-tuning LoRA local)

Usa `Qwen2.5-7B-Instruct` con el adaptador LoRA entrenado sobre el dataset del MINSA, cargado directamente con MLX (sin Ollama).

### 1. Preparar el entorno (una sola vez)

Requiere [Homebrew](https://brew.sh) instalado.

```bash
cd entrenamiento
./setup_entorno.sh
```

Esto instala Python 3.12 y crea un entorno virtual (`venv/`) con `mlx-lm`.

### 2. Entrenar el modelo de texto

```bash
cd entrenamiento
./entrenar_texto.sh
```

Convierte el dataset a formato de chat, descarga `Qwen2.5-7B-Instruct-4bit` (~4-5 GB, solo la primera vez) y entrena el LoRA (300 iteraciones, ~10-15 minutos en un M4). El adaptador queda guardado en `entrenamiento/adaptador_lora_texto/`.

Para probarlo rápido por consola, sin la web:

```bash
source venv/bin/activate
python3 probar_modelo_texto.py "Alguien se está atragantando, qué hago?"
```

### 3. Levantar la web

```bash
cd web_modelo_entrenado
source ../entrenamiento/venv/bin/activate
python3 server.py
```

Abre **http://localhost:8766**

La barra de estado en la web te confirma si está usando el modelo ya entrenado o todavía el modelo base.

## Opción C — Entrenar también el modelo de imagen (opcional, lento)

**Aviso:** con solo 10 imágenes de entrenamiento, este LoRA aprende principalmente el estilo visual, no maniobras nuevas. El chat ya funciona con las 10 ilustraciones existentes sin necesidad de este paso — es un ejercicio académico adicional, no una mejora garantizada. Puede tomar 1-3+ horas en un Mac de 16 GB.

```bash
cd entrenamiento
./setup_entorno_imagenes.sh   # instala PyTorch + diffusers (una sola vez)
./entrenar_imagenes.sh        # entrena el LoRA de SDXL-Turbo
```

El adaptador queda en `entrenamiento/adaptador_lora_imagenes/`.

## Ambos chats a la vez

Puedes correr las dos webs en paralelo (puertos distintos, 8765 y 8766) para comparar respuestas RAG vs. modelo entrenado sobre la misma pregunta.

## Notas

- Las ilustraciones y cualquier LoRA visual son una demostración académica. Las maniobras deben validarse con profesionales antes de utilizarlas como material sanitario.
- Ante una emergencia real en Perú: 106 (SAMU) o 116 (bomberos).
