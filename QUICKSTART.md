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

### Generar una imagen nueva con el LoRA de imagen (ya entrenado, opcional)

El repositorio ya incluye el adaptador `entrenamiento/adaptador_lora_imagenes/` (SDXL-Turbo + LoRA, entrenado sobre las 10 ilustraciones). El chat **no genera esta imagen automáticamente** junto con la respuesta de texto, porque toma varios minutos:

> Debajo de cada ilustración fija, aparece un botón **"🎨 Generar con IA (LoRA)"**. Hay que hacer click ahí explícitamente para que el servidor cargue SDXL-Turbo (la primera vez tarda más) y genere una imagen nueva desde cero con ese tema — puede tomar 3-5 minutos por imagen en un Mac Mini M4.

Si prefieres probarlo por consola en vez de la web:

```bash
cd entrenamiento
source venv/bin/activate
python3 generar_imagen.py "PAUX_STYLE educational first-aid illustration of an adult responder kneeling beside an adult training mannequin and performing chest compressions" salida.png
```

### Volver a entrenar el modelo de imagen desde cero (opcional)

No es necesario para usar el chat — el adaptador ya está entrenado y en el repo. Solo hace falta si quieres reentrenarlo con otros parámetros.

**Aviso:** con solo 10 imágenes de entrenamiento, este LoRA aprende principalmente el estilo visual, no maniobras nuevas — el resultado tiene ruido y artefactos visibles. Tomó **~9.5 horas** en un Mac Mini M4 de 16 GB (mucho más lento de lo esperado, ya que MPS no está tan optimizado como CUDA para este tipo de entrenamiento).

```bash
cd entrenamiento
./setup_entorno_imagenes.sh   # instala PyTorch + diffusers (una sola vez)
./entrenar_imagenes.sh        # entrena el LoRA de SDXL-Turbo
```

## Ambos chats a la vez

Puedes correr las dos webs en paralelo (puertos distintos, 8765 y 8766) para comparar respuestas RAG vs. modelo entrenado sobre la misma pregunta.

## Notas

- Las ilustraciones y cualquier LoRA visual son una demostración académica. Las maniobras deben validarse con profesionales antes de utilizarlas como material sanitario.
- Ante una emergencia real en Perú: 106 (SAMU) o 116 (bomberos).
