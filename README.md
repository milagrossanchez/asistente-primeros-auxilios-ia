# Asistente Generativo de Primeros Auxilios (Perú)

Proyecto académico de un asistente conversacional de primeros auxilios que corre **100% local**, en un Mac (Apple Silicon), sin depender de la nube.

El contenido se basa en la Cartilla Educativa de Atención de Primeros Auxilios del MINSA (126 preguntas y respuestas) y en diez ilustraciones educativas generadas previamente.

> Es una prueba de concepto educativa. No sustituye la evaluación de personal sanitario ni los servicios de emergencia. Ante una emergencia real en Perú, comunicarse con el 106 (SAMU) o el 116 (bomberos).

## Dos formas de chatear, en dos webs separadas

| | Carpeta | Puerto | Cómo responde |
|---|---|---|---|
| **Chat RAG** | `archivos_generales/` | `8765` | Busca la respuesta más parecida en el dataset del MINSA y se la pasa como contexto a `qwen3.5:9b` vía **Ollama**. No requiere entrenamiento, funciona de inmediato. |
| **Chat con modelo entrenado** | `web_modelo_entrenado/` | `8766` | Usa `Qwen2.5-7B-Instruct` con un **adaptador LoRA entrenado localmente** (fine-tuning real sobre el dataset), cargado directamente con **MLX** (framework de Apple para Apple Silicon), sin pasar por Ollama. |

Ambas webs muestran automáticamente la ilustración fija correspondiente (de las 10 disponibles en `imagenes/`) cuando la pregunta coincide con uno de los temas cubiertos — esto ocurre sin ningún paso adicional, apenas responde el chat.

### Generar una imagen nueva con IA (opcional, solo en el chat LoRA)

Además de la ilustración fija, el chat en `web_modelo_entrenado/` (puerto 8766) tiene un **LoRA de imagen entrenado localmente** (SDXL-Turbo + fine-tuning sobre las 10 ilustraciones). Esta generación **no ocurre automáticamente** porque es lenta (varios minutos por imagen en Apple Silicon sin CUDA):

> Debajo de cada ilustración fija aparece un botón **"🎨 Generar con IA (LoRA)"**. Hay que darle click explícitamente para que el modelo genere una imagen nueva desde cero con ese adaptador — no se genera sola ni junto con la respuesta de texto.

### Cómo se vincula el texto con la imagen (importante: no es IA en este paso)

La elección de qué imagen mostrar —tanto la fija como la que se genera al hacer click— **no la decide ningún modelo de lenguaje ni de visión**. Es un mecanismo de dos pasos, ambos por coincidencia de texto plano:

1. **Detección del tema** (`find_images()` en `server.py`): compara la pregunta y la respuesta generada contra una tabla fija de 10 reglas (`IMAGE_RULES`), cada una con una lista de palabras clave (ej. `"quemadura"`, `"quemad"` → `04_enfriar_quemadura.png`). Es un `if palabra_clave in texto`, no una red neuronal.
2. **Elección del prompt de generación** (solo al presionar "Generar con IA"): el nombre de archivo detectado en el paso 1 se usa como clave de un segundo diccionario fijo, `IMAGE_PROMPTS` (también en `server.py`), que devuelve el mismo texto en inglés con el que esa imagen fue entrenada (ej. `"PAUX_STYLE educational first-aid illustration of a responder cooling a burn..."`). Ese prompt es siempre idéntico para el mismo tema — no se redacta a partir de los detalles específicos de tu pregunta.

En resumen: el LoRA de texto nunca "le dice" al LoRA de imagen qué generar. Ambos están conectados por un diccionario de correspondencia escrito a mano con 10 entradas, no por aprendizaje conjunto.

## Estructura

```text
ProyectoIA_Generativa_PrimerosAuxilios/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── imagenes/                           10 ilustraciones fijas (RCP, atragantamiento, etc.)
├── dataset/
│   └── dataset_primeros_auxilios.csv   126 pares pregunta-respuesta (MINSA)
├── pdfs/
│   └── Cartilla_educativa_de_primeros_auxilio.pdf
├── archivos_generales/                 Chat RAG (puerto 8765)
│   ├── server.py
│   ├── index.html
│   └── dataset_primeros_auxilios.csv
├── web_modelo_entrenado/               Chat con modelo LoRA entrenado (puerto 8766)
│   ├── server.py
│   ├── index.html
│   └── imagenes_generadas/             Imágenes creadas al presionar "Generar con IA"
└── entrenamiento/                      Scripts, configuración y adaptadores del fine-tuning
    ├── setup_entorno.sh                Instala Python 3.12 + mlx-lm
    ├── preparar_dataset_mlx.py         Convierte el CSV a formato de chat JSONL
    ├── entrenar_texto.sh               Fine-tuning LoRA de Qwen2.5-7B-Instruct
    ├── probar_modelo_texto.py          Prueba rápida por consola del modelo entrenado
    ├── setup_entorno_imagenes.sh       Instala PyTorch + diffusers (backend MPS)
    ├── preparar_dataset_imagenes.py    Prepara las 10 imágenes + captions
    ├── entrenar_imagenes.sh            Fine-tuning LoRA de SDXL-Turbo
    ├── generar_imagen.py               Prueba de generación por consola, sin la web
    ├── adaptador_lora_texto/           Adaptador LoRA de texto ya entrenado
    └── adaptador_lora_imagenes/        Adaptador LoRA de imagen ya entrenado
```

## Arquitectura técnica

| | Pipeline de texto | Pipeline de imagen |
|---|---|---|
| **Entorno** | Mac Mini M4, 16 GB RAM (GPU Apple Silicon vía Metal) | Mac Mini M4, 16 GB RAM (GPU Apple Silicon vía Metal, sin CUDA) |
| **Modelo base** | `Qwen2.5-7B-Instruct` (4-bit, `mlx-community`) | `stabilityai/sdxl-turbo` |
| **Método** | Fine-tuning LoRA con **MLX-LM** | Fine-tuning LoRA con **Diffusers** (backend MPS) |
| **Dataset** | 126 pares pregunta-respuesta (Cartilla MINSA) → 114 train / 12 validación | 10 ilustraciones con captions (`PAUX_STYLE`) |
| **Evaluación** | Loss de entrenamiento/validación (equivalente a perplejidad) | — (dataset demasiado pequeño para evaluación cuantitativa fiable) |

## Parámetros del fine-tuning de texto

- Modelo base: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- Parámetros entrenables: ~0.076% del modelo (5.77M de 7,615M) — típico de LoRA
- Capas ajustadas: 8 (las últimas)
- Iteraciones: 300
- Batch size: 2
- Memoria pico observada: ~6.8 GB

## Parámetros del fine-tuning de imagen

- Modelo base: `stabilityai/sdxl-turbo`
- Resolución: 384 × 384
- Rango LoRA: 4
- Batch: 1, acumulación de gradiente: 2
- Pasos: 100 (20 épocas sobre las 10 imágenes)
- Backend: PyTorch + MPS (Metal), sin CUDA
- Tiempo real de entrenamiento: ~9.5 horas en un Mac Mini M4 (más lento de lo esperado; MPS no está tan optimizado como CUDA para este tipo de carga)
- Loss final: 0.0546 (bajó desde 0.343 al inicio)
- Generación de una imagen nueva con el adaptador entrenado: ~5 minutos por imagen (el decode del VAE se ejecuta en CPU como workaround a un cuelgue conocido de PyTorch+MPS con SDXL)

**Aviso importante:** con solo 10 imágenes de entrenamiento, el LoRA visual aprende principalmente el estilo y la composición general, no maniobras médicamente precisas. Las imágenes generadas con el botón "Generar con IA" tienen ruido y artefactos visibles — es un ejercicio académico de demostración del pipeline completo, no una fuente confiable de material educativo. Las 10 ilustraciones fijas que usa el chat por defecto no dependen de este LoRA.

## Dataset visual

Las diez imágenes cubren RCP, atragantamiento, hemorragia, quemadura, fractura, esguince, posición de recuperación, convulsión, sangrado nasal y cabestrillo. Comparten el activador `PAUX_STYLE` y una estética educativa consistente.

Son imágenes sintéticas creadas para demostrar el entrenamiento. No constituyen referencias médicas certificadas y deben revisarse o reemplazarse antes de un uso real.

Consulta `QUICKSTART.md` para las instrucciones paso a paso de ejecución.
