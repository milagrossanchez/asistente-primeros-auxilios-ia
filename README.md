# Asistente Generativo de Primeros Auxilios (Perú)

Proyecto académico de un asistente conversacional de primeros auxilios que corre **100% local**, en un Mac (Apple Silicon), sin depender de la nube.

El contenido se basa en la Cartilla Educativa de Atención de Primeros Auxilios del MINSA (126 preguntas y respuestas) y en diez ilustraciones educativas generadas previamente.

> Es una prueba de concepto educativa. No sustituye la evaluación de personal sanitario ni los servicios de emergencia. Ante una emergencia real en Perú, comunicarse con el 106 (SAMU) o el 116 (bomberos).

## Dos formas de chatear, en dos webs separadas

| | Carpeta | Puerto | Cómo responde |
|---|---|---|---|
| **Chat RAG** | `archivos_generales/` | `8765` | Busca la respuesta más parecida en el dataset del MINSA y se la pasa como contexto a `qwen3.5:9b` vía **Ollama**. No requiere entrenamiento, funciona de inmediato. |
| **Chat con modelo entrenado** | `web_modelo_entrenado/` | `8766` | Usa `Qwen2.5-7B-Instruct` con un **adaptador LoRA entrenado localmente** (fine-tuning real sobre el dataset), cargado directamente con **MLX** (framework de Apple para Apple Silicon), sin pasar por Ollama. |

Ambas webs muestran la ilustración correspondiente (de las 10 disponibles en `../imagenes/`) cuando la pregunta coincide con uno de los temas cubiertos.

## Estructura

```text
ialocal/
├── imagenes/                              10 ilustraciones (RCP, atragantamiento, etc.)
└── ProyectoIA_Generativa_PrimerosAuxilios/
    ├── README.md
    ├── QUICKSTART.md
    ├── requirements.txt
    ├── dataset/
    │   └── dataset_primeros_auxilios.csv  126 pares pregunta-respuesta (MINSA)
    ├── pdfs/
    │   └── Cartilla_educativa_de_primeros_auxilio.pdf
    ├── archivos_generales/                Chat RAG (puerto 8765)
    │   ├── server.py
    │   ├── index.html
    │   └── dataset_primeros_auxilios.csv
    ├── web_modelo_entrenado/              Chat con modelo LoRA entrenado (puerto 8766)
    │   ├── server.py
    │   └── index.html
    └── entrenamiento/                     Scripts y configuración del fine-tuning
        ├── setup_entorno.sh               Instala Python 3.12 + mlx-lm
        ├── preparar_dataset_mlx.py        Convierte el CSV a formato de chat JSONL
        ├── entrenar_texto.sh              Fine-tuning LoRA de Qwen2.5-7B-Instruct
        ├── probar_modelo_texto.py         Prueba rápida por consola del modelo entrenado
        ├── setup_entorno_imagenes.sh      Instala PyTorch + diffusers (backend MPS)
        ├── preparar_dataset_imagenes.py   Prepara las 10 imágenes + captions
        ├── entrenar_imagenes.sh           Fine-tuning LoRA de SDXL-Turbo
        ├── adaptador_lora_texto/          Adaptador LoRA de texto ya entrenado
        └── venv/                          Entorno virtual (Python 3.12)
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
- Pasos: 100
- Backend: PyTorch + MPS (Metal), sin CUDA

**Aviso importante:** con solo 10 imágenes de entrenamiento, el LoRA visual aprende principalmente el estilo, no maniobras nuevas. Es un ejercicio académico — las 10 ilustraciones ya generadas y usadas en el chat no dependen de que este entrenamiento termine ni mejore el resultado.

## Dataset visual

Las diez imágenes cubren RCP, atragantamiento, hemorragia, quemadura, fractura, esguince, posición de recuperación, convulsión, sangrado nasal y cabestrillo. Comparten el activador `PAUX_STYLE` y una estética educativa consistente.

Son imágenes sintéticas creadas para demostrar el entrenamiento. No constituyen referencias médicas certificadas y deben revisarse o reemplazarse antes de un uso real.

Consulta `QUICKSTART.md` para las instrucciones paso a paso de ejecución.
