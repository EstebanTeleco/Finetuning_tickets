# Fine-tuning de Llama-3.2-3B para clasificación de tickets de soporte

Fine-tuning de Llama-3.2-3B-Instruct con QLoRA para clasificar tickets de soporte al cliente en español en 5 categorías. Todo el proceso corre en Google Colab con la GPU T4 gratuita.

## Por qué este approach

Un RAG resuelve bien preguntas sobre una base de conocimiento, pero para clasificación estructurada (donde importa la latencia, una salida consistente, y poder correr el modelo sin depender de una API externa) tiene más sentido un modelo chico fine-tuneado. Este repo es el flujo completo: generación de datos, fine-tuning con QLoRA, y evaluación contra un baseline clásico para chequear si realmente vale la pena.

## Resultados

| Método | Accuracy | F1 (macro) |
|---|---|---|
| TF-IDF + Regresión Logística | 95.0% | 95.1% |
| Llama-3.2-3B zero-shot (sin fine-tuning) | 53.3% | 48.1% |
| Llama-3.2-3B + QLoRA | 91.7% | 91.5% |

El fine-tuning mejoró bastante sobre el modelo base, pero el baseline de TF-IDF resultó ligeramente superior en este dataset sintético. Esto tiene sentido: al ser generado por un LLM, el dataset probablemente tiene un vocabulario más marcado y predecible por categoría que el que tendría texto real, lo cual favorece directamente a un método como TF-IDF que se apoya en frecuencia de palabras. Con datos reales y más variados, la ventaja de un modelo con mejor comprensión semántica como Llama debería notarse más — queda pendiente validar esto con una muestra de tickets reales.

Gráficos en `resultados/comparacion_metodos.png` y `resultados/matriz_confusion.png`.

## Categorías

- `facturacion_pagos` — cobros, facturas, métodos de pago, reembolsos
- `problema_tecnico` — bugs, errores, fallos del producto
- `cuenta_acceso` — login, contraseñas, verificación, permisos
- `consulta_producto` — dudas sobre funcionalidades, comparación de planes
- `cancelacion_baja` — cancelar suscripción, dar de baja, downgrade

## Stack

Llama-3.2-3B-Instruct como modelo base, QLoRA en 4-bit (nf4) vía `peft` + `bitsandbytes` + `trl`. El dataset (~750 tickets) es sintético, generado con la API de Groq (Llama-3.3-70B, gratis) a partir de 40 ejemplos escritos a mano para anclar el estilo. Evaluación con accuracy, F1 macro, matriz de confusión y comparación contra TF-IDF + regresión logística y contra el modelo base sin fine-tunear.

## Estructura

```
data/
  seed_examples.jsonl     ejemplos semilla escritos a mano
  train.jsonl / val.jsonl / test.jsonl   (se generan con el script 1)
notebooks/
  01_generar_dataset_groq.py   genera el dataset sintético (vía API de Groq)
  02_finetune_qlora.py    fine-tuning con QLoRA
  03_evaluar.py           evaluación y comparación de métodos
resultados/
  comparacion_metodos.png
  matriz_confusion.png
  resultados_evaluacion.json
modelo_finetuneado/        adapter LoRA
```

## Cómo correrlo

1. Abrir `notebooks/01_generar_dataset_groq.py` en Colab, subir `data/seed_examples.jsonl`, correr con una API key de Groq (gratis, se saca en console.groq.com).
2. Correr `notebooks/02_finetune_qlora.py` en Colab con GPU T4 (3 epochs, ~750 ejemplos, tarda unos 20-30 min).
3. Correr `notebooks/03_evaluar.py` para sacar la tabla y los gráficos.

## Algunas decisiones

QLoRA en vez de fine-tuning completo porque con 16GB de VRAM (T4 gratuita) no entra otra cosa, y de paso solo se entrena ~1-2% de los parámetros. Los ejemplos semilla ayudan a que el dataset generado no salga repetitivo, que es lo que pasa si le pedís a un LLM 500 ejemplos de una sin ningún anclaje. Y la comparación contra TF-IDF está porque un modelo de 3B con fine-tuning no siempre le gana a algo mucho más simple — mejor medirlo que asumirlo, y en este caso la duda estaba justificada: TF-IDF terminó ganando.

## Para seguir

- Segunda etiqueta de urgencia (multi-task)
- Comparar contra Phi-3.5-mini y Mistral-7B
- Demo con Gradio en HF Spaces
- Cuantización a GGUF para correr en CPU
