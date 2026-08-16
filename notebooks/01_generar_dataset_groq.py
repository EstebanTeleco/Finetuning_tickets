# genera el dataset sintetico de tickets de soporte (5 clases)
# para correr en Colab, celda por celda (separadas con # %%)
#
# VERSION CON GROQ (gratis) en vez de Anthropic
# conseguir API key gratis en: https://console.groq.com -> API Keys

# %% instalar dependencias
# !pip install -q groq

# %% config
import os
import json
import random
import time
from getpass import getpass

from groq import Groq

os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")
client = Groq()

MODEL = "llama-3.3-70b-versatile"  # modelo grande y gratis en Groq para generar los datos,
                                     # el que despues fine-tuneamos es el chico (Llama 3.2 3B)

CATEGORIAS = {
    "facturacion_pagos": "cobros, facturas, metodos de pago, reembolsos, cambios de plan/precio",
    "problema_tecnico": "bugs, errores del sistema, fallos, caidas, comportamiento inesperado del producto",
    "cuenta_acceso": "login, contrasenas, verificacion en dos pasos, permisos de usuario, recuperacion de cuenta",
    "consulta_producto": "dudas sobre funcionalidades, como usar algo, comparacion de planes, integraciones",
    "cancelacion_baja": "cancelar suscripcion, dar de baja, pausar cuenta, downgrade de plan",
}

TONOS = [
    "cliente neutral y formal",
    "cliente frustrado o molesto",
    "cliente muy breve e informal, con errores de tipeo",
    "cliente educado que da mucho contexto de mas",
    "cliente urgente que pide respuesta inmediata",
    "cliente confundido que no sabe bien como explicar el problema",
]

LONGITUDES = ["una oracion corta", "2-3 oraciones", "un parrafo con varios detalles"]

N_POR_LOTE = 8
LOTES_POR_CATEGORIA = 15  # 15 x 8 = ~120 por categoria, se puede subir si hace falta mas volumen

# %% cargar las semillas escritas a mano (le dan el tono real al resto)
def cargar_semillas(path="seed_examples.jsonl"):
    semillas = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ej = json.loads(line)
            semillas.setdefault(ej["label"], []).append(ej["text"])
    return semillas

# subir seed_examples.jsonl a Colab antes de correr esto
SEMILLAS = cargar_semillas()

# %% pide un lote de tickets al modelo (via Groq)
def generar_lote(categoria, descripcion, n=N_POR_LOTE):
    tono = random.choice(TONOS)
    longitud = random.choice(LONGITUDES)
    ejemplos_ancla = random.sample(SEMILLAS[categoria], k=min(3, len(SEMILLAS[categoria])))

    prompt = f"""Genera {n} tickets de soporte al cliente unicos y realistas en espanol rioplatense/neutro,
para una empresa de software SaaS. Todos deben pertenecer a la categoria: "{categoria}"
({descripcion}).

Estilo para este lote: {tono}, longitud aproximada: {longitud}.

Ejemplos de referencia del nivel de detalle esperado (no los repitas, son solo guia):
{chr(10).join(f"- {e}" for e in ejemplos_ancla)}

Reglas:
- Cada ticket tiene que sonar como si lo hubiera escrito un cliente real.
- Varia vocabulario, estructura de oraciones y nivel de detalle entre los {n} ejemplos.
- Mezcla algo de tipeo descuidado con ejemplos bien escritos.
- No repitas el mismo producto/feature en todos los ejemplos del lote.
- Responde solo con un array JSON de strings, nada mas, sin markdown.

Formato: ["ticket 1", "ticket 2", ...]"""

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = resp.choices[0].message.content.strip()
    texto = texto.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        tickets = json.loads(texto)
        return [t.strip() for t in tickets if isinstance(t, str) and len(t.strip()) > 5]
    except json.JSONDecodeError:
        print(f"  no se pudo parsear un lote de {categoria}, lo descarto")
        return []

# %% loop principal de generacion
dataset = []
vistos = set()

for categoria, descripcion in CATEGORIAS.items():
    print(f"generando {categoria}")
    for lote in range(LOTES_POR_CATEGORIA):
        tickets = generar_lote(categoria, descripcion)
        nuevos = 0
        for t in tickets:
            clave = t.lower().strip()
            if clave not in vistos:
                vistos.add(clave)
                dataset.append({"text": t, "label": categoria})
                nuevos += 1
        print(f"  lote {lote + 1}/{LOTES_POR_CATEGORIA}: +{nuevos} nuevos "
              f"(llevo {sum(1 for d in dataset if d['label'] == categoria)} de esta categoria)")
        time.sleep(2.5)  # Groq free tier tiene rate limit por minuto, hay que ir mas despacio que con Anthropic

print(f"\ntotal generado: {len(dataset)}")

# %% sumar las semillas al dataset final, ya son buenas
for label, textos in SEMILLAS.items():
    for t in textos:
        dataset.append({"text": t, "label": label})

random.shuffle(dataset)
print(f"total con semillas: {len(dataset)}")
for cat in CATEGORIAS:
    n = sum(1 for d in dataset if d["label"] == cat)
    print(f"  {cat}: {n}")

# %% split train/val/test, estratificado por clase
def split_estratificado(data, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    por_clase = {}
    for ej in data:
        por_clase.setdefault(ej["label"], []).append(ej)

    train, val, test = [], [], []
    for label, ejemplos in por_clase.items():
        random.shuffle(ejemplos)
        n = len(ejemplos)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        val.extend(ejemplos[:n_val])
        test.extend(ejemplos[n_val:n_val + n_test])
        train.extend(ejemplos[n_val + n_test:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test

train, val, test = split_estratificado(dataset)
print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

# %% guardar a disco
def guardar_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for ej in data:
            f.write(json.dumps(ej, ensure_ascii=False) + "\n")

guardar_jsonl(train, "train.jsonl")
guardar_jsonl(val, "val.jsonl")
guardar_jsonl(test, "test.jsonl")

print("listo: train.jsonl, val.jsonl, test.jsonl")

# %% pegarle una revisada rapida antes de seguir
# conviene mirar a mano unos 20-30 al azar y confirmar que las labels tienen sentido
muestra = random.sample(dataset, 15)
for ej in muestra:
    print(f"[{ej['label']}] {ej['text']}\n")
