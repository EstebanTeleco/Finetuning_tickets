# compara 3 formas de resolver el problema sobre el mismo test set:
#   1. TF-IDF + regresion logistica (baseline clasico)
#   2. Llama-3.2-3B sin fine-tuning (zero-shot)
#   3. Llama-3.2-3B + el adapter LoRA que entrenamos
#
# sirve para ver si el fine-tuning realmente valio la pena o si
# un metodo mas simple ya resolvia esto igual de bien.

# %% imports
import json
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

CATEGORIAS = [
    "facturacion_pagos",
    "problema_tecnico",
    "cuenta_acceso",
    "consulta_producto",
    "cancelacion_baja",
]

def cargar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

train_data = cargar_jsonl("train.jsonl")
test_data = cargar_jsonl("test.jsonl")

X_test_text = [e["text"] for e in test_data]
y_test = [e["label"] for e in test_data]

resultados = {}

# %% baseline: TF-IDF + regresion logistica
X_train_text = [e["text"] for e in train_data]
y_train = [e["label"] for e in train_data]

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)
preds_baseline = clf.predict(X_test_tfidf)

resultados["TF-IDF + LogReg"] = {
    "accuracy": accuracy_score(y_test, preds_baseline),
    "f1_macro": f1_score(y_test, preds_baseline, average="macro"),
    "preds": list(preds_baseline),
}

# %% cargar el modelo base (lo usamos para zero-shot y despues con el adapter)
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
)

SYSTEM_PROMPT = (
    "Sos un clasificador de tickets de soporte al cliente. "
    "Dado el texto de un ticket, respondé ÚNICAMENTE con una de estas "
    f"categorías, sin explicación adicional: {', '.join(CATEGORIAS)}."
)

def clasificar(modelo, texto):
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": texto},
    ]
    prompt = tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)
    with torch.no_grad():
        out = modelo.generate(
            **inputs, max_new_tokens=15, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    respuesta = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()

    # busco la categoria dentro de la respuesta por si el modelo agrega texto de mas
    for cat in CATEGORIAS:
        if cat in respuesta:
            return cat
    return "sin_clasificar"

# %% zero-shot (modelo base, sin fine-tuning)
preds_zero_shot = [clasificar(base_model, t) for t in X_test_text]

resultados["Llama-3.2-3B (zero-shot)"] = {
    "accuracy": accuracy_score(y_test, preds_zero_shot),
    "f1_macro": f1_score(y_test, preds_zero_shot, average="macro", labels=CATEGORIAS),
    "preds": preds_zero_shot,
}

# %% ahora con el adapter LoRA encima
modelo_ft = PeftModel.from_pretrained(base_model, "./modelo_finetuneado")

preds_finetuned = [clasificar(modelo_ft, t) for t in X_test_text]

resultados["Llama-3.2-3B + QLoRA (fine-tuneado)"] = {
    "accuracy": accuracy_score(y_test, preds_finetuned),
    "f1_macro": f1_score(y_test, preds_finetuned, average="macro", labels=CATEGORIAS),
    "preds": preds_finetuned,
}

# %% tabla comparativa
print(f"{'metodo':<35} {'accuracy':<12} {'f1 (macro)':<12}")
print("-" * 60)
for nombre, r in resultados.items():
    print(f"{nombre:<35} {r['accuracy']:<12.3f} {r['f1_macro']:<12.3f}")

# %% reporte por clase del modelo fine-tuneado
print(classification_report(y_test, preds_finetuned, labels=CATEGORIAS))

# %% matriz de confusion
cm = confusion_matrix(y_test, preds_finetuned, labels=CATEGORIAS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CATEGORIAS, yticklabels=CATEGORIAS)
plt.xlabel("prediccion")
plt.ylabel("real")
plt.title("matriz de confusion - modelo fine-tuneado")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150)
plt.show()

# %% grafico comparando los 3 metodos
metodos = list(resultados.keys())
accuracies = [resultados[m]["accuracy"] for m in metodos]
f1s = [resultados[m]["f1_macro"] for m in metodos]

x = np.arange(len(metodos))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, accuracies, width, label="accuracy")
ax.bar(x + width/2, f1s, width, label="f1 (macro)")
ax.set_xticks(x)
ax.set_xticklabels(metodos, rotation=15, ha="right")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("comparacion de metodos - test set")
plt.tight_layout()
plt.savefig("comparacion_metodos.png", dpi=150)
plt.show()

# %% guardar resultados para el README
resumen = {
    nombre: {"accuracy": r["accuracy"], "f1_macro": r["f1_macro"]}
    for nombre, r in resultados.items()
}
with open("resultados_evaluacion.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("guardado en resultados_evaluacion.json")
