# fine-tuning de Llama-3.2-3B-Instruct con QLoRA para clasificar
# tickets de soporte en 5 categorias. corre en Colab con GPU T4 (16GB).

# %% instalar dependencias
# !pip install -q -U transformers peft bitsandbytes accelerate trl datasets

# %% imports y config
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # hay que aceptar la licencia en HF antes
# alternativa sin gate: "microsoft/Phi-3.5-mini-instruct"

CATEGORIAS = [
    "facturacion_pagos",
    "problema_tecnico",
    "cuenta_acceso",
    "consulta_producto",
    "cancelacion_baja",
]

# %% cargar el dataset del paso anterior
def cargar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

train_data = cargar_jsonl("train.jsonl")
val_data = cargar_jsonl("val.jsonl")

# %% pasar cada ejemplo al formato de chat que espera el modelo
SYSTEM_PROMPT = (
    "Sos un clasificador de tickets de soporte al cliente. "
    "Dado el texto de un ticket, respondé ÚNICAMENTE con una de estas "
    f"categorías, sin explicación adicional: {', '.join(CATEGORIAS)}."
)

def a_chat(ejemplo):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ejemplo["text"]},
            {"role": "assistant", "content": ejemplo["label"]},
        ]
    }

train_ds = Dataset.from_list([a_chat(e) for e in train_data])
val_ds = Dataset.from_list([a_chat(e) for e in val_data])

# %% cargar el modelo en 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# %% config de LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# deberia dar algo asi como 1-2% de los parametros totales

# %% argumentos de entrenamiento
training_args = SFTConfig(
    output_dir="./resultados_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=True,
    report_to="none",
    max_seq_length=512,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

# %% entrenar
trainer.train()

# %% guardar el adapter (pesa unos pocos MB, no el modelo entero)
trainer.save_model("./modelo_finetuneado")
tokenizer.save_pretrained("./modelo_finetuneado")
print("adapter guardado en ./modelo_finetuneado")

# %% subir a HF Hub (opcional)
# from huggingface_hub import login
# login()
# model.push_to_hub("tu-usuario/llama3.2-3b-tickets-soporte-lora")
# tokenizer.push_to_hub("tu-usuario/llama3.2-3b-tickets-soporte-lora")
