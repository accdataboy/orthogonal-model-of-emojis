# Fine-tune demo with controllable dataset size. No W&B. Single file. Transformers ≥ 4.46.
# Credit to John Smith: https://huggingface.co/datasets/John6666/forum1/blob/main/ft_transformer_small_sentiment_analysis.md
# python -m venv .venv
# source .venv/bin/activate
# pip install torch transformers accelerate peft datasets evaluate numpy
# Docs:
# Transformers seq. class. guide: https://huggingface.co/docs/transformers/en/tasks/sequence_classification
# Trainer API + callbacks: https://huggingface.co/docs/transformers/en/trainer
# - Trainer API: https://huggingface.co/docs/transformers/en/main_classes/trainer
# Load local CSV files: https://huggingface.co/docs/datasets/en/loading
# - Datasets select/shuffle: https://huggingface.co/docs/datasets/en/process
# Evaluate F1 metric: https://huggingface.co/spaces/evaluate-metric/f1
# - Multi-label with problem_type: https://discuss.huggingface.co/t/multilabel-text-classification-trainer-api/11508

import os, random, numpy as np, torch
from typing import Dict, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import evaluate

# --------- config ---------
os.environ["WANDB_DISABLED"] = "true"     # hard-disable W&B
TASK = os.environ.get("TASK", "emotion")  # "sentiment" (unused) or "emotion"
MODEL = os.environ.get("MODEL", "distilbert/distilroberta-base")
TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "0"))  # 0 = full train
EVAL_SAMPLES  = int(os.environ.get("EVAL_SAMPLES",  "0"))  # 0 = full val
BATCH_TRAIN, BATCH_EVAL, EPOCHS, LR, SEED = 64, 64, 10, 1e-5, 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def set_device():
    """
    Set device to CUDA if available, else MPS (Mac), else CPU.

    This defaults to using the best available device (usually).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

DEVICE = set_device()

# --------- load raw dataset ---------
raw = load_dataset("csv", data_files={"train":"data/train.csv","validation":"data/validation.csv","test":"data/test.csv"})
# load_dataset("databoyface/ome-src-v5.2")
raw = raw.class_encode_column("label")
ds_train, ds_val, ds_test = raw["train"], raw["validation"], raw["test"]
label_names = raw["train"].features["label"].names

# --------- optionally limit dataset size ---------
if TRAIN_SAMPLES > 0:
    n = min(TRAIN_SAMPLES, len(ds_train))
    ds_train = ds_train.shuffle(seed=SEED).select(range(n))
if EVAL_SAMPLES > 0:
    n = min(EVAL_SAMPLES, len(ds_val))
    ds_val = ds_val.shuffle(seed=SEED).select(range(n))

id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}
num_labels = len(label_names)

# --------- tokenizer + mapping ---------
tok = AutoTokenizer.from_pretrained(MODEL)

def map_sentiment(batch: Dict[str, Any]):
    x = tok(batch["text"], truncation=True)
    x["labels"] = batch["label"]
    return x

ds_train_tok = ds_train.map(map_sentiment, batched=True)
ds_val_tok   = ds_val.map(map_sentiment,   batched=True)

# --------- model ---------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id
)

model.config.problem_type = "single_label_classification"  

# --------- metrics ---------
acc = evaluate.load("accuracy"); f1 = evaluate.load("f1")
def compute_metrics(eval_pred, thr=0.5):
    logits, labels = eval_pred

    preds = logits.argmax(-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# --------- training args (no W&B) ---------
args = TrainingArguments(
    output_dir=f"distilroberta-base-omemoji",
    eval_strategy="epoch",          # Transformers ≥ 4.46
    save_strategy="epoch",
    report_to="none",               # disable external loggers
    load_best_model_at_end=True,
    metric_for_best_model=("f1_macro"),
    greater_is_better=True,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=500,
    seed=SEED,
)

# --------- trainer (processing_class only) ---------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train_tok,
    eval_dataset=ds_val_tok,
    processing_class=tok,  # replaces tokenizer
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# --------- quick inference ---------
from transformers import pipeline

model.eval()
pipe = pipeline("text-classification", model=trainer.model, tokenizer=tok, device=DEVICE,top_k=1)
for t in ["I am so happy.", "She sounds nervous", "I regret what I said and feel sad", "I love this.", "This is awful.", "Meh, it's fine."]:
    print(t, "->", pipe(t))

