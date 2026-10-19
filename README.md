# 🤬 Toxic Comment Detector using DistilBERT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-FF5500?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

An end-to-end deep learning system for automated toxicity detection in short text comments, built with a fine-tuned **DistilBERT** model, PyTorch, Hugging Face Transformers, and Gradio. Developed and trained end-to-end in Google Colab (`Toxicity_Detector.ipynb`).

---

## 📌 Project Overview

Online platforms face an ongoing challenge with toxic, abusive, and hateful comments. Manual moderation cannot scale to millions of daily posts.

This project implements an automated binary toxicity classifier trained on the **Jigsaw Toxic Comment Classification** dataset. By fine-tuning `distilbert-base-uncased`, the model leverages multi-head bidirectional self-attention to understand context and detect subtle offensive language efficiently.

---

## 📊 Dataset

Loaded directly from the Hugging Face Hub: [`Heliosoph/Jigsaw-Toxic-Comments`](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments) (`train.csv.gz`), which is itself derived from the original **Jigsaw Toxic Comment Classification Challenge** dataset (English Wikipedia talk page edits, released by Google/Jigsaw).

- **Sample used:** 50% of the dataset (`df.sample(frac=0.5, random_state=42)`) for efficient training on GPU
- **Split:** 80% train / 20% test (`train_test_split(test_size=0.2, seed=42)`)
- **Label:** Binary — `1` if `toxic == 1`, else `0`
- **Primary metric:** F1-score on the positive (toxic) class, due to class imbalance

---

## 🏗️ Technical Architecture & Pipeline

```
[ Raw Text ] → [ Cleaning (nettoyer) ] → [ DistilBERT Tokenizer (WordPiece, 128 tokens) ]
                                                              │
[ Prediction Output ] ← [ Classification Head ] ← [ DistilBERT Encoder ([CLS] Vector) ]
```

1. **Cleaning:** Lowercased text, URL removal, strips everything except letters, digits, `!`, and `?` (keeps emotional punctuation).
2. **Tokenization:** WordPiece tokenizer (`distilbert-base-uncased`), padded/truncated to a max sequence length of 128 tokens.
3. **Model:** `distilbert-base-uncased` (6 Transformer blocks, 66M parameters) with a binary classification head (`num_labels=2`).
4. **Training:** Fine-tuned end-to-end with Hugging Face `Trainer`.

---

## 📈 Performance & Results

Fine-tuned for **3 epochs**, batch size 32, FP16 mixed precision, on a Colab **NVIDIA T4 GPU**.

### Key Metrics (measured on the held-out test set, 15,958 examples)

| Metric | Value |
| :--- | :---: |
| **F1-Score (Toxic Class)** | **81.2%** |
| **Global Accuracy** | **96.0%** |
| **Precision (Toxic Class)** | **82.0%** |
| **Recall (Toxic Class)** | **81.0%** |
| **F1-Score (Non-Toxic Class)** | **98.0%** |

### Confusion Matrix (Test Set)

|  | Predicted Non-Toxic | Predicted Toxic |
| :--- | :---: | :---: |
| **Actual Non-Toxic** | 14,163 (TN) | 275 (FP) |
| **Actual Toxic** | 293 (FN) | 1,227 (TP) |

- ✅ Toxic comments correctly caught: **1,227**
- ⚠️ Toxic comments missed: **293**
- 🚨 False alarms: **275**

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository

```bash
git https://github.com/EttaoussiMustapha/Toxic-Comment-Detector-Based-on-DistilBERT-Model.git
cd toxic-comment-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install transformers torch pandas scikit-learn gradio matplotlib datasets
```

### 3. Run It

The full pipeline (data loading → training → evaluation → Gradio demo) lives in **`Toxicity_Detector.ipynb`**. Open it in Jupyter, Colab, or VS Code and run the cells top to bottom. It was originally built and trained in **Google Colab**, so a couple of cells (installing `huggingface_hub`, mounting Google Drive to save/reload the model) are Colab-specific — see notes below if running locally.

---

## 💻 Code Overview

The notebook is organized into numbered sections:

### 1. Data Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "Heliosoph/Jigsaw-Toxic-Comments",
    data_files="train.csv.gz",
    split="train"
)

df = dataset.to_pandas()
print("Dataset dimensions:", df.shape)
```

### 2. Cleaning

```python
import re

def nettoyer(texte):
    texte = str(texte).lower()
    texte = re.sub(r"http\S+", "", texte)
    texte = re.sub(r"[^a-zA-Z0-9\s!?]", "", texte)
    return texte.strip()

df["texte_propre"] = df["comment_text"].apply(nettoyer)
df["label"] = (df["toxic"] == 1).astype(int)
df = df[["texte_propre", "label"]].dropna()
```

### 3. Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokeniser(batch):
    return tokenizer(
        batch["texte_propre"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
```

### 4. Dataset Split

```python
from datasets import Dataset

df_sample = df.sample(frac=0.5, random_state=42)

dataset = Dataset.from_pandas(df_sample)
dataset = dataset.map(tokeniser, batched=True)

split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]   # 80% — training
test_dataset  = split["test"]    # 20% — final evaluation
```

### 5. Model & Training

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

training_args = TrainingArguments(
    output_dir                  = "./results",
    num_train_epochs            = 3,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    eval_strategy               = "no",
    save_strategy               = "epoch",
    load_best_model_at_end      = False,
    fp16                        = True,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset
)

trainer.train()
```

### 6. Evaluation

```python
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

f1 = f1_score(y_true, y_pred, average="binary")
print(f"F1-Score (toxic): {f1*100:.1f}%")

print(classification_report(y_true, y_pred, target_names=["Non toxic", "Toxic"]))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Non toxic", "Toxic"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

### 7. Inference

```python
def predire(texte):
    tokens = tokenizer(
        nettoyer(texte),
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        proba_toxique = model(**tokens).logits.softmax(dim=1)[0][1].item()

    if proba_toxique >= 0.5:
        label = "🔴 Toxic"
        confiance = proba_toxique
    else:
        label = "🟢 Non-toxic"
        confiance = 1 - proba_toxique

    return f"{label} — confidence: {confiance*100:.0f}%"
```

### 8. Gradio Demo

```python
import gradio as gr

interface = gr.Interface(
    fn          = predire,
    inputs      = gr.Textbox(placeholder="Enter a comment..."),
    outputs     = "text",
    title       = "🔍 Toxicity Detector",
    description = "Analyzes whether a comment is toxic or not."
)

interface.launch(share=True)
```

> **Note on saving/reloading the model:** the notebook saves the fine-tuned model to Google Drive (`model.save_pretrained("/content/drive/MyDrive/toxicity_model")`) so it doesn't need to be retrained every session. If you're running outside Colab, swap this for a local path or push the model to the Hugging Face Hub instead.

---

## 🛡️ Future Enhancements

- [ ] Fine-tune on 100% of the dataset to target an even higher F1-score.
- [ ] Implement class-weighted loss (`CrossEntropyLoss`) to penalize minority class misclassifications.
- [ ] Support multi-label classification across all 6 Jigsaw categories (threat, insult, obscenity, etc.).
- [ ] Switch to XLM-RoBERTa for multilingual toxicity detection.
- [ ] Add model interpretability with SHAP / LIME.
- [ ] Push the trained model to the Hugging Face Hub for easier sharing/reloading outside Colab.

---

## 📜 License

This project utilizes data from the Jigsaw Toxic Comment Classification Challenge, released under the Creative Commons CC0 (Public Domain) license.
