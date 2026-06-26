"""
Step 4: Evaluation (generate classification report and confusion matrix).

Loads the saved best model, runs it on the held-out test set,
and produces:
  - outputs/classification_report.txt    (precision/recall/F1 per class)
  - outputs/confusion_matrix.png         (heatmap visualization)

Run:
    python src/evaluate.py
"""

import json
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script use
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

MODEL_DIR  = "outputs/best_model"
DATA_PATH  = "data/processed/labeled_sentences.csv"
OUTPUT_DIR = "outputs"
RANDOM_SEED = 42
MAX_LENGTH  = 128
BATCH_SIZE  = 16

LABEL2ID = {"hawkish": 0, "dovish": 1, "neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = ["hawkish", "dovish", "neutral"]


class PolicySentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.encodings = tokenizer(
            sentences, max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        self.labels = torch.tensor([LABEL2ID[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title("FinBERT Sentiment Classifier: Confusion Matrix", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data and reconstruct the same test split used during training
    df = pd.read_csv(DATA_PATH)
    _, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"Test set: {len(test_df)} sentences")

    # Load model
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    model.eval()

    # Build test dataset
    test_ds = PolicySentenceDataset(
        test_df["sentence"].tolist(), test_df["label"].tolist(), tokenizer
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Run inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    pred_names  = [ID2LABEL[p] for p in all_preds]
    label_names = [ID2LABEL[l] for l in all_labels]

    # Classification report
    report = classification_report(label_names, pred_names, digits=3)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    report_path = f"{OUTPUT_DIR}/classification_report.txt"
    with open(report_path, "w") as f:
        f.write("FinBERT Central Bank Sentiment Classifier\n")
        f.write("Fine-tuned on FOMC minutes + RBA statements\n\n")
        f.write(report)
    print(f"Report saved → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(label_names, pred_names, labels=CLASS_NAMES)
    plot_confusion_matrix(cm, CLASS_NAMES, f"{OUTPUT_DIR}/confusion_matrix.png")

    # Also print it as text for quick inspection
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':>10}", "  ".join(f"{c:>8}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>10}", "  ".join(f"{v:>8}" for v in row))

    # Update training_results.json with the correct classification report
    results_path = Path(OUTPUT_DIR.replace("best_model", "")) / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        results["classification_report"] = classification_report(
            label_names, pred_names, output_dict=True
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\ntraining_results.json updated with classification report.")


if __name__ == "__main__":
    main()
