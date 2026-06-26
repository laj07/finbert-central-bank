"""
Step 2: Fine-tune FinBERT on the labeled FOMC/RBA dataset.

What this script does, in plain English:
  1. Loads labeled_sentences.csv
  2. Splits into 80% train / 20% test (stratified so equal class ratios in both)
  3. Tokenizes each sentence using FinBERT's tokenizer
  4. Fine-tunes the pre-trained FinBERT model with a 3-class head on top
  5. Saves the best model checkpoint to outputs/best_model/

FinBERT (ProsusAI/finbert) is a BERT model made by Google in 2018 pre-trained 
on ~5M financial news articles, wikipedia + a huge book corpus, 
it read the internet and learned how english sentences work 
in terms of financial concepts. 

We're adding a classification head and training it to distinguish 
hawkish / neutral / dovish monetary policy language.

What happens in one epoch:
(An epoch = one full pass through all 160 training sentences.)
    - Feed a sentence in
    - Model outputs three scores
    - Compare to the correct label amd calculate how wrong it was (this is the loss)
    - Nudge the weights slightly in the direction that reduces that wrongness (this is backpropagation)
    - Repeat for next sentence

Run:
    python src/train.py
"""

import os
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import random


# Config 
MODEL_NAME  = "ProsusAI/finbert"   # pre-trained financial BERT
DATA_PATH   = "data/processed/labeled_sentences.csv"
OUTPUT_DIR  = "outputs/best_model"
RESULTS_DIR = "outputs"

LABEL2ID = {"hawkish": 0, "dovish": 1, "neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH  = 128    # max tokens per sentence (most policy sentences < 60 tokens)
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-5   # standard for BERT fine-tuning
RANDOM_SEED = 42
PATIENCE = 3        # stop if val_acc doesn't improve for 3 epochs
MIN_DELTA = 0.01    # improvement must be at least 1% to count

# Dataset class 

class PolicySentenceDataset(Dataset):
    """
    Wraps our labeled CSV into a PyTorch Dataset.
    Each item returns tokenized input IDs + the numeric label.
    """
    def __init__(self, sentences, labels, tokenizer):
        self.encodings = tokenizer(
            sentences,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            [LABEL2ID[l] for l in labels], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


# Training loop 

def train_epoch(model, loader, optimizer, device):
    """One full pass through the training data."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()

        # Gradient clipping prevents exploding gradients with BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    """Evaluate on val/test set. Returns loss, accuracy, all preds and labels."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


# Main 

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    print("=" * 60)
    print("FinBERT Central Bank Sentiment Classifier (Fine-tuning)")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} labeled sentences")
    print(df["label"].value_counts().to_string())

    # 2. Train/test split (stratified keeps class balance in both splits)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

    # 3. Load tokenizer and model from HuggingFace
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # FinBERT has a different head; we replace it
    )

    # 4. Build datasets and dataloaders
    train_ds = PolicySentenceDataset(
        train_df["sentence"].tolist(), train_df["label"].tolist(), tokenizer
    )
    test_ds = PolicySentenceDataset(
        test_df["sentence"].tolist(), test_df["label"].tolist(), tokenizer
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 5. Optimizer and device
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"Training on: {device}")
    print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LR}\n")

    # 6. Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, _, _ = eval_epoch(model, test_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 3),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 3),
        })

        # Save best model
        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  New best model saved (val_acc={val_acc:.3f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 7. Final evaluation with full classification report
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    _, _, all_preds, all_labels = eval_epoch(model, test_loader, device)

    pred_names  = [ID2LABEL[p] for p in all_preds]
    label_names = [ID2LABEL[l] for l in all_labels]

    report = classification_report(label_names, pred_names, digits=3)
    print(report)

    # 8. Save results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results = {
        "model": MODEL_NAME,
        "dataset_size": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "best_val_acc": round(best_val_acc, 3),
        "training_history": history,
        "classification_report": classification_report(
            label_names, pred_names, digits=3, output_dict=True
        ),
    }
    with open(f"{RESULTS_DIR}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/training_results.json")
    print(f"Model saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
