"""
Step 3: Inference (classify new sentences).

Usage (interactive):
    python src/predict.py

Usage (single sentence):
    python src/predict.py --sentence "The Committee raised rates by 50 basis points."

Usage (batch from CSV):
    python src/predict.py --file my_sentences.csv --col sentence_column_name
"""

import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR  = "outputs/best_model"
LABEL_EMOJI = {"hawkish": "🦅", "dovish": "🕊️", "neutral": "⚖️"}
LABEL_COLOR = {
    "hawkish": "\033[91m",   # red
    "dovish":  "\033[94m",   # blue
    "neutral": "\033[93m",   # yellow
}
RESET = "\033[0m"


def load_model(model_dir: str):
    """Load the fine-tuned model and tokenizer from disk."""
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def classify(sentence: str, tokenizer, model) -> dict:
    """
    Classify a single sentence.
    Returns a dict with label, confidence, and all class probabilities.
    """
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    # model.config.id2label maps index → label name
    id2label = model.config.id2label
    label_probs = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    pred_label  = max(label_probs, key=label_probs.get)
    confidence  = label_probs[pred_label]

    return {
        "sentence":   sentence,
        "label":      pred_label,
        "confidence": confidence,
        "probabilities": label_probs,
    }


def print_result(result: dict):
    label = result["label"]
    emoji = LABEL_EMOJI.get(label, "")
    color = LABEL_COLOR.get(label, "")

    print(f"\n  Sentence:   {result['sentence'][:90]}{'...' if len(result['sentence']) > 90 else ''}")
    print(f"  Prediction: {color}{label.upper()}{RESET} {emoji}  (confidence: {result['confidence']:.1%})")
    print(f"  Hawkish:  {result['probabilities'].get('hawkish', 0):.1%}  |  "
          f"Dovish: {result['probabilities'].get('dovish', 0):.1%}  |  "
          f"Neutral: {result['probabilities'].get('neutral', 0):.1%}")


def interactive_mode(tokenizer, model):
    """Run an interactive REPL for sentence classification."""
    print("\n" + "=" * 60)
    print("Central Bank Sentiment Classifier: Interactive Mode")
    print("Type a sentence from an FOMC/RBA document to classify it.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    examples = [
        "The Committee raised the target range for the federal funds rate by 75 basis points.",
        "Inflation remains well above the Committee's longer-run goal of 2 percent.",
        "The Board decided to hold the cash rate unchanged at this meeting.",
        "The Committee will be prepared to adjust the stance of monetary policy as appropriate.",
        "Downside risks to the economic outlook have increased materially.",
    ]
    print("\nExample sentences to try:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex}")

    while True:
        print()
        sentence = input("Enter sentence (or 'quit'): ").strip()
        if sentence.lower() in ("quit", "exit", "q"):
            break
        if not sentence:
            continue
        result = classify(sentence, tokenizer, model)
        print_result(result)


def batch_mode(file_path: str, col: str, tokenizer, model):
    """Classify all sentences in a CSV column and save results."""
    df = pd.read_csv(file_path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    results = []
    for _, row in df.iterrows():
        result = classify(str(row[col]), tokenizer, model)
        results.append(result)

    out_df = pd.DataFrame(results)
    out_path = file_path.replace(".csv", "_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
    print(out_df[["sentence", "label", "confidence"]].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Classify policy sentences as Hawkish/Dovish/Neutral")
    parser.add_argument("--sentence", type=str, help="Single sentence to classify")
    parser.add_argument("--file",     type=str, help="CSV file with sentences to batch-classify")
    parser.add_argument("--col",      type=str, default="sentence", help="Column name in CSV")
    parser.add_argument("--model",    type=str, default=MODEL_DIR, help="Path to fine-tuned model")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found at '{args.model}'")
        print("Run 'python src/train.py' first to train the model.")
        return

    tokenizer, model = load_model(args.model)

    if args.sentence:
        result = classify(args.sentence, tokenizer, model)
        print_result(result)

    elif args.file:
        batch_mode(args.file, args.col, tokenizer, model)

    else:
        interactive_mode(tokenizer, model)


if __name__ == "__main__":
    main()
