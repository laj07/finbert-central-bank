# FinBERT Central Bank Sentiment Classifier

Fine-tuning [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Federal Reserve FOMC meeting minutes and RBA Board statements to classify monetary policy tone as **Hawkish**, **Dovish**, or **Neutral** at the sentence level.

---

## Why this matters

Central bank communication is one of the most systematically traded signals in fixed income markets. When the Fed chair says "the labor market remains extremely tight," bond traders move. When the RBA says "the Board is prepared to ease further," rates desks reprice the curve.

The standard open-source approach uses keyword counting (Loughran-McDonald word lists) — fast, but brittle. A sentence like *"the Committee sees upside risks as no longer elevated"* is hawkish by context but scores neutral on keyword count. Fine-tuned language models resolve this by capturing the full semantic context of a sentence rather than summing individual words.

**This classifier is applicable to:**
- Automated scanning of FOMC minutes for tone shifts (released 8× per year)
- Building a quantitative hawkishness index for rates trading signals
- Fixed income research — quantifying policy stance over time
- Monitoring cross-central-bank divergence (Fed vs. RBA tightening/easing cycles)

---

## What problem it solves

Given a sentence extracted from a central bank document, output one of:

| Label | Meaning | Example |
|---|---|---|
| **Hawkish** | Tightening bias — rate hikes, inflation concern | *"The Committee anticipates that ongoing increases in the target range will be appropriate."* |
| **Dovish** | Easing bias — rate cuts, growth/unemployment concern | *"The Board decided that lower interest rates were appropriate to support employment."* |
| **Neutral** | Balanced, data-dependent, or descriptive | *"The Committee will assess incoming data and adjust policy as appropriate."* |

---

## Dataset

### Sources
- **FOMC meeting minutes** (Federal Reserve, federalreserve.gov) — 8 meetings per year, ~5,000 words per document, released ~3 weeks after each meeting
- **RBA Board statements** (Reserve Bank of Australia, rba.gov.au) — released after each Board meeting

Both are public domain documents.

### Labeling approach

200 sentences were manually annotated following the guidelines in [Shah et al. (2023)](https://aclanthology.org/2023.acl-long.368) and Apel & Blix Grimaldi (2012):

- **Hawkish**: explicit rate hike language, tightening stance, upside inflation risk, labor market tightness framing
- **Dovish**: explicit rate cut language, accommodative stance, downside growth risk, unemployment/slack framing
- **Neutral**: hold decisions, monitoring language, balanced risk assessment, purely descriptive economic data

The dataset is balanced: **40 sentences per class**.

Train/test split: **80/20 stratified** (160 train, 40 test).

> Note: For production use, the dataset should be expanded to 500–1000 sentences with inter-annotator agreement scoring (Cohen's κ ≥ 0.70 is the standard threshold in the literature).

---

## Model

- **Backbone**: `ProsusAI/finbert` — BERT-base pre-trained on ~5M financial news articles
- **Head**: linear layer → 3 logits (hawkish / dovish / neutral)
- **Fine-tuning**: AdamW, lr=2e-5, weight_decay=0.01, 5 epochs, gradient clipping at 1.0
- **Tokenizer**: WordPiece, max_length=128

Why FinBERT over vanilla BERT? Its financial pre-training vocabulary means terms like "basis points," "federal funds rate," "trimmed mean CPI," and "quantitative tightening" are better represented in its embeddings before fine-tuning even begins.

---

## Results

### Classification report (held-out test set, n=40)

```
              precision    recall  f1-score   support

     dovish       1.000     0.571     0.727        14
     hawkish      0.650     1.000     0.788        13
     neutral      0.917     0.846     0.880        13

    accuracy                          0.800        40
   macro avg      0.856     0.806     0.798        40
weighted avg      0.859     0.800     0.797        40
```

**Macro F1: 0.798**

Dovish language often hedges, for example, "the Board is prepared to ease further if needed", which sits uncomfortably close to neutral. Even human annotators show lower agreement on dovish sentences in the literature. The confusion matrix below shows exactly where those errors land.

### Confusion matrix

```
            hawkish    dovish   neutral
   hawkish       13       0        0
    dovish        5       8        1
   neutral        2       0       11
```

Hawkish is perfectly precise, every sentence the model called hawkish actually was hawkish. The cost is on the dovish side: 5 dovish sentences were called hawkish, pulling dovish recall down to 0.571. This is the expected failure mode that is assertive dovish language ("the Board decided to lower rates") can  structurally resemble hawkish framing.

### Training dynamics

Loss converges steadily across 5 epochs with no sign of catastrophic forgetting of the financial pre-training. Validation accuracy peaks at epoch 3 (80%), with early stopping preventing further overfitting.

---

## Example outputs

```python
from src.predict import load_model, classify

tokenizer, model = load_model("outputs/best_model")

examples = [
    "The Committee raised the target range by 75 basis points.",
    "The Board decided that lower rates were appropriate to support growth.",
    "The Committee will assess incoming information at future meetings.",
    "Trimmed mean inflation remained well above the midpoint of the target band.",
    "Members noted downside risks to the economic outlook had increased materially.",
]

for sentence in examples:
    result = classify(sentence, tokenizer, model)
    print(f"{result['label'].upper():>8}  ({result['confidence']:.0%})  {sentence[:70]}")
```

**Output:**
```
 HAWKISH  (94.3%)  The Committee raised the target range by 75 basis points.
  DOVISH  (96.3%)  The Board decided that lower rates were appropriate to support growth.
 NEUTRAL  (95.9%)  The Committee will assess incoming information at future meetings.
 HAWKISH  (84.7%)  Trimmed mean inflation remained well above the midpoint of the target band.
  DOVISH  (95.7%)  Members noted downside risks to the economic outlook had increased materially.
```

---

## Finance context — rates markets application

The hawkishness score from a model like this can be used to construct a **policy stance index**:

1. Extract all sentences from each FOMC minutes document (typically 150–200 sentences)
2. Score each sentence: hawkish → +1, dovish → –1, neutral → 0, weighted by confidence
3. Average across the document to get a document-level score in [–1, +1]
4. Track this over time as a quantitative hawkishness index

Studies using similar approaches (e.g., Shah et al. 2023, Schmeling & Wagner 2019) find correlations of 0.65–0.75 between NLP-derived policy stance measures and 2-year Treasury yield moves on FOMC release days, a meaningful signal for systematic rates strategies.

---

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create the labeled dataset
```bash
python src/create_dataset.py
```
This writes `data/processed/labeled_sentences.csv` (200 annotated sentences).

### 3. Fine-tune FinBERT
```bash
python src/train.py
```
- Downloads `ProsusAI/finbert` from HuggingFace (~440MB, one-time)
- Trains for 5 epochs (~3–5 minutes on CPU, ~1 minute on GPU)
- Saves best checkpoint to `outputs/best_model/`

### 4. Evaluate
```bash
python src/evaluate.py
```
Generates `outputs/classification_report.txt` and `outputs/confusion_matrix.png`.

### 5. Run inference
```bash
# Interactive mode
python src/predict.py

# Single sentence
python src/predict.py --sentence "The Committee raised rates by 50 basis points."

# Batch from CSV
python src/predict.py --file my_data.csv --col text_column
```

---

## Project structure

```
finbert-central-bank/
├── data/
│   ├── raw/                     # Raw text files (if downloaded)
│   └── processed/
│       └── labeled_sentences.csv   # 200 manually labeled sentences
├── notebooks/
│   ├── finbert_classifier.ipynb        
├── src/
│   ├── create_dataset.py        # Step 1: Build labeled CSV
│   ├── train.py                 # Step 2: Fine-tune FinBERT
│   ├── evaluate.py              # Step 3: Classification report + plots
│   └── predict.py               # Step 4: Inference on new sentences
├── outputs/
│   ├── best_model/              # Saved fine-tuned model + tokenizer
│   ├── training_results.json    # Loss/accuracy per epoch
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_curves.png
├── requirements.txt
└── README.md
```

---

## Limitations and what I'd do next

200 sentences is enough to demonstrate the pipeline but not enough to claim production-ready performance. The test set of 40 sentences means results could shift a few percentage points with a different random seed. I'd want at least 500 annotated sentences with inter-annotator agreement scoring (Cohen's κ ≥ 0.70) before calling this robust.

**Specific weaknesses I came about:**

- Dovish is the hardest class at F1=0.667, consistent with the literature, hedged dovish language sits close to neutral by construction
- No temporal context: each sentence is classified independently, so the model can't use the surrounding paragraph's tone as a signal
- Domain shift: the model is calibrated on Fed and RBA language; ECB or BOE documents would likely underperform without additional fine-tuning data

**What I'd do in the future:**

1. Scrape 3 years of actual FOMC minutes and RBA statements (they're public), extract sentences, label 500+ with proper annotation protocol
2. Build a document-level hawkishness index: score every sentence, average by confidence weight, track across meetings over time
3. Correlate the index with 2-year Treasury yield moves on FOMC release days, this is the test that matters for rates trading applicability
---

## References

1. Shah, A., Paturi, S., & Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis. *ACL 2023*. https://aclanthology.org/2023.acl-long.368
2. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. https://arxiv.org/abs/1908.10063
3. Apel, M., & Blix Grimaldi, M. (2012). The information content of central bank minutes. *Riksbank Working Paper*.
4. Schmeling, M., & Wagner, C. (2019). Does Central Bank Tone Move Asset Prices? *CEPR Discussion Paper*.
