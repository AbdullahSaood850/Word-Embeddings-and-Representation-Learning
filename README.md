 Word Embeddings and Representation Learning
---

## Overview

This project explores three generations of word embedding techniques applied to **4-class Twitter sentiment classification** (Positive, Negative, Neutral, Irrelevant).

| Module | Technique | Classifier |
|---|---|---|
| Module 1 | Bag-of-Words (BoW) & TF-IDF | Logistic Regression |
| Module 2 | Word2Vec (CBOW) | Feedforward Neural Network |
| Module 3 | BERT (bert-base-uncased) | Logistic Regression |

---

## Dataset

**Twitter Entity Sentiment Analysis** — [Kaggle (jp797498e)](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

- File: `twitter_training.csv`
- Columns: `tweet_id`, `entity`, `sentiment`, `tweet_text`
- Labels: `Positive`, `Negative`, `Neutral`, `Irrelevant`

---

## Project Structure

```
NLP_A3_23F0014.bin          # Jupyter notebook (JSON format)
README.md                   # This file
NLP_A3_Report.docx          # Full project report
```

---

## Setup & Requirements

### Install Dependencies

```bash
pip install gensim transformers torch tensorflow scikit-learn nltk pandas numpy matplotlib
```

### NLTK Downloads (run once)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run on Kaggle

1. Upload the notebook to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Add the dataset: **Twitter Entity Sentiment Analysis** (jp797498e)
3. Enable GPU accelerator for Module 3 (BERT)
4. Run all cells in order

---

## Module Breakdown

### Module 1 — Frequency-Based Embeddings

**Steps:**
1. Preprocess tweets — lowercase, tokenize, remove stopwords
2. Vectorize with `CountVectorizer` (BoW) and `TfidfVectorizer` (TF-IDF), `max_features=5000`
3. Train Logistic Regression on each
4. Evaluate: Accuracy, Precision, Recall, F1

**Limitation:** No understanding of word meaning or context. "I am NOT happy" looks similar to "I am happy".

---

### Module 2 — Prediction-Based Embeddings (Word2Vec)

**Steps:**
1. Train Word2Vec CBOW model on cleaned tweets (`vector_size=100`, `window=5`, `epochs=10`)
2. Convert each tweet to a 100-dimensional vector by averaging word vectors
3. Train a Feedforward Neural Network (64-unit hidden layer + Dropout 0.3)
4. PCA visualization of word vectors in 2D

**Improvement:** Semantically similar words cluster together. Dense 100D vectors instead of sparse 5000D.

**Limitation:** One fixed vector per word — cannot handle polysemy (e.g., "bank" = money vs river).

---

### Module 3 — Contextualized Embeddings (BERT)

**Steps:**
1. Load `bert-base-uncased` from HuggingFace
2. Extract [CLS] token embeddings (768D) for up to 2,000 tweets
3. Train Logistic Regression on BERT embeddings
4. Polysemy check: compare BERT embeddings for "bank" in two different contexts

**Improvement:** Context-aware embeddings, polysemy handling, bidirectional attention.

**Note:** Only 2,000 samples used due to computational constraints on Kaggle.

---

## Results Summary

| Technique | Accuracy | F1 Score |
|---|---|---|
| BoW + Logistic Regression | ~0.72 | ~0.71 |
| TF-IDF + Logistic Regression | ~0.74 | ~0.73 |
| Word2Vec + FNN | ~0.73 | ~0.72 |
| **BERT + Logistic Regression** | **~0.82** | **~0.81** |

> Note: Exact values depend on your run. BERT is consistently the best performer.

---

## Key Findings

- **BERT** achieves the highest accuracy by leveraging bidirectional context and large-scale pretraining.
- **TF-IDF** outperforms BoW by down-weighting common uninformative words.
- **Word2Vec** captures semantic similarity but loses sentence structure through averaging.
- BERT is the **only method** that correctly handles polysemy — the word "bank" gets different vectors in financial vs geographic contexts.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `scikit-learn` | Vectorizers, Logistic Regression, metrics |
| `gensim` | Word2Vec training |
| `tensorflow / keras` | Feedforward Neural Network |
| `transformers` | BERT tokenizer and model |
| `torch` | BERT inference |
| `nltk` | Tokenization, stopwords |
| `pandas / numpy` | Data handling |
| `matplotlib` | Plotting |
| `scipy` | Cosine similarity |

---

## References

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR.
- HuggingFace Transformers — https://huggingface.co/transformers/
- Gensim Word2Vec — https://radimrehurek.com/gensim/
- Scikit-learn — https://scikit-learn.org/
- Dataset — https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
