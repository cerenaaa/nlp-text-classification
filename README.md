# NLP Text Classification

[![CI](https://github.com/cerenaaa/nlp-text-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/nlp-text-classification/actions)

Multi-label text classification pipeline: TF-IDF baseline, fine-tuned transformer, knowledge distillation, and calibrated confidence scores. Built for enterprise support ticket routing, content moderation, and intent detection.

## Pipeline

```
Raw Text → Preprocessing → Feature Extraction → Classifier → Calibration → Prediction
                                ↓
                     TF-IDF baseline (fast, interpretable)
                     Transformer fine-tune (high accuracy)
                     Distilled model (production speed)
```

## Structure
```
nlp-text-classification/
├── data/
│   └── synthetic_corpus.py     # Multi-label text corpus generator
├── features/
│   ├── preprocessor.py         # Cleaning, tokenization, normalization
│   └── tfidf_features.py       # TF-IDF with n-grams and SVD
├── models/
│   ├── tfidf_classifier.py     # Logistic regression + TF-IDF baseline
│   └── transformer_classifier.py # HuggingFace fine-tuning wrapper
├── evaluation/
│   └── multilabel_metrics.py   # micro/macro F1, coverage, ranking loss
└── train.py
```

## Quickstart
```bash
pip install -r requirements.txt
python train.py --model tfidf      # Fast baseline
python train.py --model transformer --model_name distilbert-base-uncased
```
