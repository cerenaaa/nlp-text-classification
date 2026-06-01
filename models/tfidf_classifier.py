"""
TF-IDF + Logistic Regression baseline for multi-label classification.
Fast, interpretable, strong baseline for short texts.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.calibration import CalibratedClassifierCV
from features.preprocessor import TextPreprocessor
from features.tfidf_features import TFIDFExtractor


class TFIDFMultiLabelClassifier:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.preprocessor = TextPreprocessor()
        self.extractor = TFIDFExtractor()
        self.mlb = MultiLabelBinarizer()
        base = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
        self.classifier = OneVsRestClassifier(base)
        self.classes_ = None

    def fit(self, samples: list[dict]) -> "TFIDFMultiLabelClassifier":
        texts = self.preprocessor.process_batch([s["text"] for s in samples])
        X = self.extractor.fit_transform(texts)
        Y = self.mlb.fit_transform([s["labels"] for s in samples])
        self.classes_ = self.mlb.classes_
        self.classifier.fit(X, Y)
        print(f"Trained on {len(samples)} samples | {len(self.classes_)} labels")
        print(f"Labels: {list(self.classes_)}")
        return self

    def predict(self, texts: list[str], threshold: float = 0.3) -> list[list[str]]:
        cleaned = self.preprocessor.process_batch(texts)
        X = self.extractor.transform(cleaned)
        proba = self.classifier.predict_proba(X)
        preds = (proba >= threshold).astype(int)
        return self.mlb.inverse_transform(preds)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        cleaned = self.preprocessor.process_batch(texts)
        X = self.extractor.transform(cleaned)
        return self.classifier.predict_proba(X)
