"""TF-IDF feature extraction with n-grams and optional SVD dimensionality reduction."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


class TFIDFExtractor:
    def __init__(self, max_features: int = 20_000, ngram_range: tuple = (1, 2),
                 use_svd: bool = True, n_components: int = 300):
        self.use_svd = use_svd
        steps = [("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, strip_accents="unicode", analyzer="word",
            token_pattern=r"\w{2,}", min_df=2,
        ))]
        if use_svd:
            steps.append(("svd", TruncatedSVD(n_components=n_components, random_state=42)))
        self.pipeline = Pipeline(steps)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.fit_transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.transform(texts)
