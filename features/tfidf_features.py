"""TF-IDF feature extraction with n-grams and optional SVD."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


class TFIDFExtractor:
    def __init__(self, max_features: int = 20_000, ngram_range: tuple = (1, 2),
                 use_svd: bool = True, n_components: int = 300):
        self.use_svd = use_svd
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._pipeline = None

    def _build_pipeline(self, n_features: int) -> Pipeline:
        steps = [("tfidf", TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range,
            sublinear_tf=True, strip_accents="unicode", analyzer="word",
            token_pattern=r"\w{2,}", min_df=2,
        ))]
        if self.use_svd:
            # n_components must be strictly less than n_features
            n_comp = min(self.n_components, max(1, n_features - 1))
            steps.append(("svd", TruncatedSVD(n_components=n_comp, random_state=42)))
        return Pipeline(steps)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        # Build with a placeholder; SVD components adjusted after TF-IDF
        tfidf = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range,
            sublinear_tf=True, strip_accents="unicode", analyzer="word",
            token_pattern=r"\w{2,}", min_df=2,
        )
        X_tfidf = tfidf.fit_transform(texts)
        n_features = X_tfidf.shape[1]

        if self.use_svd and n_features > 1:
            n_comp = min(self.n_components, n_features - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            X_out = svd.fit_transform(X_tfidf)
            self._pipeline = Pipeline([("tfidf", tfidf), ("svd", svd)])
        else:
            X_out = X_tfidf.toarray()
            self._pipeline = Pipeline([("tfidf", tfidf)])
        return X_out

    def transform(self, texts: list[str]) -> np.ndarray:
        result = self._pipeline.transform(texts)
        return result.toarray() if hasattr(result, "toarray") else result
