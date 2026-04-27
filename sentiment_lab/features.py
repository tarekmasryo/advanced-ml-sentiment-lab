from __future__ import annotations

from typing import Any

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def build_advanced_features(
    texts: list[str],
    max_word_features: int,
    use_char: bool,
    char_max: int,
) -> tuple[Any, tuple[TfidfVectorizer, ...]]:
    """Fit TF-IDF vectorizers on training text only and return transformed train matrix."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=max_word_features,
        min_df=2,
        max_df=0.95,
    )
    x_word = word_vec.fit_transform(texts)

    vectorizers: list[TfidfVectorizer] = [word_vec]
    matrices = [x_word]

    if use_char:
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            max_features=char_max,
            min_df=2,
        )
        x_char = char_vec.fit_transform(texts)
        vectorizers.append(char_vec)
        matrices.append(x_char)

    x_all = hstack(matrices) if len(matrices) > 1 else matrices[0]
    return x_all, tuple(vectorizers)


def transform_advanced_features(
    texts: list[str],
    vectorizers: tuple[TfidfVectorizer, ...],
) -> Any:
    """Transform text using already-fitted vectorizers.

    This deliberately never calls ``fit``. It is used for validation and
    inference so evaluation/inference text cannot influence vocabulary or IDF.
    """
    if not vectorizers:
        raise ValueError("At least one fitted vectorizer is required.")

    matrices = [vec.transform(texts) for vec in vectorizers]
    return hstack(matrices) if len(matrices) > 1 else matrices[0]


def make_feature_union(max_word_features: int, use_char: bool, char_max: int):
    """Build a scikit-learn transformer for leakage-free Pipeline-based evaluation."""
    from sklearn.pipeline import FeatureUnion

    transformers = [
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=max_word_features,
                min_df=2,
                max_df=0.95,
            ),
        )
    ]
    if use_char:
        transformers.append(
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 6),
                    max_features=char_max,
                    min_df=2,
                ),
            )
        )
    return FeatureUnion(transformers)
