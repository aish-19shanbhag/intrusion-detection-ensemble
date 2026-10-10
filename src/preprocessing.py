"""Feature preprocessing and feature selection.

Preprocessing (encode categoricals, drop constant columns, standardize) is fitted on
training data only and stored inside one scikit-learn Pipeline, so the same
transformations run at training time and at prediction time.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, RANDOM_STATE


def build_preprocessor(features: list = FEATURE_COLUMNS) -> Pipeline:
    """Keep the chosen features -> encode categoricals -> drop constant columns -> standardize.

    Unseen categories (for example a new `service` value) are encoded as -1
    instead of crashing, which matters when scoring new traffic.
    """
    categorical = [c for c in features if c in CATEGORICAL_COLUMNS]
    numeric = [c for c in features if c not in CATEGORICAL_COLUMNS]
    encode = ColumnTransformer(
        [
            ("categorical",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("encode", encode),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ])


def rank_features(X: pd.DataFrame, y: np.ndarray, sample_size: int = 30000) -> pd.Series:
    """Rank all 41 features by mutual information with the attack category.

    Mutual information measures how much knowing a feature reduces uncertainty
    about the class, and it captures non-linear relationships. A stratified-size
    random sample keeps this fast; the ranking is stable at this size.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Xs = X.iloc[idx].copy()
    for col in CATEGORICAL_COLUMNS:
        Xs[col] = Xs[col].astype("category").cat.codes
    discrete = [c in CATEGORICAL_COLUMNS for c in X.columns]
    scores = mutual_info_classif(Xs, y[idx], discrete_features=discrete, random_state=RANDOM_STATE)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)
