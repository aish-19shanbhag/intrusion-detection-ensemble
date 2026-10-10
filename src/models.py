"""The three base classifiers and the Maximum Voting Ensemble that combines them."""

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS, RANDOM_STATE
from src.preprocessing import build_preprocessor

DEFAULT_PARAMS = {
    "features": FEATURE_COLUMNS,
    "decision_tree": {"max_depth": None, "min_samples_leaf": 1},
    "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3},
    "weights": [1, 1, 1],
}


def base_models(params: dict = DEFAULT_PARAMS) -> dict:
    """Each base learner on its own, used for benchmarking against the ensemble."""
    return {
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, **params["decision_tree"]),
        "xgboost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE,
                                 n_jobs=-1, **params["xgboost"]),
    }


def build_ensemble(params: dict = DEFAULT_PARAMS) -> VotingClassifier:
    """Maximum Voting Ensemble: each model votes for a class; the weighted majority wins.

    The three learners make different kinds of mistakes (a probabilistic model,
    a single tree, and a boosted forest), which is what makes voting useful.
    """
    return VotingClassifier(
        estimators=list(base_models(params).items()),
        voting="hard",
        weights=params["weights"],
    )


def build_pipeline(model, features: list = FEATURE_COLUMNS) -> Pipeline:
    """Preprocessing and model in one object: fit once, then predict on raw rows."""
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])
