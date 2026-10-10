"""Fast checks on a small sample: run with  pytest -q"""

import numpy as np
import pandas as pd

from src.config import ATTACK_CATEGORY_MAP, CLASS_NAMES, TRAIN_FILE
from src.data_loader import load_nsl_kdd, split_xy
from src.evaluate import false_positive_rate
from src.models import DEFAULT_PARAMS, build_ensemble, build_pipeline

SAMPLE = load_nsl_kdd(TRAIN_FILE).sample(3000, random_state=0)


def test_every_label_maps_to_a_category():
    assert set(ATTACK_CATEGORY_MAP.values()) == set(CLASS_NAMES)
    assert SAMPLE["attack_cat"].notna().all()


def test_split_shapes():
    X, y = split_xy(SAMPLE)
    assert X.shape == (3000, 41)
    assert y.min() >= 0 and y.max() < len(CLASS_NAMES)


def test_pipeline_trains_and_handles_unseen_category():
    X, y = split_xy(SAMPLE)
    pipe = build_pipeline(build_ensemble(DEFAULT_PARAMS)).fit(X, y)
    new = X.head(5).copy()
    new["service"] = "a_service_never_seen"
    assert len(pipe.predict(new)) == 5
    assert (pipe.predict(X) == y).mean() > 0.95


def test_false_positive_rate():
    normal = CLASS_NAMES.index("normal")
    y_true = np.array([normal, normal, normal, normal, 0])
    y_pred = np.array([normal, normal, normal, 0, 0])
    assert false_positive_rate(y_true, y_pred) == 0.25
