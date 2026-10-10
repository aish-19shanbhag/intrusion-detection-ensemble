"""Load NSL-KDD files and attach the 5-category attack label."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ATTACK_CATEGORY_MAP, CLASS_NAMES, COLUMN_NAMES, FEATURE_COLUMNS


def load_nsl_kdd(path: Path) -> pd.DataFrame:
    """Read one NSL-KDD file and add an `attack_cat` column (dos, normal, probe, r2l, u2r)."""
    df = pd.read_csv(path, names=COLUMN_NAMES)
    df = df.drop(columns=["difficulty"])
    df["attack_cat"] = df["label"].map(ATTACK_CATEGORY_MAP)
    unmapped = df["attack_cat"].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} rows have labels missing from ATTACK_CATEGORY_MAP")
    return df


def split_xy(df: pd.DataFrame):
    """Return the feature frame and the integer target (index into CLASS_NAMES)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["attack_cat"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy(dtype=np.int64)
    return X, y
