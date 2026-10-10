"""Score new network connections with the saved ensemble.

Run:  python -m src.predict data/sample_connections.csv
The CSV needs the 41 NSL-KDD feature columns (extra columns are ignored).
"""

import sys

import joblib
import pandas as pd

from src.config import CLASS_DESCRIPTIONS, FEATURE_COLUMNS, MODEL_FILE


def load_model(path=MODEL_FILE):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["class_names"]


def predict_frame(df: pd.DataFrame, pipeline=None, class_names=None) -> pd.DataFrame:
    """Return the predicted category and each base model's vote for every row."""
    if pipeline is None:
        pipeline, class_names = load_model()
    X = df[FEATURE_COLUMNS]
    out = pd.DataFrame(index=df.index)
    out["prediction"] = [class_names[i] for i in pipeline.predict(X)]
    out["meaning"] = out["prediction"].map(CLASS_DESCRIPTIONS)
    # Individual votes: run the fitted preprocessing once, then ask each base model
    Xt = pipeline.named_steps["preprocess"].transform(X)
    for name, model in pipeline.named_steps["model"].named_estimators_.items():
        out[f"vote_{name}"] = [class_names[i] for i in model.predict(Xt)]
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.predict <file.csv>")
    df = pd.read_csv(sys.argv[1])
    result = predict_frame(df)
    if "attack_cat" in df.columns:
        result.insert(0, "actual", df["attack_cat"])
    print(result.to_string())


if __name__ == "__main__":
    main()
