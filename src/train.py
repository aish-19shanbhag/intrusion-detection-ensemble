"""Train the tuned ensemble, benchmark it against each base model, and save everything.

Run:  python -m src.train      (after python -m src.tune)

Two evaluations are reported:
  A. Hold-out: a stratified 80/20 split of KDDTrain (the primary result).
  B. Official benchmark: train on all of KDDTrain, test on KDDTest, which contains
     attack types never seen in training. This is the harder, more realistic test.
"""

import json
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import (BEST_PARAMS_FILE, CLASS_NAMES, FIGURES_DIR, METRICS_DIR, MODEL_FILE,
                        RANDOM_STATE, TEST_FILE, TEST_SIZE, TRAIN_FILE)
from src.data_loader import load_nsl_kdd, split_xy
from src.evaluate import (plot_confusion_matrix, plot_model_comparison, plot_per_class_recall,
                          summarize)
from src.models import DEFAULT_PARAMS, base_models, build_ensemble, build_pipeline


def all_models(params):
    models = base_models(params)
    models["ensemble"] = build_ensemble(params)
    return models


def evaluate_all(params, X_tr, y_tr, X_te, y_te):
    """Fit every model on the same training data and score it on the same test data."""
    results, fitted = {}, {}
    for name, model in all_models(params).items():
        pipe = build_pipeline(model, params["features"])
        start = time.time()
        pipe.fit(X_tr, y_tr)
        train_seconds = time.time() - start
        pred = pipe.predict(X_te)
        results[name] = summarize(y_te, pred)
        results[name]["train_seconds"] = round(train_seconds, 1)
        fitted[name] = (pipe, pred)
        print(f"  {name:13s} accuracy={results[name]['accuracy']:.4%} "
              f"FPR={results[name]['false_positive_rate']:.4%} macroF1={results[name]['macro_f1']:.4f}")
    return results, fitted


def measure_latency(pipe, X, repeats=200):
    """Median time to score one connection, and throughput on a batch of 10,000."""
    single = []
    for i in range(repeats):
        row = X.iloc[[i]]
        start = time.perf_counter()
        pipe.predict(row)
        single.append(time.perf_counter() - start)
    batch = X.iloc[:10000]
    start = time.perf_counter()
    pipe.predict(batch)
    batch_seconds = time.perf_counter() - start
    return {"single_row_median_ms": round(float(np.median(single)) * 1000, 2),
            "batch_10k_seconds": round(batch_seconds, 3),
            "rows_per_second": int(len(batch) / batch_seconds)}


def main():
    params = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else DEFAULT_PARAMS
    train_df, test_df = load_nsl_kdd(TRAIN_FILE), load_nsl_kdd(TEST_FILE)
    X, y = split_xy(train_df)
    X_official, y_official = split_xy(test_df)
    metrics = {"params": {k: v for k, v in params.items() if k != "features"},
               "features_kept": params["features"]}

    # A. Hold-out evaluation
    print("A. Hold-out (stratified 80/20 split of KDDTrain)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=RANDOM_STATE, stratify=y)
    holdout, fitted = evaluate_all(params, X_tr, y_tr, X_te, y_te)
    metrics["holdout"] = {"train_rows": len(X_tr), "test_rows": len(X_te), "results": holdout}
    ens_pipe, ens_pred = fitted["ensemble"]
    plot_confusion_matrix(y_te, ens_pred, "Ensemble: hold-out confusion matrix",
                          FIGURES_DIR / "confusion_matrix_holdout.png")
    plot_model_comparison(holdout, "accuracy", "Hold-out accuracy by model",
                          FIGURES_DIR / "model_comparison_holdout.png")

    # Stability: 5-fold cross-validation of the full pipeline on the training split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_pipeline(build_ensemble(params), params["features"]),
                             X_tr, y_tr, cv=cv, scoring="accuracy")
    metrics["cross_validation"] = {"folds": 5, "accuracy_mean": float(scores.mean()),
                                   "accuracy_std": float(scores.std()),
                                   "fold_scores": [float(s) for s in scores]}
    print(f"  5-fold CV accuracy: {scores.mean():.4%} +/- {scores.std():.4%}")

    metrics["latency"] = measure_latency(ens_pipe, X_te)
    print("  latency:", metrics["latency"])

    # B. Official benchmark
    print("B. Official benchmark (train KDDTrain, test KDDTest)")
    official, fitted_off = evaluate_all(params, X, y, X_official, y_official)
    metrics["official_benchmark"] = {"train_rows": len(X), "test_rows": len(X_official),
                                     "results": official}
    plot_confusion_matrix(y_official, fitted_off["ensemble"][1],
                          "Ensemble: official KDDTest confusion matrix",
                          FIGURES_DIR / "confusion_matrix_official.png")
    plot_model_comparison(official, "accuracy", "Official KDDTest accuracy by model",
                          FIGURES_DIR / "model_comparison_official.png")
    plot_per_class_recall(official, "Recall per attack category (official KDDTest)",
                          FIGURES_DIR / "per_class_recall_official.png")

    # Save the production model (trained on all of KDDTrain) and the metrics
    joblib.dump({"pipeline": fitted_off["ensemble"][0], "class_names": CLASS_NAMES,
                 "params": metrics["params"]}, MODEL_FILE, compress=3)
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved", MODEL_FILE.name, "and reports/metrics/metrics.json")


if __name__ == "__main__":
    main()
