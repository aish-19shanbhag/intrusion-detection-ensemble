"""Feature selection and hyperparameter tuning with stratified cross-validation.

Run:  python -m src.tune

Only the 80% training split is used here. The 20% hold-out is never touched
during tuning, so the final hold-out score is an honest estimate.
"""

import itertools
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import (BEST_PARAMS_FILE, CV_FOLDS, FIGURES_DIR, METRICS_DIR, RANDOM_STATE,
                        TEST_SIZE, TRAIN_FILE)
from src.data_loader import load_nsl_kdd, split_xy
from src.models import DEFAULT_PARAMS, base_models, build_ensemble, build_pipeline
from src.preprocessing import rank_features

CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rows = []


def cv_score(model, features, X, y, stage, setting):
    """Mean cross-validated accuracy and macro-F1 for one configuration."""
    start = time.time()
    s = cross_validate(build_pipeline(model, features), X, y, cv=CV,
                       scoring=["accuracy", "f1_macro"], n_jobs=1)
    row = {"stage": stage, "setting": setting,
           "cv_accuracy": s["test_accuracy"].mean(), "cv_accuracy_std": s["test_accuracy"].std(),
           "cv_macro_f1": s["test_f1_macro"].mean(), "seconds": round(time.time() - start, 1)}
    rows.append(row)
    print(f"[{stage}] {setting}: acc={row['cv_accuracy']:.5f} macroF1={row['cv_macro_f1']:.4f}", flush=True)
    return row


def best_of(stage_rows, tolerance=0.0):
    """Highest CV accuracy; with a tolerance, the earliest (simplest) setting within it."""
    top = max(r["cv_accuracy"] for r in stage_rows)
    return next(r for r in stage_rows if r["cv_accuracy"] >= top - tolerance)


def main():
    df = load_nsl_kdd(TRAIN_FILE)
    X, y = split_xy(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=RANDOM_STATE, stratify=y)
    params = json.loads(json.dumps(DEFAULT_PARAMS))

    # 1. Feature selection: rank by mutual information, then choose how many to keep
    ranking = rank_features(X_train, y_train)
    ranking.rename("mutual_information").to_csv(METRICS_DIR / "feature_ranking.csv")
    fig, ax = plt.subplots(figsize=(7, 7))
    ranking[::-1].plot.barh(ax=ax, color="#1f4e79")
    ax.set_xlabel("mutual information with attack category")
    ax.set_title("Feature ranking (training split)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_ranking.png", dpi=150)
    plt.close(fig)

    stage = []
    for k in (10, 20, 30, len(ranking)):
        feats = list(ranking.index[:k])
        r = cv_score(base_models(params)["xgboost"], feats, X_train, y_train, "features", f"top {k}")
        r["k"] = k
        stage.append(r)
    # Keep the smallest feature set within 0.02 percentage points of the best
    k_best = best_of(stage, tolerance=0.0002)["k"]
    params["features"] = list(ranking.index[:k_best])

    # 2. XGBoost
    stage = []
    for n, d, lr in itertools.product((100, 300), (6, 10), (0.1, 0.3)):
        p = {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        trial = {**params, "xgboost": p}
        r = cv_score(base_models(trial)["xgboost"], params["features"], X_train, y_train, "xgboost", json.dumps(p))
        r["p"] = p
        stage.append(r)
    params["xgboost"] = best_of(stage)["p"]

    # 3. Decision tree
    stage = []
    for d, leaf in itertools.product((None, 30), (1, 2)):
        p = {"max_depth": d, "min_samples_leaf": leaf}
        trial = {**params, "decision_tree": p}
        r = cv_score(base_models(trial)["decision_tree"], params["features"], X_train, y_train, "decision_tree", json.dumps(p))
        r["p"] = p
        stage.append(r)
    params["decision_tree"] = best_of(stage)["p"]

    # 4. Voting weights (naive bayes, decision tree, xgboost)
    stage = []
    for w in ([1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 2, 3]):
        trial = {**params, "weights": w}
        r = cv_score(build_ensemble(trial), params["features"], X_train, y_train, "voting_weights", str(w))
        r["p"] = w
        stage.append(r)
    params["weights"] = best_of(stage)["p"]

    pd.DataFrame(rows).drop(columns=["p", "k"], errors="ignore").to_csv(
        METRICS_DIR / "tuning_results.csv", index=False)
    BEST_PARAMS_FILE.write_text(json.dumps(params, indent=2))
    print("\nBest parameters saved to", BEST_PARAMS_FILE)
    print(json.dumps({k: v for k, v in params.items() if k != "features"}, indent=2),
          f"\nfeatures kept: {len(params['features'])}")


if __name__ == "__main__":
    main()
