"""Metrics and charts used in every evaluation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.config import CLASS_NAMES

NORMAL = CLASS_NAMES.index("normal")
DISPLAY = {"naive_bayes": "Naive Bayes", "decision_tree": "Decision Tree", "xgboost": "XGBoost", "ensemble": "Ensemble"}


def false_positive_rate(y_true, y_pred) -> float:
    """Share of genuinely normal traffic that the model flags as an attack."""
    normal = y_true == NORMAL
    return float((y_pred[normal] != NORMAL).mean())


def detection_rate(y_true, y_pred) -> float:
    """Share of attack traffic flagged as some attack (attack-vs-normal recall)."""
    attack = y_true != NORMAL
    return float((y_pred[attack] != NORMAL).mean())


def summarize(y_true, y_pred) -> dict:
    """All headline metrics for one model on one test set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "detection_rate": detection_rate(y_true, y_pred),
        "per_class_recall": dict(zip(CLASS_NAMES, map(float, recall_score(
            y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)))),
        "report": classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                        target_names=CLASS_NAMES, digits=4, zero_division=0,
                                        output_dict=True),
    }


def plot_confusion_matrix(y_true, y_pred, title, path):
    """Row-normalized confusion matrix: each row shows where one true class went."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.5 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict, metric: str, title: str, path):
    """Bar chart of one metric across the base models and the ensemble."""
    names = list(results)
    values = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([DISPLAY.get(n, n) for n in names], values,
                  color=["#9aa5b1"] * (len(names) - 1) + ["#1f4e79"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_recall(results: dict, title: str, path):
    """Grouped bars: recall for each attack category, per model."""
    names = list(results)
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, n in enumerate(names):
        vals = [results[n]["per_class_recall"][c] for c in CLASS_NAMES]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=DISPLAY.get(n, n))
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(names), loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
