"""
Intrusion Detection using Ensemble Techniques

Multi-class network intrusion classification on the NSL-KDD dataset,
using a Maximum Voting Ensemble that combines Gaussian Naive Bayes,
a Decision Tree, and XGBoost.

Classifies network traffic into one of five categories:
Normal, DoS, Probe, R2L (Remote to Local), or U2R (User to Root).
"""

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Maps the ~40 specific attack labels in NSL-KDD down to the standard
# 5-category grouping: Normal, DoS, Probe, R2L, U2R
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "mailbomb": "dos", "worm": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r", "perl": "u2r",
    "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def load_and_prepare(train_path: str, test_path: str) -> pd.DataFrame:
    """Load both NSL-KDD files and combine them into a single pool
    (matching the common Kaggle-notebook approach of a random
    train/test split, rather than NSL-KDD's official held-out test)."""
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.drop(columns=["difficulty"])
    df["attack_cat"] = df["label"].map(ATTACK_CATEGORY_MAP)
    return df


def encode_features(df: pd.DataFrame):
    for col in ["protocol_type", "service", "flag"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["attack_cat"])
    X = df.drop(columns=["label", "attack_cat"]).values
    return X, y, label_encoder


def build_ensemble() -> VotingClassifier:
    """Maximum Voting Ensemble combining Gaussian Naive Bayes,
    Decision Tree, and XGBoost via hard (majority) voting."""
    return VotingClassifier(
        estimators=[
            ("naive_bayes", GaussianNB()),
            ("decision_tree", DecisionTreeClassifier(random_state=42)),
            ("xgboost", XGBClassifier(eval_metric="mlogloss", random_state=42)),
        ],
        voting="hard",
    )


def false_positive_rate(y_test, preds, label_encoder) -> float:
    """FPR for the 'normal' class: normal traffic incorrectly
    flagged as any kind of attack."""
    cm = confusion_matrix(y_test, preds)
    classes = list(label_encoder.classes_)
    normal_idx = classes.index("normal")
    total_normal = cm[normal_idx].sum()
    correctly_normal = cm[normal_idx][normal_idx]
    return (total_normal - correctly_normal) / total_normal


def main():
    print("Loading and preparing NSL-KDD dataset...")
    df = load_and_prepare("KDDTrain.txt", "KDDTest.txt")
    print(f"Total records: {df.shape[0]}")
    print(df["attack_cat"].value_counts())

    X, y, label_encoder = encode_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining Maximum Voting Ensemble (Naive Bayes + Decision Tree + XGBoost)...")
    ensemble = build_ensemble()
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)
    fpr = false_positive_rate(y_test, preds, label_encoder)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(f"False Positive Rate (normal traffic flagged as attack): {fpr * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()
