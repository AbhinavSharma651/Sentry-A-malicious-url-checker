"""
Train a malicious URL classifier on malicious_phish.csv.

Pipeline:
 1. Load CSV (url, type)
 2. Extract lexical features from each URL
 3. Train/test split (stratified)
 4. Train a RandomForestClassifier
 5. Evaluate (accuracy, per-class precision/recall/F1, confusion matrix)
 6. Save model + label encoder + feature name list to model/
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from features import extract_features_batch, FEATURE_NAMES

DATA_PATH = "/mnt/user-data/uploads/malicious_phish.csv"
MODEL_DIR = "model"

def main():
    t0 = time.time()
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["url", "type"])
    print(f"  {len(df):,} rows, classes: {df['type'].value_counts().to_dict()}")

    print("Extracting lexical features (this takes a minute for 650k rows)...")
    feat_dicts = extract_features_batch(df["url"].astype(str).tolist())
    X = pd.DataFrame(feat_dicts, columns=FEATURE_NAMES)
    print(f"  done in {time.time() - t0:.1f}s, feature matrix shape {X.shape}")

    le = LabelEncoder()
    y = le.fit_transform(df["type"])
    print(f"  classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    t1 = time.time()
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    print(f"  trained in {time.time() - t1:.1f}s")

    print("Evaluating on held-out test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(le.classes_)
    print(cm)

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: -x[1],
    )
    print("\nTop 10 feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name:22s} {imp:.4f}")

    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, f"{MODEL_DIR}/url_classifier.joblib", compress=3)
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")
    with open(f"{MODEL_DIR}/feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classes": list(le.classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "confusion_matrix": cm.tolist(),
        "top_features": [(n, float(i)) for n, i in importances[:10]],
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel + artifacts saved to {MODEL_DIR}/")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
