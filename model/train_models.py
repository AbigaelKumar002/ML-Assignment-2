"""
Assignment 2 - Model Training Script
Dataset: Optical Recognition of Handwritten Digits (UCI / sklearn built-in)
Multi-class classification (10 classes: digits 0-9)
Trains: Logistic Regression, Decision Tree, KNN, Naive Bayes (Gaussian), Random Forest
Saves: trained models (pickle), scaler, test_data.csv, metrics_summary.csv
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
data = load_digits(as_frame=True)
df = data.frame  # 64 pixel-intensity features + 'target' (digit 0-9)

X = df.drop(columns=["target"])
y = df["target"]

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}, Instances: {X.shape[0]}, Classes: {y.nunique()}")

# ---------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the test data (features + true label) - this is what the Streamlit app uses
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT_DIR, "test_data.csv"), index=False)
print("Saved test_data.csv")

# ---------------------------------------------------------
# 3. Scale features
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)  # shape (n_samples, 10) for multi-class

    acc = accuracy_score(y_test, y_pred)
    # multi-class AUC: one-vs-rest, weighted by class support
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(os.path.join(BASE_DIR, f"{fname}.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"{name:30s} Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")

# ---------------------------------------------------------
# 5. Save metrics summary (used to build README table)
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(ROOT_DIR, "metrics_summary.csv"), index=False)
print("\nSaved metrics_summary.csv")
print(results_df.to_string(index=False))
