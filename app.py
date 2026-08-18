import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Handwritten Digits Classification Demo", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_scaler():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(filename):
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        return pickle.load(f)


st.title("🔢 Handwritten Digit Classification — Model Comparison App")
st.markdown(
    """
    This app demonstrates **5 classification models** trained on the
    **Optical Recognition of Handwritten Digits** dataset (originally from UCI,
    64 pixel-intensity features, 1797 instances, 10-class target: digits 0-9).

    Upload the provided `test_data.csv`, pick a model, and view its performance.
    """
)

# ---------------------------------------------------------
# 1. Dataset upload
# ---------------------------------------------------------
st.header("1️⃣ Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (must contain 64 pixel feature columns + a 'target' column)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded data with shape {data.shape}")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column with true digit labels (0-9).")
        st.stop()

    X = data.drop(columns=["target"])
    y_true = data["target"]

    # ---------------------------------------------------------
    # Optional: preview a sample digit image
    # ---------------------------------------------------------
    with st.expander("🖼️ Preview a sample digit (8x8 pixel grid)"):
        idx = st.slider("Row index", 0, len(data) - 1, 0)
        pixel_vals = X.iloc[idx].values.reshape(8, 8)
        fig_img, ax_img = plt.subplots(figsize=(2, 2))
        ax_img.imshow(pixel_vals, cmap="gray_r")
        ax_img.set_title(f"True label: {y_true.iloc[idx]}")
        ax_img.axis("off")
        st.pyplot(fig_img)

    # ---------------------------------------------------------
    # 2. Model selection
    # ---------------------------------------------------------
    st.header("2️⃣ Select a Model")
    model_choice = st.selectbox("Choose a classification model:", list(MODEL_FILES.keys()))

    scaler = load_scaler()
    model = load_model(MODEL_FILES[model_choice])

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Feature mismatch with training data: {e}")
        st.stop()

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)  # shape (n_samples, 10)

    # ---------------------------------------------------------
    # 3. Evaluation metrics
    # ---------------------------------------------------------
    st.header("3️⃣ Evaluation Metrics")

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC (OvR)", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ---------------------------------------------------------
    # 4. Confusion matrix + classification report
    # ---------------------------------------------------------
    st.header("4️⃣ Confusion Matrix & Classification Report")

    c1, c2 = st.columns(2)

    with c1:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("Actual digit")
        ax.set_title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)

    with c2:
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))

    # ---------------------------------------------------------
    # 5. Compare all models on this data
    # ---------------------------------------------------------
    st.header("5️⃣ Compare All Models on This Test Data")
    if st.checkbox("Run and compare all 5 models"):
        rows = []
        for name, fname in MODEL_FILES.items():
            m = load_model(fname)
            yp = m.predict(X_scaled)
            ypr = m.predict_proba(X_scaled)
            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_true, yp), 4),
                "AUC": round(roc_auc_score(y_true, ypr, multi_class="ovr", average="weighted"), 4),
                "Precision": round(precision_score(y_true, yp, average="weighted"), 4),
                "Recall": round(recall_score(y_true, yp, average="weighted"), 4),
                "F1": round(f1_score(y_true, yp, average="weighted"), 4),
                "MCC": round(matthews_corrcoef(y_true, yp), 4),
            })
        st.dataframe(pd.DataFrame(rows))

else:
    st.info("👆 Upload the `test_data.csv` file from this repository to get started.")
