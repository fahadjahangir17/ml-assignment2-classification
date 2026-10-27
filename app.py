"""
Shopper Intent Lab: a Streamlit front end for five classifiers that predict
whether an online-shop session ends in a purchase (the `Revenue` column of the
UCI Online Shoppers Purchasing Intention dataset).

Run locally:  streamlit run app.py
"""
import io
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_curve)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))
import train_models as tm  # noqa: E402  (needs the sys.path line above)

st.set_page_config(page_title="Shopper Intent Lab", page_icon="🛒", layout="wide")

MODEL_NAMES = list(tm.MODEL_FILES.keys())
SAMPLE_PATH = os.path.join(ROOT, "test_data.csv")


# ---------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load the saved pipelines. If a pickle was written by a different
    scikit-learn version and won't open, retrain from the bundled CSV instead,
    so the app still starts on whatever version the host installs."""
    try:
        return {n: joblib.load(os.path.join(tm.MODEL_DIR, f))
                for n, f in tm.MODEL_FILES.items()}, False
    except Exception:
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        df = tm.load_clean_data()
        X_tr, _, y_tr, _ = train_test_split(
            df[tm.FEATURES], df[tm.TARGET], test_size=0.2,
            stratify=df[tm.TARGET], random_state=tm.RANDOM_STATE)
        params = pd.read_json(os.path.join(tm.MODEL_DIR, "best_params.json"), typ="series")
        models = {}
        for name, (est, _grid) in tm.candidate_models().items():
            est.set_params(**params[name])
            models[name] = Pipeline([("prep", tm.make_preprocessor()), ("clf", est)]).fit(X_tr, y_tr)
        return models, True


@st.cache_data
def load_saved_metrics():
    return pd.read_csv(os.path.join(tm.MODEL_DIR, "metrics.csv"))


def prepare(df):
    """Check the upload has the expected columns and fix the boolean types."""
    missing = [c for c in tm.FEATURES if c not in df.columns]
    if missing:
        return None, missing
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    if tm.TARGET in df.columns:
        df[tm.TARGET] = df[tm.TARGET].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0})
    return df, []


# ----------------------------------------------------------------------- header
st.title("🛒 Shopper Intent Lab")
st.write(
    "Upload a CSV of website sessions, pick a model, and see how well it spots the "
    "sessions that end in a purchase. Buyers are only about 1 in 6 sessions, so "
    "accuracy alone is a weak signal here. Recall, F1 and MCC tell you more."
)

models, retrained = load_models()
if retrained:
    st.info("The saved model files were built with a different scikit-learn version, "
            "so the models were retrained from the bundled data on startup.")

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Data")
    upload = st.file_uploader("Upload test data (CSV)", type="csv",
                              help="Same columns as test_data.csv. Include `Revenue` to get scores.")
    st.caption("No file uploaded? The app scores the bundled test_data.csv so you can see it working.")
    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", MODEL_NAMES, index=1)
    st.caption("Every model is a pipeline: scaling and one-hot encoding happen "
               "inside, so the CSV can stay in its raw form.")

if upload is not None:
    raw = pd.read_csv(upload)
    source = f"uploaded file `{upload.name}`"
elif os.path.exists(SAMPLE_PATH):
    raw = pd.read_csv(SAMPLE_PATH)
    source = "bundled `test_data.csv`"
else:
    st.warning("Upload a CSV to get started.")
    st.stop()

data, missing = prepare(raw)
if data is None:
    st.error("The file is missing these columns: " + ", ".join(missing))
    st.stop()

has_labels = tm.TARGET in data.columns and data[tm.TARGET].notna().all()
if has_labels and data[tm.TARGET].nunique() < 2:
    st.warning("The `Revenue` column has only one class, so AUC and the other scores cannot be computed. "
               "Showing predictions only.")
    has_labels = False
X = data[tm.FEATURES]
model = models[model_name]
pred = model.predict(X)
score = tm.get_scores(model, X)

st.caption(f"Scoring {len(data):,} sessions from the {source} with **{model_name}**.")

if not has_labels and tm.TARGET not in data.columns:
    st.warning("No `Revenue` column found, so only predictions are shown (no metrics).")

tabs = st.tabs(["Metrics", "Confusion matrix", "ROC curve", "Compare all models", "Predictions"])

# ---------------------------------------------------------------------- metrics
with tabs[0]:
    if has_labels:
        y = data[tm.TARGET].astype(int)
        m = tm.compute_metrics(y, pred, score)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, f"{v:.4f}")
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        st.markdown(
            f"In plain terms: of **{tp + fn}** sessions that really ended in a purchase, "
            f"the model caught **{tp}** and missed **{fn}**. It also flagged **{fp}** "
            f"sessions as buyers that never bought."
        )
    else:
        st.write(f"The model predicts **{int(pred.sum())}** buying sessions "
                 f"out of {len(pred):,}.")

# -------------------------------------------------------------- confusion matrix
with tabs[1]:
    if has_labels:
        left, right = st.columns([1, 1])
        with left:
            fig, ax = plt.subplots(figsize=(4.2, 3.8))
            ConfusionMatrixDisplay.from_predictions(
                y, pred, display_labels=["No purchase", "Purchase"],
                cmap="Blues", ax=ax, colorbar=False)
            ax.set_title(f"{model_name}")
            st.pyplot(fig)
        with right:
            st.subheader("Classification report")
            rep = classification_report(y, pred, target_names=["No purchase", "Purchase"],
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")
    else:
        st.info("Add a `Revenue` column to see the confusion matrix and report.")

# -------------------------------------------------------------------- ROC curve
with tabs[2]:
    if has_labels:
        fpr, tpr, _ = roc_curve(y, score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {m['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("Add a `Revenue` column to draw the ROC curve.")

# ------------------------------------------------------------ compare all models
with tabs[3]:
    if has_labels:
        rows = []
        for name, mdl in models.items():
            p, s = mdl.predict(X), tm.get_scores(mdl, X)
            rows.append({"ML Model Name": name, **tm.compute_metrics(y, p, s)})
        comp = pd.DataFrame(rows).set_index("ML Model Name")
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                     width="stretch")
        st.caption("Green marks the best value in each column.")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        comp[["AUC", "F1", "MCC"]].plot.bar(ax=ax, rot=15, colormap="viridis")
        ax.set_ylim(0, 1.1)
        ax.legend(ncol=3, loc="upper center", frameon=False)
        ax.set_xlabel("")
        st.pyplot(fig)
    else:
        st.info("Add a `Revenue` column to compare the models.")

# ------------------------------------------------------------------ predictions
with tabs[4]:
    out = data.copy()
    out["Predicted_Revenue"] = pred
    out["Purchase_Score"] = score.round(4)
    st.dataframe(out.head(200), width="stretch")
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("Download all predictions (CSV)", buf.getvalue(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")
    st.caption("Purchase_Score is a probability for most models. For the SVM it is the "
               "signed distance from the decision boundary, so only its ordering matters.")
