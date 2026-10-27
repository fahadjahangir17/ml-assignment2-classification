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

st.set_page_config(page_title="Shopper Intent Lab", layout="wide")

MODEL_NAMES = list(tm.MODEL_FILES.keys())
SAMPLE_PATH = os.path.join(ROOT, "test_data.csv")

# ------------------------------------------------------------------------ look
INK, PAPER, BLUE, CORAL, AMBER, SLATE = "#121B33", "#F6F7FA", "#2447F5", "#E8445A", "#F5A524", "#D5DBE8"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": "#B9C1D3",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Public+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] {{ font-family: 'Public Sans', system-ui, sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .headline, .num {{
    font-family: 'Bricolage Grotesque', 'Public Sans', system-ui, sans-serif; letter-spacing: -0.02em; }}
.stApp h1 {{ font-size: 2.8rem; font-weight: 800; line-height: 1.05; padding-bottom: 0.2rem; }}
.stApp .headline {{ font-size: 2.1rem; font-weight: 800; line-height: 1.1; margin: 1.6rem 0 0.3rem; }}
.stApp .subline {{ font-size: 1.05rem; margin: 0 0 1.1rem; max-width: 70ch; }}
.grid {{ display: grid; grid-template-columns: repeat(40, 1fr); gap: 3px; max-width: 820px; }}
.c {{ display: block; aspect-ratio: 1; border-radius: 2px; }}
.tp {{ background: {BLUE}; }}
.fn {{ background: {PAPER}; box-shadow: inset 0 0 0 2px {CORAL}; }}
.fp {{ background: {AMBER}; }}
.tn {{ background: {SLATE}; }}
.stApp .legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.8rem; list-style: none; padding: 0 !important; margin: 1rem 0 0.3rem; }}
.stApp .legend li {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; margin: 0; padding: 0; }}
.legend .c {{ width: 14px; }}
.stApp .scale {{ font-size: 0.85rem; opacity: 0.75; margin: 0 0 1.8rem; }}
.stats {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 1.2rem; max-width: 820px; margin-bottom: 1.5rem; }}
.stApp .num {{ font-size: 2rem; font-weight: 800; line-height: 1; }}
.stApp .lbl {{ font-size: 0.9rem; margin-top: 0.25rem; }}
@media (max-width: 640px) {{
    .grid {{ grid-template-columns: repeat(20, 1fr); }}
    .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .stApp h1 {{ font-size: 2.1rem; }}
}}
</style>""", unsafe_allow_html=True)


def allocate(counts, cells=400):
    """Share `cells` squares between the groups in proportion to their counts.
    Every group that exists gets at least one square, so small groups stay visible."""
    raw = [c * cells / sum(counts) for c in counts]
    squares = [max(int(r), 1) if c else 0 for r, c in zip(raw, counts)]
    by_remainder = sorted(range(len(counts)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    i = 0
    while sum(squares) < cells:
        squares[by_remainder[i % len(counts)]] += 1
        i += 1
    while sum(squares) > cells:
        squares[squares.index(max(squares))] -= 1
    return squares


def session_grid(groups, total):
    """groups: list of (css class, label, count). Returns HTML for the grid and its legend."""
    squares = allocate([n for _, _, n in groups])
    scale_note = (f"Each square stands for about {total / 400:.0f} sessions." if total >= 400
                  else "The grid shows proportions across 400 squares.")
    cells = "".join(f'<span class="c {css}"></span>' * k for (css, _, _), k in zip(groups, squares))
    summary = ", ".join(f"{n:,} {label}" for _, label, n in groups)
    legend = "".join(f'<li><i class="c {css}"></i><span><b>{n:,}</b> {label}</span></li>'
                     for css, label, n in groups)
    return (f'<div class="grid" role="img" aria-label="Sessions: {summary}">{cells}</div>'
            f'<ul class="legend">{legend}</ul>'
            f'<div class="scale">{scale_note}</div>')


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
st.title("Shopper Intent Lab")
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

if has_labels:
    y = data[tm.TARGET].astype(int)
    m = tm.compute_metrics(y, pred, score)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tiles = "".join(f'<div><div class="num">{v:.4f}</div><div class="lbl">{k}</div></div>'
                    for k, v in m.items())
    st.markdown(
        f'<div class="headline">{tp:,} of {tp + fn:,} real buyers caught</div>'
        f'<div class="subline">The model missed {fn:,} buyers and flagged {fp:,} sessions that never bought.</div>'
        + session_grid([("tp", "buyers caught", tp), ("fn", "buyers missed", fn),
                        ("fp", "false alarms", fp), ("tn", "correctly left alone", tn)], len(data))
        + f'<div class="stats">{tiles}</div>',
        unsafe_allow_html=True)
else:
    flagged = int(pred.sum())
    st.markdown(
        f'<div class="headline">{flagged:,} of {len(pred):,} sessions flagged as likely buyers</div>'
        f'<div class="subline">Add a Revenue column to see how many of them were right.</div>'
        + session_grid([("tp", "flagged as buyers", flagged),
                        ("tn", "not flagged", len(pred) - flagged)], len(data)),
        unsafe_allow_html=True)

tabs = st.tabs(["Confusion matrix", "ROC curve", "Compare all models", "Predictions"])

# -------------------------------------------------------------- confusion matrix
with tabs[0]:
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
with tabs[1]:
    if has_labels:
        fpr, tpr, _ = roc_curve(y, score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2.5, color=BLUE, label=f"{model_name} (AUC = {m['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#8A94A8", label="Random guess")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("Add a `Revenue` column to draw the ROC curve.")

# ------------------------------------------------------------ compare all models
with tabs[2]:
    if has_labels:
        rows = []
        for name, mdl in models.items():
            p, s = mdl.predict(X), tm.get_scores(mdl, X)
            rows.append({"ML Model Name": name, **tm.compute_metrics(y, p, s)})
        comp = pd.DataFrame(rows).set_index("ML Model Name")
        st.dataframe(comp.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color: #A9BEFF; color: #000000; font-weight: 700;"),
                     width="stretch")
        st.caption("Shaded cells mark the best value in each column.")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        comp[["AUC", "F1", "MCC"]].plot.bar(ax=ax, rot=15, color=[BLUE, AMBER, INK])
        ax.set_ylim(0, 1.1)
        ax.legend(ncol=3, loc="upper center", frameon=False)
        ax.set_xlabel("")
        st.pyplot(fig)
    else:
        st.info("Add a `Revenue` column to compare the models.")

# ------------------------------------------------------------------ predictions
with tabs[3]:
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
