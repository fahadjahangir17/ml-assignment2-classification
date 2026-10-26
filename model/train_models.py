"""
Trains five classifiers on the UCI Online Shoppers Purchasing Intention data
and saves each one as a single sklearn Pipeline (preprocessing + model).

Run it from the repo root:
    python model/train_models.py

What it writes:
    model/*.pkl             one fitted pipeline per model
    model/metrics.csv       the six test-set metrics for every model
    model/best_params.json  the hyper-parameters cross-validation picked
    test_data.csv           the 20% hold-out split, raw columns plus Revenue
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# one small file per model, all in this folder (model/)
import decision_tree
import knn
import logistic_regression
import naive_bayes
import svm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "online_shoppers_intention.csv")
MODEL_DIR = os.path.join(ROOT, "model")
TARGET = "Revenue"
RANDOM_STATE = 42

# Month and VisitorType are text, so they get one-hot encoded.
# Everything else is numeric and gets standardised.
CATEGORICAL = ["Month", "VisitorType"]
NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay",
           "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend"]

# Same column order as the original CSV (17 features, target excluded).
FEATURES = ["Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
            "OperatingSystems", "Browser", "Region", "TrafficType",
            "VisitorType", "Weekend"]

MODELS = [logistic_regression, decision_tree, knn, naive_bayes, svm]
MODEL_FILES = {m.NAME: m.FILE for m in MODELS}


def load_clean_data(path=DATA_PATH):
    """Read the CSV, drop exact duplicate sessions, turn the booleans into 0/1."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def candidate_models():
    """Each model with the small grid we search over. The 'clf__' prefix points
    at the classifier step inside the pipeline."""
    return {m.NAME: (m.build(), m.PARAM_GRID) for m in MODELS}


def get_scores(model, X):
    """Continuous score for the positive class, used for AUC and the ROC curve.
    The SVM has no predict_proba here, so it falls back to its decision function.
    AUC only cares about ranking, so the raw margin works fine."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_clean_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    # The Streamlit app uploads this file, so keep it raw (no scaling applied).
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    print(f"rows={len(df)}  train={len(X_train)}  test={len(X_test)}  "
          f"buyer rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, best_params = [], {}
    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        m = compute_metrics(y_test, y_pred, get_scores(best, X_test))
        m["CV AUC (train)"] = search.best_score_
        rows.append({"ML Model Name": name, **m})
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        joblib.dump(best, os.path.join(MODEL_DIR, MODEL_FILES[name]))
        print(f"{name:24s} {search.best_params_}  AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
