"""Support Vector Machine with an RBF kernel."""
from sklearn.svm import SVC

NAME = "Support Vector Machine"
FILE = "svm.pkl"


def build():
    # no probability=True: AUC is computed from decision_function instead (see get_scores)
    return SVC(kernel="rbf", class_weight="balanced", random_state=42)


PARAM_GRID = {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01]}
