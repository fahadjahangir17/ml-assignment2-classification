"""Decision Tree: learns if/then splits, the easiest model here to read."""
from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"
FILE = "decision_tree.pkl"


def build():
    return DecisionTreeClassifier(class_weight="balanced", random_state=42)


# depth and leaf size are the two knobs that stop the tree memorising the training rows
PARAM_GRID = {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [10, 30, 60]}
