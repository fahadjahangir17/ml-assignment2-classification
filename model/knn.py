"""k-Nearest Neighbours: votes among the k most similar training sessions."""
from sklearn.neighbors import KNeighborsClassifier

NAME = "kNN"
FILE = "knn.pkl"


def build():
    # no class_weight option here, so this model sees the raw 84/16 split
    return KNeighborsClassifier()


PARAM_GRID = {"clf__n_neighbors": [5, 11, 21, 41], "clf__weights": ["uniform", "distance"]}
