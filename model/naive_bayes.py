"""Gaussian Naive Bayes: assumes features are independent given the class."""
from sklearn.naive_bayes import GaussianNB

NAME = "Naive Bayes"
FILE = "naive_bayes.pkl"


def build():
    return GaussianNB()


# var_smoothing adds a small variance floor so no feature gets an extreme likelihood
PARAM_GRID = {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}
