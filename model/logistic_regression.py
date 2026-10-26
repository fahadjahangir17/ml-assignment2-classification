"""Logistic Regression: linear model that outputs a purchase probability."""
from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"
FILE = "logistic_regression.pkl"

# balanced class weights because only about 16% of sessions are purchases
def build():
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

# C is the inverse of regularisation strength; smaller means a simpler model
PARAM_GRID = {"clf__C": [0.01, 0.1, 1, 10]}
