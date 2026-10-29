# Shopper Intent Lab: predicting online purchases with five classifiers

ML Assignment 2, M.Tech (DSE), BITS Pilani WILP. Live app: https://fahad-shopper-intent.streamlit.app

## Problem statement

An online shop wants to know, while a visitor is still browsing, whether the session will end in a purchase. If the model is good, the shop can show a discount or a live-chat prompt to the people who are on the fence and leave the confident buyers alone.

This is a binary classification task. Each row is one website session, and the label `Revenue` is 1 if the session ended in a purchase and 0 if it did not. I train five different classifiers on the same data, score them with six metrics on a held-out test set, and put them behind a Streamlit app so anyone can upload a CSV and compare them.

Only about 1 session in 6 ends in a purchase, so a model that always says "no purchase" would already score about 84% accuracy. Most of the analysis below is about not being fooled by that.

## Dataset description

Source: UCI Machine Learning Repository, "Online Shoppers Purchasing Intention Dataset" (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). Collected by Sakar, Polat, Katircioglu and Kastro (2019), Neural Computing and Applications 31, 6893-6908.

| Property | Value |
|---|---|
| Task | Binary classification (target: `Revenue`) |
| Rows in the raw file | 12,330 sessions |
| Rows after removing exact duplicates | 12,205 (125 duplicate rows dropped) |
| Features | 17 (10 numeric, 7 categorical or flag columns) |
| Missing values | None |
| Class balance | 10,297 no purchase, 1,908 purchase (15.6% positive) |

The 17 features are:

- Page-visit counts and time spent, per page type: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`.
- Google Analytics measures: `BounceRates`, `ExitRates`, `PageValues`.
- `SpecialDay`, a 0 to 1 score for how close the visit is to a shopping holiday such as Valentine's Day.
- `Month`, `VisitorType` (new, returning or other) and `Weekend`.
- Four integer codes for `OperatingSystems`, `Browser`, `Region` and `TrafficType`.

Preprocessing, all inside one scikit-learn `Pipeline` per model so an uploaded raw CSV can be scored directly:

- Exact duplicate rows are removed before splitting, so the same session cannot land in both train and test.
- Numeric columns are standardised with `StandardScaler`. `Month` and `VisitorType` are one-hot encoded.
- The four integer-coded columns (OS, browser, region, traffic type) are kept as plain numbers. Their codes are arbitrary labels, so this is a simplification I made to keep the feature count manageable. One-hot encoding them is a sensible next experiment.
- The split is 80/20, stratified on `Revenue`, with `random_state=42`. That gives 9,764 training rows and 2,441 test rows. The test rows are saved as `test_data.csv`.
- Hyper-parameters were chosen by 5-fold cross-validated grid search on the training rows only, scored by ROC AUC. The test set was touched once, for the final numbers.

## GitHub repository link

https://github.com/fahadjahangir17/ml-assignment2-classification

```
ml-assignment2-classification/
|-- app.py                  Streamlit app
|-- requirements.txt
|-- README.md
|-- test_data.csv           2,441-row hold-out set, raw columns plus Revenue
|-- data/
|   `-- online_shoppers_intention.csv   full dataset from UCI
`-- model/
    |-- train_models.py         trains, tunes and evaluates all five models
    |-- logistic_regression.py  one file per model: estimator and search grid
    |-- decision_tree.py
    |-- knn.py
    |-- naive_bayes.py
    |-- svm.py
    |-- ml_assignment2.ipynb    executed notebook: EDA, tuning, metrics, plots
    |-- *.pkl                   one fitted pipeline per model
    |-- metrics.csv             the comparison table below, as a file
    `-- best_params.json        hyper-parameters picked by cross-validation
```

To reproduce everything:

```
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

The notebook `model/ml_assignment2.ipynb` is self-contained (it needs only pandas, scikit-learn, matplotlib and joblib, checks in its first cell that the installed versions work together and repairs them if not, and finds the CSV on its own), so it runs top to bottom in a plain Jupyter Notebook. It repeats the same steps with plots and reproduces the numbers in `model/metrics.csv`.

## Models used

All five models see exactly the same training rows and are scored on the same 2,441 test rows. "Positive" means a purchase. Logistic Regression, Decision Tree and SVM use `class_weight="balanced"` because of the 84/16 split. kNN and Gaussian Naive Bayes have no such option, so they run unweighted.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8521 | 0.9110 | 0.5181 | 0.7880 | 0.6251 | 0.5562 |
| Decision Tree | 0.8562 | 0.9252 | 0.5255 | 0.8351 | 0.6451 | 0.5844 |
| kNN | 0.8775 | 0.8739 | 0.7943 | 0.2932 | 0.4283 | 0.4347 |
| Naive Bayes | 0.7218 | 0.8277 | 0.3370 | 0.8037 | 0.4749 | 0.3834 |
| Support Vector Machine | 0.8759 | 0.9143 | 0.5795 | 0.7539 | 0.6553 | 0.5887 |

Tuned settings: Logistic Regression C=0.1; Decision Tree max_depth=7 and min_samples_leaf=60; kNN k=41 with distance weighting; Naive Bayes var_smoothing=1e-3; SVM with an RBF kernel, C=1 and gamma=0.01.

Confusion-matrix counts on the test set (out of 382 real buyers and 2,059 non-buyers):

| Model | True positives | False negatives | False positives | True negatives |
|---|---|---|---|---|
| Logistic Regression | 301 | 81 | 280 | 1,779 |
| Decision Tree | 319 | 63 | 288 | 1,771 |
| kNN | 112 | 270 | 29 | 2,030 |
| Naive Bayes | 307 | 75 | 604 | 1,455 |
| Support Vector Machine | 288 | 94 | 209 | 1,850 |

### Observations on each model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A solid baseline. AUC of 0.911 and recall of 0.788 mean it ranks buyers well and catches about four in five of them. Precision is only 0.518, so roughly half of the sessions it flags never buy. The largest coefficient by far is `PageValues` (+2.21 on the scaled feature), followed by a November effect, and `ExitRates` pushes the prediction down. Training is instant and the coefficients are easy to explain, but a straight line only partly captures the step in the data: 56.3% of sessions with a positive `PageValues` buy, against 3.9% of sessions where it is zero. |
| Decision Tree | The best ranker (AUC 0.925) and the best at finding buyers (recall 0.835, only 63 missed). One feature does most of the work: `PageValues` carries 79% of the total importance, with `Month` and `ProductRelated_Duration` far behind. Depth 7 with at least 60 rows per leaf kept it from memorising the training set, since its cross-validated AUC (0.921) is close to its test AUC. The cost is precision of 0.526, with 288 false alarms. It is also the only model whose rules can be read out loud to a marketing team. |
| kNN | It posts the highest accuracy (0.878) and the highest precision (0.794) of the five, but both numbers are misleading. Recall is 0.293: it misses 270 of 382 buyers. With k=41 most neighbourhoods are dominated by non-buyers, and nothing in kNN corrects for that. Always predicting "no purchase" scores 84.4%, so 87.8% is a small gain for a model that fails at the job it was built for. It also has the largest saved file (2.3 MB) because it stores the whole training set. |
| Naive Bayes | The weakest model on almost every metric: lowest accuracy (0.722), AUC (0.828) and MCC (0.383), and 604 false positives. Its recall (0.804) is good, so it finds buyers, but it flags far too many other people along the way. The likely cause is the independence assumption. `BounceRates` and `ExitRates` have a correlation of 0.90, and `ProductRelated` against `ProductRelated_Duration` is 0.86, so the model counts the same evidence twice and becomes overconfident. It is cheap to train and works as a first benchmark, but I would not ship it. |
| Support Vector Machine | The best balance of the five. It has the top F1 (0.655) and the top MCC (0.589), the second-best accuracy (0.876) and the second-best AUC (0.914). Compared with the Decision Tree it misses 31 more buyers but raises 79 fewer false alarms (209 against 288), which is why precision improves to 0.580. The RBF boundary cannot be explained to a non-technical reader, and SVM training time grows quickly with the number of rows, which would matter on a bigger log. |
| Overall, winner for the dataset? | Support Vector Machine, by a very small margin over the Decision Tree. MCC is the metric I trust most on an 84/16 split because it uses all four confusion-matrix cells, and the SVM leads it (0.5887 against 0.5844) along with F1. The gap is only 0.004, so I would call it a near tie. If the shop cares most about not missing buyers, or wants rules it can read, the Decision Tree is the better pick: best AUC, best recall, and fully interpretable. kNN should not be used as it stands. |

## Streamlit app

Deployed on Streamlit Community Cloud from the `main` branch, file `app.py`. It has:

- A CSV upload box in the sidebar. With nothing uploaded, the app scores the bundled `test_data.csv` so it never opens empty.
- A drop-down to choose any of the five models.
- A grid of 400 squares that shows how many real buyers the model caught and missed and how many false alarms it raised, with the six metrics (Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient) shown beneath it.
- Four tabs under the grid: Confusion Matrix (with a full Classification Report), ROC Curve, Compare All Models and Predictions.
- Compare All Models is a side-by-side table of the five models on the uploaded data, with the best value in each column shaded, plus a bar chart of AUC Score, F1 Score and MCC Score.
- Predictions lists the first 200 rows with the model's output and has a button to download all predictions as a CSV.

If the uploaded file has no `Revenue` column (or only one class in it), the app still returns predictions and skips the metrics. A file with a missing feature column gets a clear error naming the column. If the saved `.pkl` files were built with a different scikit-learn version and cannot load, the app retrains the five models from `data/` on startup, so it does not crash on a version mismatch.
