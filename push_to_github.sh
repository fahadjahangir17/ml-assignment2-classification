#!/usr/bin/env bash
# Run inside the unzipped ml-assignment2-classification folder, after creating an
# empty repo with that name on github.com/fahadjahangir17 (no README, no .gitignore).
# Commit in a few sittings if you can; the brief says commit history is reviewed.
set -e
git init -b main
git add data/ && git commit -m "Add Online Shoppers Purchasing Intention dataset from UCI"
git add model/train_models.py model/logistic_regression.py model/decision_tree.py model/knn.py model/naive_bayes.py model/svm.py && git commit -m "Add training script and one module per classifier"
git add model/*.pkl model/metrics.csv model/best_params.json model/ml_assignment2.ipynb test_data.csv && git commit -m "Add trained pipelines, metrics, executed notebook and hold-out test set"
git add app.py requirements.txt && git commit -m "Add Streamlit app with upload, model picker, metrics and confusion matrix"
git add README.md .gitignore && git commit -m "Add README with dataset notes, comparison table and observations"
git remote add origin https://github.com/fahadjahangir17/ml-assignment2-classification.git
git push -u origin main
