"""
shap_prune_from_optimal20.py
------------------------------
Applies SHAP-based pruning starting from the TRUE-OPTIMAL top-20 feature
set (found via feature_reduction_tradeoff.py), rather than the paper's
26-feature set used in the earlier shap_pruned_dt.py experiment. Tests
whether SHAP can improve on an already-strong baseline, or whether n=20
is already at a local optimum with nothing further to prune profitably.

Run with:  python src/shap_prune_from_optimal20.py
"""

import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix)


def load_data():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def load_top20():
    importances = pd.read_csv("results/tables/giwrf_importances.csv", index_col=0)
    ranked = importances.sort_values("importance", ascending=False).index.tolist()
    return ranked[:20]


def make_dt():
    return DecisionTreeClassifier(
        criterion="entropy", class_weight="balanced", random_state=10,
        max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
        min_impurity_decrease=0.00006,
    )


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn),
    }


def main():
    X_train, X_test, y_train, y_test = load_data()
    top20 = load_top20()
    print(f"Top-20 features: {top20}")

    dt20 = make_dt()
    dt20.fit(X_train[top20], y_train)
    preds20 = dt20.predict(X_test[top20])
    metrics20 = evaluate(y_test, preds20)
    print(f"\nBaseline (top-20, no pruning): {metrics20}")

    X_sample = X_test[top20].sample(n=2000, random_state=42)
    explainer = shap.TreeExplainer(dt20)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[1]
    if sv.ndim == 2 and sv.shape[0] == len(top20) and sv.shape[1] != len(top20):
        sv = sv.T

    mean_abs_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=top20
    ).sort_values(ascending=False)
    print(f"\nSHAP ranking within top-20:\n{mean_abs_shap}")

    # Drop the 3 lowest-SHAP-value features from the top-20 set
    lowest3 = mean_abs_shap.tail(3).index.tolist()
    pruned17 = [f for f in top20 if f not in lowest3]
    print(f"\nDropping lowest-SHAP 3: {lowest3}")
    print(f"Pruned set ({len(pruned17)} features): {pruned17}")

    dt17 = make_dt()
    dt17.fit(X_train[pruned17], y_train)
    preds17 = dt17.predict(X_test[pruned17])
    metrics17 = evaluate(y_test, preds17)
    print(f"\nPruned (17 features): {metrics17}")

    results = pd.DataFrame([
        {"condition": "top20_unpruned", "n_features": 20, **metrics20},
        {"condition": "top20_shap_pruned_17feat", "n_features": 17, **metrics17},
    ])
    results.to_csv("results/tables/shap_prune_from_optimal20.csv", index=False)
    print("\n=== Comparison ===")
    print(results.to_string(index=False))
    print("\nSaved results/tables/shap_prune_from_optimal20.csv")


if __name__ == "__main__":
    main()
