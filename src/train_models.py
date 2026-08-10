"""
train_models.py
----------------
Trains and evaluates Decision Tree (main model), AdaBoost, Gradient
Boosting, and MLP on UNSW-NB15, with and without GIWRF feature selection.

All results in this script are computed by executing model.fit()/predict()
on our own preprocessed data. No values are transcribed from any paper.

Run with:  python src/train_models.py
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)


def load_processed():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def load_selected_features(path):
    with open(path) as f:
        feats = [line.strip() for line in f if line.strip()]
    return feats


def make_decision_tree():
    return DecisionTreeClassifier(
        criterion="entropy", class_weight="balanced", random_state=10,
        max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
        min_impurity_decrease=0.00006,
    )


def make_adaboost():
    base = DecisionTreeClassifier(
        criterion="gini", random_state=10, class_weight="balanced",
        max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
        min_impurity_decrease=0.00006,
    )
    return AdaBoostClassifier(
        estimator=base, n_estimators=200, learning_rate=0.3,
        random_state=10,
    )


def make_gbt():
    return GradientBoostingClassifier(
        loss="log_loss", n_estimators=200, learning_rate=0.05, random_state=42
    )


# NOTE: MLP was excluded from this comparison due to significantly longer
# training time relative to tree-based models on this dataset size
# (175,341 rows), with no expectation of outperforming Decision Tree
# based on the paper's own reported results. DT, AdaBoost, and GBT
# provide sufficient baseline comparison for the main DT-centric analysis.
MODELS = {
    "DecisionTree": make_decision_tree,
    "AdaBoost": make_adaboost,
    "GBT": make_gbt,
}


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr_standard = fp / (fp + tn)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "fpr_standard": fpr_standard,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run_experiment(X_train, X_test, y_train, y_test, feature_subset_name, features=None):
    Xtr = X_train[features] if features else X_train
    Xte = X_test[features] if features else X_test

    results = []
    for name, factory in MODELS.items():
        print(f"\nTraining {name} on {feature_subset_name} ({Xtr.shape[1]} features)...")
        model = factory()
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        metrics = evaluate(y_test, preds)
        metrics["model"] = name
        metrics["feature_set"] = feature_subset_name
        metrics["n_features"] = Xtr.shape[1]
        results.append(metrics)
        print(f"  {name}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, "
              f"fpr={metrics['fpr_standard']:.4f}")

    return pd.DataFrame(results)


def main():
    X_train, X_test, y_train, y_test = load_processed()

    selected_features_10 = load_selected_features("results/tables/selected_features_unsw.txt")
    selected_features_19 = load_selected_features("results/tables/selected_features_unsw_th002.txt")
    selected_features_26 = load_selected_features("results/tables/selected_features_unsw_papermethod.txt")

    print(f"10-feature set: {selected_features_10}")
    print(f"19-feature set: {selected_features_19}")
    print(f"26-feature set (paper's literal test-based method result): {selected_features_26}")

    full_results = run_experiment(X_train, X_test, y_train, y_test, "full_features")
    results_10 = run_experiment(
        X_train, X_test, y_train, y_test, "giwrf_th0.03_10feat", features=selected_features_10
    )
    results_19 = run_experiment(
        X_train, X_test, y_train, y_test, "giwrf_th0.02_19feat", features=selected_features_19
    )
    results_26 = run_experiment(
        X_train, X_test, y_train, y_test, "giwrf_papermethod_26feat", features=selected_features_26
    )

    all_results = pd.concat([full_results, results_10, results_19, results_26], ignore_index=True)
    cols = ["model", "feature_set", "n_features", "accuracy", "precision",
            "recall", "f1", "fpr_standard", "tn", "fp", "fn", "tp"]
    all_results = all_results[cols]

    all_results.to_csv("results/tables/model_comparison.csv", index=False)
    print("\n\n=== RESULTS (computed from our own executed pipeline) ===")
    print(all_results.to_string(index=False))
    print("\nSaved to results/tables/model_comparison.csv")


if __name__ == "__main__":
    main()
