"""
shap_pruned_dt.py
-------------------
Follow-up to shap_explain.py: tests whether dropping the 5 features with
the lowest SHAP importance (despite high GIWRF rank) changes Decision
Tree performance on the test set. This directly probes the
multicollinearity hypothesis (e.g. dttl redundant given sttl).

Run with:  python src/shap_pruned_dt.py
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix)


def load_data():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "fpr": fp / (fp + tn),
    }


def make_dt():
    return DecisionTreeClassifier(
        criterion="entropy", class_weight="balanced", random_state=10,
        max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
        min_impurity_decrease=0.00006,
    )


def main():
    X_train, X_test, y_train, y_test = load_data()

    ranking = pd.read_csv("results/tables/shap_vs_giwrf_ranking.csv", index_col=0)

    full_26 = ranking.index.tolist()
    # Bottom 5 by SHAP value = least useful to the actual trained DT
    lowest_shap_5 = ranking.sort_values("shap_rank", ascending=False).head(5).index.tolist()
    pruned_21 = [f for f in full_26 if f not in lowest_shap_5]

    print(f"Original 26 features: {full_26}")
    print(f"\nDropping 5 lowest-SHAP-value features: {lowest_shap_5}")
    print(f"\nPruned set ({len(pruned_21)} features): {pruned_21}")

    results = []
    for name, feats in [("original_26feat_giwrf", full_26),
                        ("pruned_21feat_shap_informed", pruned_21)]:
        dt = make_dt()
        dt.fit(X_train[feats], y_train)
        preds = dt.predict(X_test[feats])
        metrics = evaluate(y_test, preds)
        metrics["condition"] = name
        metrics["n_features"] = len(feats)
        results.append(metrics)
        print(f"\n{name} ({len(feats)} features):")
        for k, v in metrics.items():
            if k not in ("condition", "n_features"):
                print(f"  {k}: {v:.4f}")

    results_df = pd.DataFrame(results)
    cols = ["condition", "n_features", "accuracy", "precision", "recall", "f1", "fpr"]
    results_df = results_df[cols]
    results_df.to_csv("results/tables/shap_pruned_comparison.csv", index=False)

    print("\n\n=== SHAP-Informed Pruning Comparison ===")
    print(results_df.to_string(index=False))
    print("\nSaved to results/tables/shap_pruned_comparison.csv")


if __name__ == "__main__":
    main()
