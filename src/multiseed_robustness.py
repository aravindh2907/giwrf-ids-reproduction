"""
multiseed_robustness.py
-------------------------
Reruns our best 17-feature configuration and the full 42-feature baseline
across 5 different Decision Tree random seeds, reporting mean +/- std for
each metric. This tests whether the 17-feature result's advantage is
robust or seed-dependent.

Run with:  python src/multiseed_robustness.py
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

SEEDS = [10, 1, 7, 42, 123]


def load_data():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_feature_sets():
    full_42 = None
    best_17 = ['sttl', 'ct_state_ttl', 'dload', 'rate', 'sbytes', 'ct_srv_dst',
               'smean', 'dmean', 'ct_dst_src_ltm', 'dbytes', 'state', 'proto',
               'tcprtt', 'dur', 'ct_srv_src', 'synack', 'dpkts']
    return {"full_42feat": full_42, "best_17feat_shap_pruned": best_17}


def make_dt(seed):
    return DecisionTreeClassifier(
        criterion="entropy", class_weight="balanced", random_state=seed,
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
    feature_sets = get_feature_sets()

    all_results = []
    for set_name, feats in feature_sets.items():
        Xtr = X_train[feats] if feats else X_train
        Xte = X_test[feats] if feats else X_test
        n_feat = Xtr.shape[1]

        print(f"\n--- {set_name} ({n_feat} features) ---")
        for seed in SEEDS:
            dt = make_dt(seed)
            dt.fit(Xtr, y_train)
            preds = dt.predict(Xte)
            metrics = evaluate(y_test, preds)
            metrics["feature_set"] = set_name
            metrics["n_features"] = n_feat
            metrics["seed"] = seed
            all_results.append(metrics)
            print(f"  seed={seed:<4} acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1']:.4f} fpr={metrics['fpr']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/tables/multiseed_robustness_raw.csv", index=False)

    summary = results_df.groupby("feature_set").agg(
        n_features=("n_features", "first"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
    ).reset_index()

    summary.to_csv("results/tables/multiseed_robustness_summary.csv", index=False)

    print("\n\n=== Multi-Seed Robustness Summary (mean +/- std across 5 seeds) ===")
    for _, row in summary.iterrows():
        print(f"\n{row['feature_set']} ({int(row['n_features'])} features):")
        print(f"  Accuracy:  {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")
        print(f"  Precision: {row['precision_mean']:.4f} +/- {row['precision_std']:.4f}")
        print(f"  Recall:    {row['recall_mean']:.4f} +/- {row['recall_std']:.4f}")
        print(f"  F1:        {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}")
        print(f"  FPR:       {row['fpr_mean']:.4f} +/- {row['fpr_std']:.4f}")

    print("\nSaved results/tables/multiseed_robustness_raw.csv")
    print("Saved results/tables/multiseed_robustness_summary.csv")

    pivot = results_df.pivot(index="seed", columns="feature_set", values="f1")
    pivot["17feat_wins"] = pivot["best_17feat_shap_pruned"] > pivot["full_42feat"]
    print("\n=== Per-seed F1 comparison (does 17-feat beat 42-feat every time?) ===")
    print(pivot.to_string())
    win_rate = pivot["17feat_wins"].sum()
    print(f"\n17-feature set beat 42-feature baseline in {win_rate}/{len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
