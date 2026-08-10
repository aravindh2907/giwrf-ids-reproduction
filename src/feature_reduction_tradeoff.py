"""
feature_reduction_tradeoff.py
-------------------------------
Deliberate feature-count sweep (not threshold-based) using GIWRF's Gini
importance ranking. Trains Decision Tree on the top-N features for
N in {42, 30, 25, 20, 15, 10, 5, 3, 1} and evaluates on the test set,
producing a full accuracy/F1/FPR vs n_features trade-off curve.

Run with:  python src/feature_reduction_tradeoff.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix)

N_VALUES = [42, 30, 25, 20, 15, 10, 5, 3, 1]


def load_data():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def load_ranking():
    importances = pd.read_csv("results/tables/giwrf_importances.csv", index_col=0)
    ranked_features = importances.sort_values("importance", ascending=False).index.tolist()
    return ranked_features


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


def run_sweep(X_train, X_test, y_train, y_test, ranked_features):
    results = []
    for n in N_VALUES:
        feats = ranked_features[:n]
        dt = make_dt()
        dt.fit(X_train[feats], y_train)
        preds = dt.predict(X_test[feats])
        metrics = evaluate(y_test, preds)
        metrics["n_features"] = n
        metrics["features"] = ", ".join(feats)
        results.append(metrics)
        print(f"n_features={n:<3} acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1']:.4f} fpr={metrics['fpr']:.4f}")

    return pd.DataFrame(results)


def plot_tradeoff(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(
        axes, ["accuracy", "f1", "fpr"],
        ["Accuracy", "F1 Score", "False Positive Rate"]
    ):
        ax.plot(results_df["n_features"], results_df[metric], marker="o")
        ax.set_xlabel("Number of Features (top-N by GIWRF importance)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Feature Count")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/feature_reduction_tradeoff.png", dpi=150)
    plt.close()
    print("\nSaved results/figures/feature_reduction_tradeoff.png")


def main():
    X_train, X_test, y_train, y_test = load_data()
    ranked_features = load_ranking()

    print(f"Full ranked feature list (top to bottom):\n{ranked_features}\n")
    print("--- Feature count sweep ---")

    results_df = run_sweep(X_train, X_test, y_train, y_test, ranked_features)

    cols = ["n_features", "accuracy", "precision", "recall", "f1", "fpr", "features"]
    results_df = results_df[cols]
    results_df.to_csv("results/tables/feature_reduction_tradeoff.csv", index=False)
    print("\nSaved results/tables/feature_reduction_tradeoff.csv")

    print("\n=== Full Trade-off Table ===")
    print(results_df.drop(columns=["features"]).to_string(index=False))

    plot_tradeoff(results_df)

    best_f1_row = results_df.loc[results_df["f1"].idxmax()]
    print(f"\nBest F1 configuration: n_features={int(best_f1_row['n_features'])}, "
          f"f1={best_f1_row['f1']:.4f}, accuracy={best_f1_row['accuracy']:.4f}, "
          f"fpr={best_f1_row['fpr']:.4f}")

    threshold = best_f1_row["f1"] - 0.01
    efficient_candidates = results_df[results_df["f1"] >= threshold]
    most_efficient = efficient_candidates.loc[efficient_candidates["n_features"].idxmin()]
    print(f"\nMost feature-efficient configuration within 1pp of best F1: "
          f"n_features={int(most_efficient['n_features'])}, "
          f"f1={most_efficient['f1']:.4f} "
          f"(vs best f1={best_f1_row['f1']:.4f} at n={int(best_f1_row['n_features'])})")


if __name__ == "__main__":
    main()
