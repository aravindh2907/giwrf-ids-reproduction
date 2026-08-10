"""
shap_explain.py
----------------
SHAP explainability analysis on our best-performing GIWRF+DT configuration
(26 features, selected via paper's literal test-based threshold method).

Compares SHAP's feature ranking against GIWRF's Gini-importance ranking
to check whether GIWRF's selected features are also the most influential
by an independent, model-agnostic explainability method.

Run with:  python src/shap_explain.py
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def load_data_and_features():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    with open("results/tables/selected_features_unsw_papermethod.txt") as f:
        features = [line.strip() for line in f if line.strip()]

    return X_train[features], X_test[features], y_train, features


def train_best_dt(X_train, y_train):
    dt = DecisionTreeClassifier(
        criterion="entropy", class_weight="balanced", random_state=10,
        max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
        min_impurity_decrease=0.00006,
    )
    dt.fit(X_train, y_train)
    return dt


def compute_shap_values(model, X_test, sample_size=2000):
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("SHAP values computed on", X_sample.shape[0], "test samples")
    return explainer, shap_values, X_sample


def compare_rankings(shap_values, X_sample, features):
    import numpy as np

    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = attack
    else:
        sv = shap_values

    # Handle different returned array shapes from SHAP (robust to transposed shapes)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_classes, n_samples, n_features) -> select class 1
        sv = sv[1]
    if sv.ndim == 2 and sv.shape[0] == len(features) and sv.shape[1] != len(features):
        # shape is (n_features, n_samples) -> transpose
        sv = sv.T

    mean_abs_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=features
    ).sort_values(ascending=False)

    giwrf_importances = pd.read_csv("results/tables/giwrf_importances.csv", index_col=0)
    giwrf_ranking = giwrf_importances.loc[features, "importance"].sort_values(ascending=False)

    comparison = pd.DataFrame({
        "shap_rank": range(1, len(mean_abs_shap) + 1),
        "shap_mean_abs_value": mean_abs_shap.values,
    }, index=mean_abs_shap.index)

    comparison["giwrf_rank"] = comparison.index.map(
        lambda f: list(giwrf_ranking.index).index(f) + 1
    )
    comparison["giwrf_importance"] = comparison.index.map(giwrf_ranking)
    comparison["rank_difference"] = comparison["giwrf_rank"] - comparison["shap_rank"]

    print("\n=== SHAP vs GIWRF Feature Ranking Comparison ===")
    print(comparison.to_string())

    comparison.to_csv("results/tables/shap_vs_giwrf_ranking.csv")
    print("\nSaved to results/tables/shap_vs_giwrf_ranking.csv")

    from scipy.stats import spearmanr
    corr, pval = spearmanr(comparison["shap_rank"], comparison["giwrf_rank"])
    print(f"\nSpearman rank correlation (SHAP vs GIWRF): {corr:.4f} (p={pval:.4f})")

    return comparison


def plot_shap_summary(shap_values, X_sample):
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("results/figures/shap_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/shap_summary_plot.png")


def main():
    X_train, X_test, y_train, features = load_data_and_features()
    print(f"Training DT on {len(features)} features: {features}")

    dt = train_best_dt(X_train, y_train)
    explainer, shap_values, X_sample = compute_shap_values(dt, X_test)

    compare_rankings(shap_values, X_sample, features)
    plot_shap_summary(shap_values, X_sample)


if __name__ == "__main__":
    main()
