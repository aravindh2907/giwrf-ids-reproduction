"""
giwrf.py
--------
Gini Impurity-based Weighted Random Forest (GIWRF) feature selection,
reproducing Eq. 3-6 of Disha & Waheed (2022).

Loads the processed train/test data saved by preprocessing.py, computes
imbalance-aware class weights (Eq. 5-6), fits a Random Forest with Gini
splitting, ranks features by importance, and sweeps thresholds to find
the paper's target ~20-feature subset for UNSW-NB15.

Run with:  python src/giwrf.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_processed():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    print("Loaded processed data:")
    print("  X_train:", X_train.shape, "| X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test


def compute_class_weights(y):
    """
    Eq. 5-6: w_p = n_n / N (weight for positive/attack class),
             w_n = n_p / N (weight for negative/normal class)
    """
    n_p = (y == 1).sum()
    n_n = (y == 0).sum()
    N = len(y)
    w_p = n_n / N
    w_n = n_p / N
    print(f"\nClass counts: n_p (attack)={n_p}, n_n (normal)={n_n}, N={N}")
    print(f"Computed weights (Eq. 5-6): w_p={w_p:.4f}, w_n={w_n:.4f}")
    return {1: w_p, 0: w_n}


def fit_giwrf(X_train, y_train, n_estimators=500, random_state=42):
    """Random Forest with Gini splitting + imbalance-aware weights."""
    class_weights = compute_class_weights(y_train)

    rf = RandomForestClassifier(
        criterion="gini",
        n_estimators=n_estimators,
        class_weight=class_weights,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    print("\nTop 15 features by Gini importance:")
    print(importances.head(15))

    return rf, importances


def threshold_sweep(X_tr, y_tr, X_val, y_val, importances,
                   thresholds=(0.01, 0.02, 0.03, 0.05, 0.07, 0.10)):
    """
    Reproduces Fig. 8. IMPORTANT: uses a VALIDATION split carved out of
    the training set (NOT the test set) to select the threshold. This
    avoids the leakage issue in the paper's original methodology, where
    the threshold appears to be chosen using test-set accuracy.
    """
    results = []
    print("\n--- Threshold sweep (validation-based, leakage-free) ---")
    for th in thresholds:
        feats = importances[importances >= th].index.tolist()
        if len(feats) == 0:
            continue
        dt = DecisionTreeClassifier(
            criterion="entropy", class_weight="balanced", random_state=10,
            max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
            min_impurity_decrease=0.00006,
        )
        dt.fit(X_tr[feats], y_tr)
        preds = dt.predict(X_val[feats])
        acc = accuracy_score(y_val, preds)
        results.append({"threshold": th, "n_features": len(feats), "val_accuracy": acc})
        print(f"  threshold={th:<5} n_features={len(feats):<3} val_accuracy={acc:.4f}")

    return pd.DataFrame(results)


def paper_method_threshold_sweep(X_train, y_train, X_test, y_test, importances,
                                  thresholds=(0.01, 0.02, 0.03, 0.05, 0.07, 0.10)):
    """
    Replicates the paper's apparent methodology (Fig. 8): select the
    importance threshold using TEST SET accuracy directly. This is
    kept separate from threshold_sweep() (our leakage-free version)
    specifically to demonstrate and quantify the effect of this
    methodological difference. Results from this function should be
    interpreted as "what the paper's literal method would have picked
    here" - not as our recommended approach.
    """
    results = []
    print("\n--- Threshold sweep (paper's literal method: test-set based) ---")
    for th in thresholds:
        feats = importances[importances >= th].index.tolist()
        if len(feats) == 0:
            continue
        dt = DecisionTreeClassifier(
            criterion="entropy", class_weight="balanced", random_state=10,
            max_depth=11, max_leaf_nodes=162, min_samples_leaf=20,
            min_impurity_decrease=0.00006,
        )
        dt.fit(X_train[feats], y_train)
        preds = dt.predict(X_test[feats])
        acc = accuracy_score(y_test, preds)
        results.append({"threshold": th, "n_features": len(feats), "test_accuracy": acc})
        print(f"  threshold={th:<5} n_features={len(feats):<3} test_accuracy={acc:.4f}")

    return pd.DataFrame(results)


def main():
    X_train, X_test, y_train, y_test = load_processed()

    rf, importances = fit_giwrf(X_train, y_train)
    importances.to_csv("results/tables/giwrf_importances.csv", header=["importance"])
    print("\nSaved full importance ranking to results/tables/giwrf_importances.csv")

    # carve a validation split out of TRAIN ONLY for threshold selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    print(f"\nValidation split for threshold search: X_tr={X_tr.shape}, X_val={X_val.shape}")

    sweep_results = threshold_sweep(X_tr, y_tr, X_val, y_val, importances)
    sweep_results.to_csv("results/tables/threshold_sweep.csv", index=False)
    print("\nSaved threshold sweep results to results/tables/threshold_sweep.csv")

    # COMPARISON NOTE:
    # - threshold_sweep() above uses a validation split (leakage-free).
    # - paper_method_threshold_sweep() below uses the test set directly,
    #   replicating the paper's apparent methodology.
    # Comparing the two selected thresholds/feature counts and their
    # eventual downstream model performance quantifies the practical
    # effect of this methodological difference.

    # Run the paper's literal (test-set-based) threshold sweep for comparison
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    paper_sweep_results = paper_method_threshold_sweep(
        X_train, y_train, X_test, y_test, importances
    )
    paper_sweep_results.to_csv("results/tables/paper_method_threshold_sweep.csv", index=False)
    print("\nSaved paper-method threshold sweep to results/tables/paper_method_threshold_sweep.csv")

    best_paper_row = paper_sweep_results.loc[paper_sweep_results["test_accuracy"].idxmax()]
    print(f"\nBest threshold via paper's literal method: {best_paper_row['threshold']} "
          f"-> {int(best_paper_row['n_features'])} features, "
          f"test_accuracy={best_paper_row['test_accuracy']:.4f}")

    paper_method_features = importances[importances >= best_paper_row["threshold"]].index.tolist()
    with open("results/tables/selected_features_unsw_papermethod.txt", "w") as f:
        f.write("\n".join(paper_method_features))
    print(f"Saved paper-method selected feature list "
          f"({len(paper_method_features)} features) to "
          f"results/tables/selected_features_unsw_papermethod.txt")

    best_row = sweep_results.loc[sweep_results["val_accuracy"].idxmax()]
    print(f"\nBest threshold on validation set: {best_row['threshold']} "
          f"-> {int(best_row['n_features'])} features, "
          f"val_accuracy={best_row['val_accuracy']:.4f}")
    print("Paper reports optimal threshold = 0.02 for UNSW-NB15, giving 20 features.")

    selected_features = importances[importances >= best_row["threshold"]].index.tolist()
    print(f"\nSelected {len(selected_features)} features:")
    print(selected_features)

    with open("results/tables/selected_features_unsw.txt", "w") as f:
        f.write("\n".join(selected_features))

    # Also save the threshold=0.02 feature set explicitly (paper's chosen
    # threshold) for side-by-side comparison against our validation-selected
    # threshold=0.03 set.
    features_at_002 = importances[importances >= 0.02].index.tolist()
    with open("results/tables/selected_features_unsw_th002.txt", "w") as f:
        f.write("\n".join(features_at_002))
    print(f"\nAlso saved threshold=0.02 feature set ({len(features_at_002)} features) "
          f"to results/tables/selected_features_unsw_th002.txt for comparison")

    print("\nSaved selected feature list to results/tables/selected_features_unsw.txt")


if __name__ == "__main__":
    main()
