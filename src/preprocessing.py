"""
preprocessing.py
-----------------
Preprocessing pipeline for UNSW-NB15 reproduction of Disha & Waheed (2022).

Steps:
  1. Load train/test CSVs (filenames are swapped relative to content -
     verified via row count and class distribution match against the paper).
  2. Drop id, attack_cat; isolate binary label as target.
  3. Response Coding for categorical columns (Eq. 1), fit on train only.
  4. Min-Max scaling (Eq. 2), fit on train only.
  5. Save processed arrays to data/processed/ for reuse by later scripts.

Run with:  python src/preprocessing.py
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CAT_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat"]
TARGET_COL = "label"

TRAIN_PATH = "data/raw/UNSW_NB15_testing-set.csv"   # actually TRAIN (175,341 rows)
TEST_PATH = "data/raw/UNSW_NB15_training-set.csv"   # actually TEST (82,332 rows)


class ResponseCoder:
    """
    Implements Eq. 1 of the paper: P(Y|A) = P(A ∩ Y) / P(A)
    For binary classification, for each category value v of a categorical
    column, this is the empirical P(label=1 | value), estimated ONLY from
    the training set.

    Unseen categories at test time fall back to the global training prior
    P(label=1) - NOT specified in the paper, documented assumption.
    """

    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.mappings_ = {}
        self.global_prior_ = None

    def fit(self, X, y):
        self.global_prior_ = y.mean()
        for col in self.cat_cols:
            self.mappings_[col] = y.groupby(X[col]).mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols:
            X[col] = X[col].map(self.mappings_[col]).fillna(self.global_prior_)
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print("Train shape:", train.shape, "(expect 175341, 45)")
    print("Test shape:", test.shape, "(expect 82332, 45)")
    return train, test


def split_X_y(df):
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def encode_categoricals(X_train, y_train, X_test):
    coder = ResponseCoder(CAT_COLS)
    X_train_enc = coder.fit_transform(X_train, y_train)
    X_test_enc = coder.transform(X_test)

    print("\n--- Response Coding check ---")
    print("Nulls after encoding - train:", X_train_enc.isnull().sum().sum())
    print("Nulls after encoding - test:", X_test_enc.isnull().sum().sum())
    for col in CAT_COLS:
        unseen = set(X_test[col].unique()) - set(coder.mappings_[col].keys())
        print(f"  {col}: {len(unseen)} unseen category values in test "
              f"(fell back to global prior {coder.global_prior_:.4f})")

    return X_train_enc, X_test_enc, coder


def scale_features(X_train_enc, X_test_enc):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_enc),
        columns=X_train_enc.columns, index=X_train_enc.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_enc),
        columns=X_test_enc.columns, index=X_test_enc.index
    )

    print("\n--- Min-Max Scaling check ---")
    print("Train min/max (first 5 cols):")
    print(X_train_scaled.describe().loc[["min", "max"]].T.head())
    print("\nTest min/max (first 5 cols, may exceed 0/1 slightly - expected):")
    print(X_test_scaled.describe().loc[["min", "max"]].T.head())

    return X_train_scaled, X_test_scaled, scaler


def save_processed(X_train_scaled, X_test_scaled, y_train, y_test):
    os.makedirs("data/processed", exist_ok=True)
    X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("\nSaved processed data to data/processed/")


def main():
    train, test = load_data()
    X_train, y_train = split_X_y(train)
    X_test, y_test = split_X_y(test)

    print("\nX_train shape:", X_train.shape, "(expect 175341, 42)")
    print("X_test shape:", X_test.shape, "(expect 82332, 42)")

    X_train_enc, X_test_enc, coder = encode_categoricals(X_train, y_train, X_test)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_enc, X_test_enc)
    save_processed(X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == "__main__":
    main()
