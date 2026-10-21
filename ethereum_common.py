"""
Shared helpers for the three Ethereum fraud-detection experiments
(unsupervised, semi-supervised and supervised).

Data loading, splitting, preprocessing and evaluation live here so that all
three paradigms see exactly the same data and the same train/test splits.
That is what makes the comparison in the paper fair.

Notes : 
  1. Duplicate accounts and identical feature rows are removed BEFORE
     splitting, so a test account can no longer have an exact copy in the
     training set.
  2. The split is stratified: train and test keep the same fraud ratio.
     (Labels are used only to balance the split, never to train the
     unsupervised models.)
  3. Every experiment is repeated over several random splits (SEEDS) and
     reported as mean +/- standard deviation.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Settings shared by all experiments

DATA_PATH = "transaction_dataset.csv"
TEST_SIZE = 0.30
SEEDS = [42, 43, 44, 45, 46]          # one random split per seed

# Metrics reported in the mean +/- std summary
METRICS = ["ROC_AUC", "AP", "Precision@k", "Precision", "Recall", "F1"]



# 1. Load, clean and de-duplicate

def load_data(path=DATA_PATH):
    """Return the feature table X and binary label y (1 = fraud)."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    target_col = next((c for c in df.columns if c.lower() == "flag"), None)
    if target_col is None:
        raise ValueError("Could not find the FLAG column.")

    df = df.dropna(axis=1, how="all")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    # Identifier columns are not behavioural features
    id_names = {"index", "address", "wallet", "wallet_address",
                "txid", "tx_id", "transaction_hash", "hash"}
    drop_cols = [
        c for c in df.columns
        if c != target_col
        and (c.lower() in id_names or c.lower().startswith("unnamed:"))
    ]
    address_col = next((c for c in df.columns if c.lower() == "address"), None)

    n_raw = len(df)

    # (a) the same wallet address listed more than once
    if address_col is not None:
        df = df.drop_duplicates(subset=address_col, keep="first")
    n_after_address = len(df)

    # (b) different addresses with exactly identical behaviour
    features = df.drop(columns=[target_col] + drop_cols)
    df = df.loc[~features.duplicated(keep="first")]

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    print("Rows in file:", n_raw)
    print("Removed duplicate addresses:", n_raw - n_after_address)
    print("Removed identical feature rows:", n_after_address - len(df))
    print("Rows kept:", len(df), "| Feature count:", X.shape[1])
    print("Legitimate accounts:", int((y == 0).sum()))
    print("Fraudulent accounts:", int((y == 1).sum()))
    print("Fraud prevalence:", round(y.mean(), 4))
    print("Removed identifier columns:", drop_cols)
    return X, y


# 2. Stratified split + preprocessing (fit on training data only)

def split_and_preprocess(X, y, seed):
    """
    Returns Xtr, Xte (processed arrays), y_train, y_test (numpy) and the
    original row index of the test accounts.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed,
        shuffle=True, stratify=y,
    )

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("quantile", QuantileTransformer(
            n_quantiles=min(1000, len(X_train)),
            output_distribution="normal",
            random_state=seed,
        )),
    ])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop",
                                     sparse_threshold=0.0)

    Xtr = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    Xte = np.asarray(preprocessor.transform(X_test), dtype=np.float32)

    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise ValueError("Non-finite values remain after preprocessing.")

    return Xtr, Xte, y_train.to_numpy(), y_test.to_numpy(), X_test.index



# 3. Evaluation (higher score = more likely fraud)

def evaluate(y_test, scores, threshold):
    """Ranking metrics use the raw scores; the rest use `scores >= threshold`."""
    prevalence = y_test.mean()
    pred = (scores >= threshold).astype(int)

    k = int(y_test.sum())
    top_k = np.argsort(-scores)[:k]

    ap = average_precision_score(y_test, scores)
    return {
        "ROC_AUC": roc_auc_score(y_test, scores),
        "AP": ap,
        "AP_lift_vs_random": ap / prevalence,
        "Precision@k": y_test[top_k].mean(),
        "Threshold": threshold,
        "Flagged_fraction": pred.mean(),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    }


# 4. Mean +/- std over the random splits

def summarize(results, prefix):
    """
    Save per-seed results and a mean/std summary, and print the summary.
    `results` has one row per (seed, model).
    """
    results.to_csv(f"{prefix}_results_by_seed.csv", index=False)

    grouped = results.groupby("model", sort=False)[METRICS]
    mean, std = grouped.mean(), grouped.std()      # sample std over seeds

    mean.join(std, lsuffix="_mean", rsuffix="_std").to_csv(
        f"{prefix}_results_summary.csv"
    )

    pretty = mean.round(4).astype(str) + " +/- " + std.round(4).astype(str)
    n_seeds = results["seed"].nunique()
    print(f"\n=== Mean +/- std over {n_seeds} random splits ===")
    print(pretty.to_string())

    print("\nSaved:")
    print(f"- {prefix}_results_by_seed.csv")
    print(f"- {prefix}_results_summary.csv")
