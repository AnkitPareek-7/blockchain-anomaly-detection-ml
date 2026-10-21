"""
Unsupervised anomaly detection on the Ethereum fraud dataset.

Models: Isolation Forest, PCA reconstruction error, Autoencoder.
Labels are never used for training, used only for evaluation (and to keep the
fraud ratio equal in the train/test split).

Each model is trained on all training accounts (a mix of legitimate and
fraudulent), and the experiment is repeated over several random splits.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from ethereum_common import SEEDS, load_data, split_and_preprocess, evaluate, summarize

# Label-free threshold: flag the top 5% highest-scoring training accounts
CONTAMINATION = 0.05

X, y = load_data()

# Autoencoder training log: epochs run before early stopping, per split
AE_LOG = []


# Models: each returns (train scores, test scores); higher = more anomalous

def run_isolation_forest(Xtr, Xte, seed):
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(Xtr)
    return -model.decision_function(Xtr), -model.decision_function(Xte)


def run_pca(Xtr, Xte, seed):
    pca = PCA(n_components=0.95, random_state=seed).fit(Xtr)

    def recon_error(A):
        A_hat = pca.inverse_transform(pca.transform(A))
        return np.mean((A - A_hat) ** 2, axis=1)

    return recon_error(Xtr), recon_error(Xte)


def run_autoencoder(Xtr, Xte, seed):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    d = Xtr.shape[1]

    inp = keras.Input(shape=(d,))
    x = keras.layers.Dense(64, activation="relu")(inp)
    z = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(z)
    out = keras.layers.Dense(d, activation="linear")(x)

    ae = keras.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    # Hold out 10% of the training rows for early stopping
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    n_val = max(1, int(0.1 * len(Xtr)))
    X_val, X_fit = Xtr[idx[:n_val]], Xtr[idx[n_val:]]

    history = ae.fit(
        X_fit, X_fit,
        validation_data=(X_val, X_val),
        epochs=50, batch_size=512, shuffle=True, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    val_loss = history.history["val_loss"]
    AE_LOG.append({"seed": seed,
                   "epochs_run": len(val_loss),
                   "best_epoch": int(np.argmin(val_loss)) + 1,
                   "best_val_loss": float(np.min(val_loss)),
                   "stopped_early": len(val_loss) < 50})
    print(f"   Autoencoder: {len(val_loss)} epochs run, "
          f"best epoch {int(np.argmin(val_loss)) + 1}")

    def recon_error(A):
        A_hat = ae.predict(A, batch_size=2048, verbose=0)
        return np.mean((A - A_hat) ** 2, axis=1)

    return recon_error(Xtr), recon_error(Xte)


MODELS = {
    "IsolationForest": run_isolation_forest,
    "PCA": run_pca,
    "Autoencoder": run_autoencoder,
}


# Run every model on every random split

rows = []
scores_first_seed = {}

for seed in SEEDS:
    Xtr, Xte, y_train, y_test, test_index = split_and_preprocess(X, y, seed)
    print(f"\n--- Seed {seed}: train {len(Xtr)} | test {len(Xte)} | "
          f"test fraud {y_test.mean():.4f} ---")

    for name, run in MODELS.items():
        print(f"Running {name}...")
        scores_train, scores_test = run(Xtr, Xte, seed)
        threshold = np.quantile(scores_train, 1 - CONTAMINATION)

        rows.append({"dataset": "Ethereum_Fraud_Detection", "paradigm": "unsupervised",
                     "seed": seed, "model": name,
                     **evaluate(y_test, scores_test, threshold)})

        if seed == SEEDS[0]:
            scores_first_seed[name] = scores_test

results = pd.DataFrame(rows)
summarize(results, "ethereum_unsupervised")

# Account-level scores of the first split (handy for ROC / PR plots)
first_split = split_and_preprocess(X, y, SEEDS[0])
score_table = pd.DataFrame({"original_row_index": first_split[4],
                            "actual_label": first_split[3]})
for name, scores in scores_first_seed.items():
    score_table[f"{name}_score"] = scores
score_table.to_csv("ethereum_unsupervised_scores.csv", index=False)
pd.DataFrame(AE_LOG).to_csv("ethereum_unsupervised_autoencoder_log.csv", index=False)
print("- ethereum_unsupervised_scores.csv (first split only)")
print("- ethereum_unsupervised_autoencoder_log.csv (epochs per split)")
