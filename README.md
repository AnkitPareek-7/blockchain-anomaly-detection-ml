# blockchain-anomaly-detection-ml
# Unsupervised vs Semi-supervised vs Supervised Anomaly Detection on Ethereum Fraud

Code for a comparative study of three learning paradigms for fraud / anomaly
detection on Ethereum accounts. All three experiments use the same cleaned
data, the same train/test splits and the same preprocessing, so the results
can be compared directly.

## Dataset

Kaggle *Ethereum Fraud Detection* dataset (`transaction_dataset.csv`):
9,841 accounts, 47 behavioural features, label `FLAG` (1 = fraud, 0 = legitimate).

> Add the dataset link and citation here, and check its licence before
> redistributing the file. `transaction_dataset.csv` is listed in `.gitignore`;
> download it and place it next to the scripts.

## Repository layout

| File | Purpose |
|---|---|
| `ethereum_common.py` | Shared helpers: loading, de-duplication, stratified split, preprocessing, evaluation, mean ± std summary |
| `ethereum_unsupervised.py` | Isolation Forest, PCA reconstruction error, Autoencoder (no labels used for training) |
| `ethereum_semisupervised.py` | Setting A: models trained on legitimate accounts only. Setting B: self-training Random Forest with 5 / 10 / 20 % of the labels |
| `ethereum_supervised.py` | Random Forest and Gradient Boosting trained on all training labels |
| `requirements.txt` | Minimum library versions |

## Setup and run

```bash
pip install -r requirements.txt
# put transaction_dataset.csv in the same folder, then:
python ethereum_unsupervised.py
python ethereum_semisupervised.py
python ethereum_supervised.py
```

Each script writes (with its own prefix, e.g. `ethereum_supervised_`):

- `*_results_by_seed.csv` - every metric for every split and model
- `*_results_summary.csv` - mean and standard deviation over the splits
- `*_scores.csv` - account-level scores of the first split (for ROC / PR plots)
- `*_autoencoder_log.csv` - epochs run per split (unsupervised and semi-supervised only)

## Methodology

**Cleaning.** Identifier columns (`Index`, `Address`, unnamed index) are removed.
Rows with a repeated address (25) and rows with identical feature values (528)
are dropped *before* splitting, leaving 9,288 accounts (7,632 legitimate,
1,656 fraudulent; 17.8 % fraud). Without this, copies of the same account can
land in both train and test and inflate the scores.

**Splits.** Stratified 70 / 30 train/test split, repeated for seeds 42-46.
Labels are used only to keep the fraud ratio equal in train and test; the
unsupervised models never train on them. Results are reported as mean ± sample
standard deviation over the five splits.

**Preprocessing** (fit on the training part only). Numeric features: median
imputation and a quantile transform to a normal distribution. Categorical
features: most-frequent imputation and one-hot encoding.

**Models and thresholds.**

| Paradigm | Model | Training data | Decision rule |
|---|---|---|---|
| Unsupervised | Isolation Forest, PCA, Autoencoder | all training accounts, no labels | flag the top 5 % of training scores |
| Semi-supervised A | same three models | legitimate training accounts only | flag above the 95th percentile of normal-training scores |
| Semi-supervised B | Self-training Random Forest | 5 / 10 / 20 % of labels + the rest unlabeled | fraud probability >= 0.5 |
| Supervised | Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`) | all training labels | fraud probability >= 0.5 |

**Metrics.** ROC-AUC, Average Precision (AP) and Precision@k (k = number of
frauds in the test set) use the raw scores and do not depend on a threshold.
Precision, Recall and F1 depend on the decision rule above, which differs
between paradigms, so **compare paradigms mainly on ROC-AUC, AP and
Precision@k**. The random-guess AP is the fraud rate, 0.178.

## Results (mean ± std over 5 random splits)

| Paradigm | Model | ROC-AUC | AP | F1 |
|---|---|---|---|---|
| Unsupervised | Isolation Forest | 0.678 ± 0.016 | 0.310 ± 0.022 | 0.219 ± 0.035 |
| Unsupervised | PCA | 0.700 ± 0.051 | 0.297 ± 0.052 | 0.114 ± 0.010 |
| Unsupervised | Autoencoder | 0.728 ± 0.067 | 0.302 ± 0.054 | 0.113 ± 0.021 |
| Semi-supervised A | Isolation Forest | 0.657 ± 0.018 | 0.289 ± 0.015 | 0.212 ± 0.031 |
| Semi-supervised A | PCA | 0.830 ± 0.052 | 0.421 ± 0.076 | 0.250 ± 0.061 |
| Semi-supervised A | Autoencoder | 0.853 ± 0.044 | 0.455 ± 0.064 | 0.268 ± 0.061 |
| Semi-supervised B | Self-training RF, 5 % labels | 0.967 ± 0.017 | 0.922 ± 0.038 | 0.800 ± 0.018 |
| Semi-supervised B | Self-training RF, 10 % labels | 0.979 ± 0.010 | 0.935 ± 0.031 | 0.836 ± 0.035 |
| Semi-supervised B | Self-training RF, 20 % labels | 0.987 ± 0.008 | 0.958 ± 0.026 | 0.889 ± 0.038 |
| Supervised | Random Forest | 0.994 ± 0.004 | 0.981 ± 0.014 | 0.939 ± 0.037 |
| Supervised | Gradient Boosting | 0.995 ± 0.003 | 0.985 ± 0.010 | 0.950 ± 0.033 |

The gap between Random Forest and Gradient Boosting is smaller than one
standard deviation, so no ranking between them should be claimed. The
differences between paradigms are much larger than the split-to-split variation.

## Limitations and notes

- **Label source.** Fraud labels come from a curated list of known fraud
  addresses and the legitimate accounts were collected separately, so a model
  may partly learn how the dataset was assembled. Scores on this dataset should
  not be read as expected performance on live Ethereum traffic.
- **Random, not temporal, split.** Accounts are split at random. A split by
  account creation date would be a harder and more realistic test.
- **Hyperparameters are fixed defaults** (for example 300 trees, 5 % flag rate,
  0.9 self-training confidence) and were not tuned. If you tune, use a separate
  validation split and never the test set.
- **Autoencoder epoch budget.** The autoencoder trains for at most 50 epochs
  with early stopping (patience 5). Early stopping never triggered: validation
  loss was still decreasing at epoch 50 in every run (see the
  `*_autoencoder_log.csv` files). The 50-epoch budget is therefore a fixed
  training budget, not a convergence point. Its results are sensitive to this
  choice: in a one-off check (not part of these scripts), allowing up to 300
  epochs lowered the unsupervised autoencoder's mean ROC-AUC from 0.728 to
  0.640 and its AP from 0.302 to 0.227, because a longer-trained autoencoder
  also learns to reconstruct fraudulent accounts. Report the 50-epoch budget
  explicitly.
- **Normal-only training uses labels** to decide which training rows are
  legitimate, which is what makes Setting A semi-supervised.

## Reproducibility

Results were produced with Python 3.13.16, numpy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.1 and TensorFlow 2.21.0 (CPU). Re-running the scripts in that
environment reproduced the summary tables exactly. Different library versions
or hardware can change TensorFlow results slightly.
