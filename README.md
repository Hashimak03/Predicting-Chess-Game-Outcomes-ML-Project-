# Predicting Chess Game Outcomes from Pre-Game Features

CS-412 Machine Learning, University of Illinois Chicago (Fall 2026)
Application-track project by Hashim Khan (solo)

## Question

How well can machine learning models predict the outcome of an online chess game
from information available before the game starts (player ratings, time control,
opening), and do they beat the simple rule "the higher-rated player wins"?

## Data

[Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess) on Kaggle
(`games.csv`, 20,058 rows). The file is not included in this repository; download it
from Kaggle and place it next to the script.

After removing duplicates and rows with missing values, 19,113 games remain
(18,225 without draws).

**Leakage control.** Columns only known after the game ends (`victory_status`, `turns`,
`created_at`, `last_move_at`, `moves`) and player IDs are excluded from the features.

## Method

- Features: white/black rating, rating difference and mean, time control, rated flag,
  ECO opening letter and code, opening ply.
- Baselines: majority class; higher-rated player wins.
- Models (scikit-learn, mostly default settings): Logistic Regression, Decision Tree
  (depth 5), k-NN (k=25), Random Forest (200 trees), Gradient Boosting.
- Tasks: binary (White vs. Black, draws removed) and three-class (with draws).
- Evaluation: stratified 80/20 train/test split (seed 42), 5-fold cross-validation on
  the training set, accuracy and macro-F1.

## Results so far

No model meaningfully beats the rating-only baseline.

| Task | Baseline (higher-rated wins) | Best model (Gradient Boosting) |
|---|---|---|
| Binary, test accuracy | 65.2% | 65.5% |
| Three-class, test accuracy | 62.3% | 62.4% |

The differences are smaller than the cross-validation standard deviation (about 0.7-0.8
points). Rating difference carries most of the signal. Full tables are in
`out/results.csv` and the report.

## Run it

```bash
pip install -r requirements.txt
python chess_baseline.py games.csv out
```

This writes the tables, figures and LaTeX snippets to `out/`.

## Repository contents

- `chess_baseline.py`: full pipeline (cleaning, features, baselines, models, evaluation).
- `out/`: generated results (`results.csv`, LaTeX tables, figures).
- `intermediate_report.tex` / `.pdf`, `references.bib`: intermediate report (Nov 1).
- `proposal.tex` / `.pdf`: project proposal.

## Remaining work

Hyperparameter tuning, feature importance and ablations, early-move features
(first N moves), and a closer look at draw prediction. Final report due Dec 1, 2026.
