# CLAUDE.md - project context

CS-412 Machine Learning (UIC), Application track, solo project by Hashim Khan (beginner in ML).
Topic: predict Lichess chess game outcomes. Data: Kaggle "datasnaek/chess" (`games.csv`, 20,058 rows).

## Deadlines
- Intermediate report: done (Nov 1).
- Final presentation: 5-8 minutes, Monday Nov 23 (confirm the date). Must cover motivation, problem,
  intuition behind the methods, and algorithm + results (a demo is optional).
- Final report: Dec 1, 2026, 8-10 pages, ACM sigconf LaTeX (Overleaf). Needs Introduction, Problem
  description/methods, Results, Conclusions/future work.
- Code, proposal and reports live in GitHub with regular commits (repo:
  github.com/Hashimak03/Predicting-Chess-Game-Outcomes-ML-Project-).

## Research questions (from the proposal)
1. Do ML models beat the baseline "higher-rated player wins"?
2. Which features matter most (ratings, opening, time control, early moves)?
3. How does accuracy change as more of the game (first N moves) is revealed?

## What is done and found (Nov 1, `chess_baseline.py`)
- Cleaning: 20,058 -> 19,113 games (945 duplicates/missing removed). 49.9% White wins, 45.4% Black, 4.6% draws.
- Leakage control: excluded victory_status, turns, created_at, last_move_at, moves, player ids.
- Features: ratings (+ diff, mean), time control, rated, ECO opening letter/code (top 30), opening_ply.
- 80/20 stratified split (seed 42), 5-fold CV on train; models: Logistic Regression, Decision Tree (depth 5),
  k-NN (k=25), Random Forest (200), Gradient Boosting (defaults).
- Result: NO model meaningfully beats the rating rule.
  Binary: rule 65.2% vs best GB 65.5% (LogReg 65.0%, kNN 63.3%, RF 63.2%); CV std about 0.7-0.8 points.
  Three-class: rule 62.3% vs GB 62.4%; macro-F1 about 41-44%; draws are LIKELY rarely predicted (hypothesis only: per-class
  results were never checked; verifying it is part of the draw-handling work).
  Rating difference carries almost all the signal (White win rate 17.7% to 85.1% across rating-diff bins).

## `early_moves.py`: RUN on real data Nov 22 (results below, NOT yet interpreted)
Replays the first N half-moves (0, 10, 20, 30, 40) with the `chess` package, builds board
features (material, queens, captures, checks, castling), trains LogReg + HistGradientBoosting (with and without
ratings), and compares to the rating rule on the same test split. Only games with >= 40 half-moves are used for
every N (otherwise early-ending games leak the result). Writes out/early_moves_accuracy.png,
out/feature_importance.png, out/early_moves_results.csv, out/feature_importance.csv.
Run (on this Windows machine `python`/`python3` do not work; use `py`, Python 3.13):  py early_moves.py games.csv out

Run status:
- Ran with no crashes, no code bugs found. Packages installed from requirements.txt (pandas 3.1, scikit-learn 1.9,
  chess 1.11). Results are deterministic: a second run gave byte-identical CSVs.
- All 13,007 binary games with >= 40 half-moves replayed cleanly (checked: exactly 13,007 of the 18,225 have >= 40),
  so no games were silently dropped by the replay.
- Only code change: feature_importance.png title/x-label were cut off; shortened the text and saved with
  bbox_inches="tight". No change to any computation.
- "Gradient Boosting" in this script = HistGradientBoostingClassifier (the Nov 1 baseline used GradientBoostingClassifier).

Raw output (games used 13,007; train/test 10,405 / 2,602; test-set std. error about 0.9 points):
- Majority class 50.5%. Higher-rated-wins rule 62.9% on this subset (NOT 65.2%: subset = games lasting >= 20 full
  moves; say this on the slide).

| Full moves revealed | LogReg CV (test) | GB CV (test) | GB no ratings CV (test) |
|---|---|---|---|
| 0  | 63.7 +- 1.3 (62.5) | 62.9 +- 1.3 (62.8) | 49.6 +- 1.1 (49.7) |
| 5  | 63.6 +- 1.4 (62.7) | 62.6 +- 1.0 (63.1) | 51.3 +- 0.9 (52.1) |
| 10 | 65.1 +- 1.7 (64.0) | 64.2 +- 1.2 (64.1) | 54.0 +- 1.3 (55.7) |
| 15 | 67.5 +- 1.7 (65.8) | 66.8 +- 0.5 (66.3) | 61.0 +- 0.6 (62.1) |
| 20 | 71.8 +- 0.8 (71.2) | 71.3 +- 0.9 (70.7) | 67.2 +- 1.3 (68.4) |
(CV = 5-fold mean +- std on train, in %; parentheses = test accuracy %. Rule = 62.9% on the same test set.)

Grouped permutation importance, GB with ratings at 20 full moves (test-accuracy points lost when the group is
shuffled, 10 shuffles averaged): Material 12.53, Ratings 5.37, Time control + rated 0.32, Castling -0.01,
Captures + checks -0.13, Queens left -0.19. (Small negatives = no measurable importance, i.e. noise.)

## Interpretation of the early_moves results (agreed Nov 22)
- Q3: first 5-10 moves add essentially nothing (within the ~0.9 pt test noise). Gains appear at 15 moves (+3.4 over the
  rule) and 20 moves (+7.8, about 70.7% vs 62.9%), well beyond noise. Curve still rising at 20; not tested beyond 20.
- Q2: material balance is the most important feature group (12.5 pts), then ratings (5.4); time control ~0.3; others noise.
  Board-only model (no ratings) reaches 68.4% at 20 moves, beating the rating rule (62.9%).
- At N=0 the models tie the rule again (62.8% vs 62.9%), consistent with the Nov 1 finding.
- Caveats to state on slides: after move 5 this is in-game prediction, not pre-game; subset is games lasting >= 20 full moves
  (so the rule scores 62.9%, not 65.2%); single split; opening (ECO) features are not used in this script.
- Presentation story: ratings alone are hard to beat before the game -> material takes over by move 20 -> about 71% at move 20.

## Remaining (priority order)
1. (Done, see interpretation above.)
2. Build the 5-8 min presentation (about 6 slides: question, data + leakage, baseline + models,
   "everything ties" finding, accuracy-vs-moves chart, takeaway).
3. Later, for the Dec 1 report: hyperparameter tuning, draw handling (class weights, per-class metrics),
   optional larger Lichess sample, then write the 8-10 page report.

## Working rules
- Report results honestly. A tie with the baseline is a valid finding; never invent or round up numbers.
- Compare differences to the CV std / test-set standard error (about 1 point) before claiming a model is better.
- Keep leakage out of features. Do not edit games.csv. Do not upload games.csv to GitHub.
- Explain things simply: the user is new to ML.
