"""CS-412 project, part 2: how much do the first moves add beyond the ratings?

Usage:   python early_moves.py games.csv out
Needs:   pip install chess

What it does
  1. Cleans the data exactly like chess_baseline.py (binary task: White vs. Black, no draws).
  2. Replays the first N half-moves of every game with python-chess and builds board features
     (material balance, queens left, captures, checks, castling) at N = 0, 10, 20, 30, 40.
  3. Trains models on ratings + time control + those board features, for each N.
  4. Writes the accuracy-vs-N chart, a feature-importance chart and CSV files to out/.

Fair-comparison rules
  * Only games with at least MAX_PLY half-moves are used, for EVERY N. Otherwise a game that
    ended early (e.g. by checkmate) would leak its result into the features.
  * The same train/test split is used for every N, and the rating-only baseline is
    evaluated on that same test set.
  * Opening-name features are NOT used here; the moves themselves replace them.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import chess
except ImportError:
    sys.exit("ERROR: run  pip install chess  first")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SNAPSHOTS = [0, 10, 20, 30, 40]          # half-moves (plies): 0, 5, 10, 15, 20 full moves
MAX_PLY = max(SNAPSHOTS)
path = sys.argv[1] if len(sys.argv) > 1 else "games.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------- load + clean (same as chess_baseline.py)
df = pd.read_csv(path)
print("Loaded", path, "shape:", df.shape)
need = ["winner", "white_rating", "black_rating", "increment_code", "rated", "moves"]
absent = [c for c in need if c not in df.columns]
if absent:
    sys.exit(f"ERROR: expected columns missing from the CSV: {absent}")
df = df.drop_duplicates(subset="id") if "id" in df.columns else df.drop_duplicates()
df = df[df["winner"].isin(["white", "black"])].copy()          # binary task: no draws
tc = df["increment_code"].astype(str).str.extract(r"(\d+)\+(\d+)").astype(float)
df["base_min"], df["inc_sec"] = tc[0], tc[1]
df["rated"] = df["rated"].astype(str).str.lower().eq("true").astype(int)
df = df.dropna(subset=["base_min", "inc_sec", "white_rating", "black_rating", "moves"]).copy()
df["rating_diff"] = df["white_rating"] - df["black_rating"]
df["rating_mean"] = (df["white_rating"] + df["black_rating"]) / 2
print("Binary games after cleaning:", len(df))

# ---------------------------------------------------------------- replay the first moves
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def board_features(board, captures, checks, castled_w, castled_b):
    w = b = wp = bp = queens = 0
    for piece in board.piece_map().values():
        if piece.piece_type == chess.KING:
            continue
        v = VALUE[piece.piece_type]
        if piece.color == chess.WHITE:
            w += v
            wp += piece.piece_type == chess.PAWN
        else:
            b += v
            bp += piece.piece_type == chess.PAWN
        queens += piece.piece_type == chess.QUEEN
    return {"mat_diff": w - b, "white_mat": w, "black_mat": b, "pawn_diff": wp - bp,
            "queens_left": queens, "captures": captures, "checks": checks,
            "castled_w": castled_w, "castled_b": castled_b}


def replay(moves_str):
    """Return {N: feature dict} for each snapshot N, or None if the game is too short / unreadable."""
    toks = str(moves_str).split()
    if len(toks) < MAX_PLY:
        return None
    board = chess.Board()
    captures = checks = castled_w = castled_b = 0
    snaps = {0: board_features(board, 0, 0, 0, 0)}
    try:
        for ply, san in enumerate(toks[:MAX_PLY], start=1):
            move = board.parse_san(san)
            captures += board.is_capture(move)
            if san.startswith("O-O"):
                if ply % 2 == 1:
                    castled_w = 1
                else:
                    castled_b = 1
            board.push(move)
            checks += board.is_check()
            if ply in SNAPSHOTS:
                snaps[ply] = board_features(board, captures, checks, castled_w, castled_b)
    except Exception:
        return None
    return snaps


print(f"Replaying the first {MAX_PLY} half-moves of every game (about a minute)...")
rows, keep = [], []
for idx, mv in df["moves"].items():
    s = replay(mv)
    if s is not None:
        rows.append(s)
        keep.append(idx)
df = df.loc[keep].copy()
print(f"Games with at least {MAX_PLY} half-moves that replayed cleanly: {len(df)}")
if len(df) < 2000:
    sys.exit("ERROR: too few games left; lower MAX_PLY / SNAPSHOTS")
FEATS = list(rows[0][0].keys())
snap_df = {N: pd.DataFrame([r[N] for r in rows], index=df.index) for N in SNAPSHOTS}

# ---------------------------------------------------------------- fixed split for every N
y = df["winner"]
tr_idx, te_idx = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
ytr, yte = y.loc[tr_idx], y.loc[te_idx]
rule = np.where(df.loc[te_idx, "rating_diff"] >= 0, "white", "black")
base_acc = accuracy_score(yte, rule)
maj_acc = (yte == ytr.value_counts().idxmax()).mean()
print(f"\nTrain {len(tr_idx)}, test {len(te_idx)}")
print(f"Majority class: {100*maj_acc:.1f}%   Higher-rated-wins rule: {100*base_acc:.1f}%")

RATING = ["white_rating", "black_rating", "rating_diff", "rating_mean"]
OTHER = ["base_min", "inc_sec", "rated"]


def make_X(N, with_ratings):
    cols = (RATING if with_ratings else []) + OTHER
    X = df[cols].copy()
    if N > 0:
        X = X.join(snap_df[N])
    return X


def build(kind, X):
    if kind == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000)
        prep = ColumnTransformer([("num", StandardScaler(), list(X.columns))])
    else:
        clf = HistGradientBoostingClassifier(random_state=SEED)
        prep = "passthrough"
    return Pipeline([("prep", prep), ("clf", clf)])


SETUPS = [("Logistic Regression", True), ("Gradient Boosting", True),
          ("Gradient Boosting, no ratings", False)]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
res = []
for N in SNAPSHOTS:
    for name, with_r in SETUPS:
        X = make_X(N, with_r)
        pipe = build(name.split(",")[0], X)
        cvs = cross_val_score(pipe, X.loc[tr_idx], ytr, cv=cv, scoring="accuracy")
        pipe.fit(X.loc[tr_idx], ytr)
        acc = accuracy_score(yte, pipe.predict(X.loc[te_idx]))
        res.append({"N_halfmoves": N, "N_fullmoves": N / 2, "model": name,
                    "cv_mean": cvs.mean(), "cv_std": cvs.std(), "test_acc": acc})
        print(f"N={N:>2} plies | {name:<30} CV {100*cvs.mean():.1f} +- {100*cvs.std():.1f} | test {100*acc:.1f}")
res = pd.DataFrame(res)
res["baseline_rule_acc"] = base_acc
res.to_csv(f"{out}/early_moves_results.csv", index=False)

# ---------------------------------------------------------------- chart: accuracy vs. moves revealed
n_te = len(te_idx)
plt.figure(figsize=(5.2, 3.4))
for name, color in [("Logistic Regression", "#2b6cb0"), ("Gradient Boosting", "#c05621"),
                    ("Gradient Boosting, no ratings", "#2f855a")]:
    r = res[res["model"] == name]
    se = np.sqrt(r["test_acc"] * (1 - r["test_acc"]) / n_te)
    plt.errorbar(r["N_fullmoves"], 100 * r["test_acc"], yerr=100 * se, marker="o",
                 capsize=2, label=name, color=color)
plt.axhline(100 * base_acc, color="gray", ls="--", lw=1, label="Higher-rated wins")
plt.xlabel("Moves revealed (full moves)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs. amount of game revealed", fontsize=10)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{out}/early_moves_accuracy.png", dpi=200)
plt.close()

# ---------------------------------------------------------------- feature importance at the largest N
N = MAX_PLY
X = make_X(N, True)
gb = build("Gradient Boosting", X).fit(X.loc[tr_idx], ytr)
# Grouped permutation importance: the four rating columns are near-copies of each other, so
# shuffling one alone would understate them. Each group is shuffled together instead.
GROUPS = {"Ratings": RATING,
          "Time control + rated": OTHER,
          "Material (piece values)": ["mat_diff", "white_mat", "black_mat", "pawn_diff"],
          "Queens left": ["queens_left"],
          "Captures + checks": ["captures", "checks"],
          "Castling": ["castled_w", "castled_b"]}
Xte = X.loc[te_idx]
base_te = accuracy_score(yte, gb.predict(Xte))
rng = np.random.default_rng(SEED)
imp = {}
for gname, cols in GROUPS.items():
    drops = []
    for _ in range(10):
        Xs = Xte.copy()
        Xs[cols] = Xte[cols].iloc[rng.permutation(len(Xte))].to_numpy()
        drops.append(base_te - accuracy_score(yte, gb.predict(Xs)))
    imp[gname] = np.mean(drops)
imp = pd.Series(imp).sort_values(ascending=False)
imp.to_csv(f"{out}/feature_importance.csv", header=["accuracy_drop_when_shuffled"])
top = imp[::-1]
plt.figure(figsize=(5.2, 3.4))
plt.barh(top.index, 100 * top.values, color="#2b6cb0")
plt.xlabel("Accuracy drop when shuffled (points)")
plt.title(f"Feature-group importance\n(Gradient Boosting, {N // 2} full moves revealed)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{out}/feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n===== COPY EVERYTHING FROM HERE TO THE END AND PASTE IT TO CLAUDE =====")
print(f"games used: {len(df)} | train/test: {len(tr_idx)}/{len(te_idx)}")
print(f"majority {100*maj_acc:.1f}% | higher-rated-wins {100*base_acc:.1f}% | test-set std. error about "
      f"{100*np.sqrt(base_acc*(1-base_acc)/n_te):.1f} points")
print("\nAccuracy by moves revealed:\n", res.round(3).to_string(index=False))
print("\nGrouped permutation importance (accuracy points lost when the group is shuffled):\n",
      (100 * imp).round(2).to_string())
print("\nFiles written to:", os.path.abspath(out))
