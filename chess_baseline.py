"""CS-412 project: predicting chess game outcomes (Lichess games, Kaggle "datasnaek/chess").

Usage:   python chess_baseline.py games.csv out
Outputs: out/stats.tex, out/results_binary.tex, out/results_three.tex,
         out/winrate_by_ratingdiff.png, out/model_accuracy.png, out/results.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
path = sys.argv[1] if len(sys.argv) > 1 else "games.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------- load
df = pd.read_csv(path)
print("Loaded", path, "shape:", df.shape)
print("Columns:", list(df.columns))
needed = ["winner", "white_rating", "black_rating", "increment_code",
          "opening_eco", "opening_ply", "rated"]
absent = [c for c in needed if c not in df.columns]
if absent:
    sys.exit(f"ERROR: expected columns missing from the CSV: {absent}")
n_raw = len(df)

# ---------------------------------------------------------------- clean
if "id" in df.columns:
    df = df.drop_duplicates(subset="id")
else:
    df = df.drop_duplicates()
df = df[df["winner"].isin(["white", "black", "draw"])].copy()

# Columns that are only known AFTER the game ends (or encode the full game): excluded.
LEAKY = ["victory_status", "turns", "created_at", "last_move_at", "moves"]
print("Excluded as leakage / identifiers:", [c for c in LEAKY if c in df.columns],
      "+ player ids")

tc = df["increment_code"].astype(str).str.extract(r"(\d+)\+(\d+)").astype(float)
df["base_min"], df["inc_sec"] = tc[0], tc[1]
df["rated"] = df["rated"].astype(str).str.lower().eq("true").astype(int)
df = df.dropna(subset=["base_min", "inc_sec", "white_rating", "black_rating",
                       "opening_eco", "opening_ply"]).copy()
df["rating_diff"] = df["white_rating"] - df["black_rating"]
df["rating_mean"] = (df["white_rating"] + df["black_rating"]) / 2
df["eco_letter"] = df["opening_eco"].str[0]
top = df["opening_eco"].value_counts().head(30).index
df["eco_code"] = df["opening_eco"].where(df["opening_eco"].isin(top), "other")

NUM = ["white_rating", "black_rating", "rating_diff", "rating_mean",
       "base_min", "inc_sec", "rated", "opening_ply"]
CAT = ["eco_letter", "eco_code"]
print("Features used:", NUM + CAT)

counts = df["winner"].value_counts()
n_games, n_draw = len(df), int(counts.get("draw", 0))
binary = df[df["winner"] != "draw"]
print("\nClass counts:\n", counts.to_string())

# ---------------------------------------------------------------- stats macros
pct = lambda k: 100 * counts.get(k, 0) / n_games
stats = {
    "nRaw": f"{n_raw:,}".replace(",", "{,}"),
    "nGames": f"{n_games:,}".replace(",", "{,}"),
    "nBinary": f"{len(binary):,}".replace(",", "{,}"),
    "nDraw": f"{n_draw:,}".replace(",", "{,}"),
    "pWhite": f"{pct('white'):.1f}", "pBlack": f"{pct('black'):.1f}",
    "pDraw": f"{pct('draw'):.1f}",
    "ratingMin": f"{int(min(df.white_rating.min(), df.black_rating.min()))}",
    "ratingMax": f"{int(max(df.white_rating.max(), df.black_rating.max()))}",
    "ratingMean": f"{df[['white_rating', 'black_rating']].stack().mean():.0f}",
    "nOpenings": f"{df['opening_eco'].nunique()}",
}
with open(f"{out}/stats.tex", "w") as f:
    for k, v in stats.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- figure 1
bins = np.arange(-400, 401, 100)
tmp = binary.copy()
tmp["bin"] = pd.cut(tmp["rating_diff"].clip(-399, 399), bins)
g = tmp.groupby("bin", observed=True)["winner"].agg(
    lambda s: (s == "white").mean())
centers = [iv.mid for iv in g.index]
plt.figure(figsize=(4.2, 2.8))
plt.plot(centers, 100 * g.values, marker="o")
plt.axhline(50, color="gray", lw=0.8, ls="--")
plt.xlabel("White rating minus Black rating")
plt.ylabel("White win rate (%)")
plt.title("Win rate vs. rating difference (no draws)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{out}/winrate_by_ratingdiff.png", dpi=200)
plt.close()


# ---------------------------------------------------------------- models
def make_prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Decision Tree (depth 5)": lambda: DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "k-NN (k=25)": lambda: KNeighborsClassifier(n_neighbors=25),
    "Random Forest (200 trees)": lambda: RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=SEED),
}


def run_task(name, d, labels):
    X, y = d[NUM + CAT], d["winner"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows = []
    maj = ytr.value_counts().idxmax()
    pred = np.full(len(yte), maj)
    rows.append(("Majority class", None, None, accuracy_score(yte, pred),
                 f1_score(yte, pred, average="macro", zero_division=0)))
    rule = np.where(Xte["rating_diff"] >= 0, "white", "black")
    rows.append(("Higher-rated player wins", None, None, accuracy_score(yte, rule),
                 f1_score(yte, rule, average="macro", zero_division=0)))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for mname, mk in MODELS.items():
        pipe = Pipeline([("prep", make_prep()), ("clf", mk())])
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy")
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        rows.append((mname, scores.mean(), scores.std(), accuracy_score(yte, p),
                     f1_score(yte, p, average="macro", zero_division=0)))
        print(f"[{name}] {mname}: CV {scores.mean():.3f}+-{scores.std():.3f} "
              f"test {accuracy_score(yte, p):.3f}")
        if mname.startswith("Random Forest"):
            names = pipe.named_steps["prep"].get_feature_names_out()
            imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
            print(f"[{name}] top RF features:\n", imp.sort_values(ascending=False).head(8).round(3).to_string())
    res = pd.DataFrame(rows, columns=["model", "cv_mean", "cv_std", "test_acc", "test_f1"])
    res.insert(0, "task", name)
    return res, len(Xtr), len(Xte)


res_b, ntr, nte = run_task("binary", binary, ["white", "black"])
res_t, ntr3, nte3 = run_task("three_class", df, ["white", "black", "draw"])
pd.concat([res_b, res_t]).to_csv(f"{out}/results.csv", index=False)

for fname, res in [("results_binary.tex", res_b), ("results_three.tex", res_t)]:
    with open(f"{out}/{fname}", "w") as f:
        lines = []
        for _, r in res.iterrows():
            cv = "--" if pd.isna(r.cv_mean) else f"{100*r.cv_mean:.1f} $\\pm$ {100*r.cv_std:.1f}"
            lines.append(f"{r.model} & {cv} & {100*r.test_acc:.1f} & {100*r.test_f1:.1f}")
        f.write(" \\\\\n".join(lines) + "%\n")  # no trailing row break; the report adds it
with open(f"{out}/stats.tex", "a") as f:
    f.write(f"\\newcommand{{\\nTrain}}{{{ntr:,}}}\n".replace(",", "{,}"))
    f.write(f"\\newcommand{{\\nTest}}{{{nte:,}}}\n".replace(",", "{,}"))

# ---------------------------------------------------------------- figure 2
plt.figure(figsize=(4.2, 2.8))
labels = [m.split(" (")[0] for m in res_b["model"]]
plt.barh(labels[::-1], 100 * res_b["test_acc"].values[::-1], color="#2b6cb0")
plt.xlim(40, 100)
plt.xlabel("Test accuracy (%), white vs. black")
plt.tight_layout()
plt.savefig(f"{out}/model_accuracy.png", dpi=200)
plt.close()

print("\n===== COPY EVERYTHING FROM HERE TO THE END AND PASTE IT TO CLAUDE =====")
print("raw rows:", n_raw, "| after cleaning:", n_games, "| no-draw:", len(binary),
      "| draws:", n_draw)
print("white/black/draw %:", stats["pWhite"], stats["pBlack"], stats["pDraw"])
print("ratings min/mean/max:", stats["ratingMin"], stats["ratingMean"], stats["ratingMax"])
print("distinct opening codes:", stats["nOpenings"], "| train/test (binary):", ntr, nte)
print("\nWhite win rate by rating-difference bin (no draws):\n", (100*g).round(1).to_string())
print("\nBINARY (white vs black):\n", res_b.round(3).to_string(index=False))
print("\nTHREE-CLASS (with draws):\n", res_t.round(3).to_string(index=False))
print("\nFiles written to:", os.path.abspath(out))
