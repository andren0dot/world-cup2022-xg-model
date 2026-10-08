"""

# %% 1. Load shots (cached to a CSV so you only download once)
import os
import warnings

import numpy as np
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")  # statsbombpy warns about missing credentials

CACHE = "shots_wc2022.csv"


def load_shots() -> pd.DataFrame:
    comps = sb.competitions()
    row = comps[
        (comps["competition_name"] == "FIFA World Cup")
        & (comps["season_name"] == "2022")
        & (comps["competition_gender"] == "male")
    ].iloc[0]
    matches = sb.matches(
        competition_id=int(row["competition_id"]), season_id=int(row["season_id"])
    )
    print(f"Found {len(matches)} matches. Downloading events (about a minute)...")

    frames = []
    for match_id in matches["match_id"]:
        events = sb.events(match_id=int(match_id))
        shots = events[events["type"] == "Shot"].copy()
        shots["match_id"] = match_id
        frames.append(shots)
    shots = pd.concat(frames, ignore_index=True)

    # Pull out x/y so the data survives being saved to CSV
    shots["x"] = shots["location"].apply(lambda loc: loc[0])
    shots["y"] = shots["location"].apply(lambda loc: loc[1])

    keep = [
        "match_id", "player", "team", "period", "minute",
        "x", "y", "shot_body_part", "shot_type", "shot_outcome",
    ]
    return shots[keep]


if os.path.exists(CACHE):
    shots = pd.read_csv(CACHE)
else:
    shots = load_shots()
    shots.to_csv(CACHE, index=False)

print(shots.shape)
print(shots.head())

# %% 2. Clean up
# Standard xG practice: leave out penalties (they're ~76% no matter what the
# model says) and penalty shootouts (period 5).
shots = shots[(shots["shot_type"] != "Penalty") & (shots["period"] != 5)].copy()
shots["goal"] = (shots["shot_outcome"] == "Goal").astype(int)

print(f"{len(shots)} shots, {shots['goal'].sum()} goals "
      f"({shots['goal'].mean():.1%} conversion)")

# %% 3. Features
# StatsBomb pitch is 120 x 80, and we always attack toward x = 120.
# The goal is centered at y = 40 and is 8 yards wide (posts at y = 36 and 44).
GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 8

dx = GOAL_X - shots["x"]
dy = GOAL_Y - shots["y"]

# Distance from the shot to the center of the goal
shots["distance"] = np.sqrt(dx**2 + dy**2)

# Angle (radians) formed by the two goalposts as seen from the shot.
# Bigger angle = more of the goal is "visible" = better chance.
shots["angle"] = np.arctan2(
    GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2
)

# Easy extras (add these once the basic version works)
shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
shots["is_set_piece"] = (shots["shot_type"] != "Open Play").astype(int)

# Start with just distance + angle. Then add "is_header", "is_set_piece".
FEATURES = ["distance", "angle"]

# %% 4. Fit a logistic regression and check it
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

X = shots[FEATURES]
y = shots["goal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:, 1]

# Baseline: a "model" that gives every shot the same average probability
baseline = np.full(len(y_test), y_train.mean())

print(f"Log loss  - baseline: {log_loss(y_test, baseline):.3f}   "
      f"model: {log_loss(y_test, pred):.3f}   (lower is better)")
print(f"ROC AUC   - model: {roc_auc_score(y_test, pred):.3f}   "
      f"(0.5 = coin flip, 1.0 = perfect)")

# %% 5. Score every shot, then poke around
# Refit on all the data so every shot gets an xG value.
model.fit(X, y)
shots["xg"] = model.predict_proba(X)[:, 1]

cols = ["player", "team", "minute", "distance", "shot_body_part", "xg", "goal"]

print("\nGoals the model thought were nearly hopeless:")
print(shots[shots["goal"] == 1].nsmallest(10, "xg")[cols].round(3).to_string(index=False))

print("\nBig chances that didn't go in:")
print(shots[shots["goal"] == 0].nlargest(10, "xg")[cols].round(3).to_string(index=False))

# %% 6. Payoff question: who beat their xG?
players = (
    shots.groupby(["player", "team"])
    .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
    .reset_index()
)
players["goals_minus_xg"] = players["goals"] - players["xg"]

# Only players with enough shots for this to mean anything
players = players[players["shots"] >= 8]

print("\nBiggest overperformers (goals minus xG):")
print(players.nlargest(10, "goals_minus_xg").round(2).to_string(index=False))

print("\nBiggest underperformers:")
print(players.nsmallest(10, "goals_minus_xg").round(2).to_string(index=False))

# %% 7. Chart: actual goals vs. xG for each player
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(players["xg"], players["goals"], alpha=0.7)

# Diagonal line: goals = xG. Above it = scored more than expected.
limit = max(players["xg"].max(), players["goals"].max()) + 0.5
ax.plot([0, limit], [0, limit], linestyle="--", color="gray")

# Label a few standouts. (Labeling everyone gets messy because many players
# share the same spot, so we pick the biggest 3 over- and 2 underperformers.)
to_label = pd.concat([
    players.nlargest(3, "goals_minus_xg"),
    players.nsmallest(2, "goals_minus_xg"),
])

# StatsBomb uses long legal names, so give the labeled players short ones.
# (If you change which players get labeled, add them here or the full name is used.)
SHORT_NAMES = {
    "Kylian Mbappé Lottin": "Mbappé",
    "Julián Álvarez": "Álvarez",
    "Marcus Rashford": "Rashford",
    "Lautaro Javier Martínez": "L. Martínez",
    "Ismaïla Sarr": "Sarr",
}

for _, r in to_label.iterrows():
    name = SHORT_NAMES.get(r["player"], r["player"])
    if r["goals"] > r["xg"]:   # overperformer: label up and to the right
        offset, align = (6, 6), "left"
    else:                      # underperformer: label underneath the dot
        offset, align = (0, -14), "center"
    ax.annotate(name, (r["xg"], r["goals"]), xytext=offset,
                textcoords="offset points", fontsize=9, ha=align)

ax.set_ylim(-0.7, limit)

ax.set_xlabel("Expected goals (xG)")
ax.set_ylabel("Actual goals")
ax.set_title("2022 World Cup: goals vs. xG (players with 8+ shots, no penalties)")
ax.text(0.02, 0.97, "Above the line = scored more than expected",
        transform=ax.transAxes, fontsize=8, va="top")
ax.text(0.98, 0.03, "Data: StatsBomb open data", transform=ax.transAxes,
        fontsize=7, ha="right")

plt.tight_layout()
plt.savefig("xg_vs_goals.png", dpi=150)
plt.show()
