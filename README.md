World Cup xG Model
A simple expected goals (xG) model built from scratch in Python using StatsBomb's free data from the 2022 FIFA World Cup.
I wanted to see if the "Moneyball" idea of judging performance by the quality of chances, not just results, holds up on real soccer data, and to find out which players finished better or worse than their chances suggested.
The question
How likely is a given shot to become a goal, and who scored more (or fewer) goals than expected at the 2022 World Cup?
How it works
Load the data. Every shot from all 64 matches (about 1,400 shots after cleaning), pulled with statsbombpy.
Clean it. Penalties and penalty shootout kicks are removed, since they're a different kind of shot and would distort the model.
Build features. From each shot's position on the pitch I calculate:
Distance to the center of the goal
Angle, meaning how wide the goal looks from where the shot was taken
Fit a logistic regression that predicts the probability a shot is a goal, trained on 75% of the shots and tested on the other 25%.
Score every shot. Each shot gets an xG value, and I compare each player's total xG to their actual goals.
Results
Metric
Baseline (same chance for every shot)
Model
Log loss (lower is better)
0.338
0.285
ROC AUC (0.5 = coin flip)
n/a
0.784

Using only distance and angle, the model clearly beats the baseline, so those two features carry real information about shot quality.
Some things it found (players with at least 8 shots):
Kylian Mbappé had 6 goals from about 2.9 xG, the biggest overperformance in the tournament.
Julián Álvarez scored 4 from about 1.4 xG.
Lautaro Martínez took 14 shots and scored none, from about 1.9 xG.

(See xg_vs_goals.png for the photo for the next paragraph)
Each dot is a player. Dots above the dashed line scored more goals than their chances suggested, and dots below it scored fewer.
The model also flagged long-range goals as very unlikely, like Hakim Ziyech's strike from about 36 yards, which it gave roughly a 1% chance.
Limitations
Small sample. A single tournament is only about 150 goals, so a player being a goal or two above or below xG is mostly luck, not proof of skill.
Only two features. The model doesn't know about the goalkeeper's position, defenders in the way, or the quality of the pass before the shot. Better xG models use all of these.
Position data only. I added header and set-piece flags in the code but haven't tested whether they improve the model yet.
Run it yourself
pip install -r requirements.txt
python xg_starter.py

The first run downloads the match data (about a minute) and caches it to a CSV, so later runs are instant.
What I'd like to try next
Add the body part and shot type features and see how much the model improves
Train on more tournaments (Euro 2024, the 2018 World Cup) for a bigger sample
Try the same idea on USL data from American Soccer Analysis
Data
Shot data comes from StatsBomb's open data. Thanks to StatsBomb for making it freely available.
<img src="statsbomb-logo.png" width="200" alt="StatsBomb">
Built by Andre Nodot
