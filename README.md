# March Machine Learning Mania 2026 — 2nd Place Solution

**Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
**Final result:** 2nd place | Brier score: **0.1149886**

---

## Summary

XGBoost + Logistic Regression ensemble trained on historical NCAA tournament results (men's back to 2003, women's to 2010). All predictions are generated from 35 matchup features expressed as Team1 − Team2 differences, covering efficiency ratings, Four Factors, Elo, Massey ordinals, strength of schedule, and recent form.

See [`kagglewriteup.md`](kagglewriteup.md) for a full writeup of the approach, feature engineering decisions, validation strategy, and what worked/didn't.

---

## Repo Structure

```
├── march-mania-2026.py       # Full pipeline: feature engineering → training → submission
├── kagglewriteup.md          # Detailed solution writeup
├── requirements.txt          # Python dependencies
├── output/
│   └── submission.csv        # The exact submission that placed 2nd
└── data/                     # Not included — download from Kaggle (see below)
```

---

## Steps to Reproduce

### 1. Clone

```bash
git clone https://github.com/BrendanCarlin/march-mania-2026.git
cd march-mania-2026
```

### 2. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download data

Download all files from the [competition data page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) into a `data/` directory at the repo root. Required files:

```
data/
├── MRegularSeasonDetailedResults.csv
├── MRegularSeasonCompactResults.csv
├── MNCAATourneyCompactResults.csv
├── MNCAATourneySeeds.csv
├── MTeams.csv
├── MMasseyOrdinals.csv
├── WRegularSeasonDetailedResults.csv
├── WRegularSeasonCompactResults.csv
├── WNCAATourneyCompactResults.csv
├── WNCAATourneySeeds.csv
├── WTeams.csv
└── SampleSubmissionStage2.csv
```

### 4. Run

```bash
mkdir output
python march-mania-2026.py
```

Outputs `output/submission.csv`. The script is fully deterministic (`random_state=42`) — re-running produces a bit-for-bit identical submission to the one that placed 2nd.

---

## Model Overview

| Component | Detail |
|---|---|
| Final blend | 60% XGBoost + 40% Logistic Regression |
| Training data | Men's NCAA tourney 2003–2025 + Women's 2010–2025 |
| Features | 35 Team1−Team2 difference features |
| CV metric | 10-fold Brier score |
| CV Brier (blended) | 0.1925 men's / 0.1480 women's |
| Prediction clip | [0.02, 0.98] |

Key features: net efficiency, Elo (with margin-of-victory multiplier), Massey ordinal consensus (mean/median/min across all ranking systems), seed differential, strength of schedule, recent form.