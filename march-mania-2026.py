"""
Generate full submission covering ALL team matchups.
Uses the trained model for seeded teams, and a fast Elo/stats-based
approach for non-seeded teams.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

DATA_DIR = 'data'
OUT_DIR = 'output'

print("Loading data...")
m_season_detail = pd.read_csv(f'{DATA_DIR}/MRegularSeasonDetailedResults.csv')
m_season_compact = pd.read_csv(f'{DATA_DIR}/MRegularSeasonCompactResults.csv')
m_tourney = pd.read_csv(f'{DATA_DIR}/MNCAATourneyCompactResults.csv')
m_seeds = pd.read_csv(f'{DATA_DIR}/MNCAATourneySeeds.csv')
m_teams = pd.read_csv(f'{DATA_DIR}/MTeams.csv')
m_ordinals = pd.read_csv(f'{DATA_DIR}/MMasseyOrdinals.csv')

w_season_detail = pd.read_csv(f'{DATA_DIR}/WRegularSeasonDetailedResults.csv')
w_season_compact = pd.read_csv(f'{DATA_DIR}/WRegularSeasonCompactResults.csv')
w_tourney = pd.read_csv(f'{DATA_DIR}/WNCAATourneyCompactResults.csv')
w_seeds = pd.read_csv(f'{DATA_DIR}/WNCAATourneySeeds.csv')
w_teams = pd.read_csv(f'{DATA_DIR}/WTeams.csv')

sub = pd.read_csv(f'{DATA_DIR}/SampleSubmissionStage2.csv')


def parse_seed(seed_str):
    return int(seed_str[1:3])


# ============================================================
# COMPUTE SEASON STATS FOR ALL TEAMS
# ============================================================

def compute_all_season_stats(detail_df, compact_df):
    """Compute season stats from compact results for all teams."""
    all_records = []

    for df in [detail_df, compact_df]:
        w = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc']].copy()
        w.columns = ['Season', 'DayNum', 'TeamID', 'Score', 'OppID', 'OppScore', 'Loc']
        w['Win'] = 1

        l = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc']].copy()
        l.columns = ['Season', 'DayNum', 'TeamID', 'Score', 'OppID', 'OppScore', 'Loc']
        l['Loc'] = l['Loc'].map({'H': 'A', 'A': 'H', 'N': 'N'})
        l['Win'] = 0

        all_records.append(pd.concat([w, l], ignore_index=True))

    games = pd.concat(all_records, ignore_index=True)
    games = games.drop_duplicates(subset=['Season', 'DayNum', 'TeamID', 'OppID'], keep='first')

    stats = games.groupby(['Season', 'TeamID']).agg(
        Wins=('Win', 'sum'),
        Games=('Win', 'count'),
        AvgPts=('Score', 'mean'),
        AvgOppPts=('OppScore', 'mean'),
    ).reset_index()

    stats['WinPct'] = stats['Wins'] / stats['Games']
    stats['PointDiff'] = stats['AvgPts'] - stats['AvgOppPts']

    # Advanced stats from detailed results
    detail_games = []
    for df in [detail_df]:
        if len(df) == 0:
            continue
        for prefix, is_winner in [('W', True), ('L', False)]:
            other = 'L' if is_winner else 'W'
            d = pd.DataFrame({
                'Season': df['Season'],
                'DayNum': df['DayNum'],
                'TeamID': df[f'{prefix}TeamID'],
                'FGM': df[f'{prefix}FGM'], 'FGA': df[f'{prefix}FGA'],
                'FGM3': df[f'{prefix}FGM3'], 'FGA3': df[f'{prefix}FGA3'],
                'FTM': df[f'{prefix}FTM'], 'FTA': df[f'{prefix}FTA'],
                'OR': df[f'{prefix}OR'], 'DR': df[f'{prefix}DR'],
                'Ast': df[f'{prefix}Ast'], 'TO': df[f'{prefix}TO'],
                'Stl': df[f'{prefix}Stl'], 'Blk': df[f'{prefix}Blk'],
                'Score': df[f'{prefix}Score'],
                'OppScore': df[f'{other}Score'],
                'OppFGA': df[f'{other}FGA'], 'OppOR': df[f'{other}OR'],
                'OppDR': df[f'{other}DR'], 'OppTO': df[f'{other}TO'],
                'OppFTM': df[f'{other}FTM'], 'OppFTA': df[f'{other}FTA'],
                'OppFGM': df[f'{other}FGM'],
            })
            detail_games.append(d)

    if detail_games:
        dg = pd.concat(detail_games, ignore_index=True)
        dg['Poss'] = dg['FGA'] - dg['OR'] + dg['TO'] + 0.475 * dg['FTA']
        dg['OppPoss'] = dg['OppFGA'] - dg['OppOR'] + dg['OppTO'] + 0.475 * dg['OppFTA']

        adv = dg.groupby(['Season', 'TeamID']).agg(
            TotalPts=('Score', 'sum'), TotalOppPts=('OppScore', 'sum'),
            FGM=('FGM', 'sum'), FGA=('FGA', 'sum'),
            FGM3=('FGM3', 'sum'), FGA3=('FGA3', 'sum'),
            FTM=('FTM', 'sum'), FTA=('FTA', 'sum'),
            OR=('OR', 'sum'), DR=('DR', 'sum'),
            Ast=('Ast', 'sum'), TO=('TO', 'sum'),
            Stl=('Stl', 'sum'), Blk=('Blk', 'sum'),
            OppFGM=('OppFGM', 'sum'), OppFGA=('OppFGA', 'sum'),
            OppOR=('OppOR', 'sum'), OppDR=('OppDR', 'sum'),
            OppTO=('OppTO', 'sum'), OppFTM=('OppFTM', 'sum'), OppFTA=('OppFTA', 'sum'),
            Poss=('Poss', 'sum'), OppPoss=('OppPoss', 'sum'),
            DetailGames=('Score', 'count'),
        ).reset_index()

        g = adv['DetailGames']
        adv['OffEff'] = (adv['TotalPts'] / adv['Poss'].replace(0, 1)) * 100
        adv['DefEff'] = (adv['TotalOppPts'] / adv['OppPoss'].replace(0, 1)) * 100
        adv['NetEff'] = adv['OffEff'] - adv['DefEff']
        adv['EFGPct'] = (adv['FGM'] + 0.5 * adv['FGM3']) / adv['FGA'].replace(0, 1)
        adv['TORate'] = adv['TO'] / adv['Poss'].replace(0, 1)
        adv['ORPct'] = adv['OR'] / (adv['OR'] + adv['OppDR']).replace(0, 1)
        adv['FTRate'] = adv['FTM'] / adv['FGA'].replace(0, 1)
        adv['FGPct'] = adv['FGM'] / adv['FGA'].replace(0, 1)
        adv['FG3Pct'] = adv['FGM3'] / adv['FGA3'].replace(0, 1)
        adv['FTPct'] = adv['FTM'] / adv['FTA'].replace(0, 1)
        adv['OppFGPct'] = adv['OppFGM'] / adv['OppFGA'].replace(0, 1)
        adv['AstPerGame'] = adv['Ast'] / g
        adv['TOPerGame'] = adv['TO'] / g
        adv['StlPerGame'] = adv['Stl'] / g
        adv['BlkPerGame'] = adv['Blk'] / g
        adv['ORPerGame'] = adv['OR'] / g
        adv['DRPerGame'] = adv['DR'] / g
        adv['Tempo'] = adv['Poss'] / g
        adv['OppEFGPct'] = (adv['OppFGM'] + 0.5 * (adv['OppFGM'] - (adv['OppFGA'] * adv['OppFGPct'] - adv['OppFGM']))) / adv['OppFGA'].replace(0, 1)
        adv['OppTORate'] = adv['OppTO'] / adv['OppPoss'].replace(0, 1)
        adv['DRPct'] = adv['DR'] / (adv['DR'] + adv['OppOR']).replace(0, 1)
        adv['OppFTRate'] = adv['OppFTM'] / adv['OppFGA'].replace(0, 1)

        adv_cols = ['Season', 'TeamID', 'OffEff', 'DefEff', 'NetEff', 'EFGPct', 'TORate',
                    'ORPct', 'FTRate', 'FGPct', 'FG3Pct', 'FTPct', 'OppFGPct',
                    'AstPerGame', 'TOPerGame', 'StlPerGame', 'BlkPerGame',
                    'ORPerGame', 'DRPerGame', 'Tempo', 'OppEFGPct', 'OppTORate',
                    'DRPct', 'OppFTRate']
        stats = stats.merge(adv[adv_cols], on=['Season', 'TeamID'], how='left')

    return stats


def compute_elo(compact_df, k=20, home_adv=100, mean_reversion=0.75):
    elo = {}
    elo_by_season = {}
    for season in sorted(compact_df['Season'].unique()):
        for team in elo:
            elo[team] = elo[team] * mean_reversion + 1500 * (1 - mean_reversion)
        for _, game in compact_df[compact_df['Season'] == season].sort_values('DayNum').iterrows():
            w, l = game['WTeamID'], game['LTeamID']
            if w not in elo: elo[w] = 1500
            if l not in elo: elo[l] = 1500
            w_elo, l_elo = elo[w], elo[l]
            if game['WLoc'] == 'H': w_elo += home_adv
            elif game['WLoc'] == 'A': l_elo += home_adv
            w_exp = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
            mov = game['WScore'] - game['LScore']
            mov_mult = np.log(max(mov, 1) + 1) * (2.2 / ((w_elo - l_elo) * 0.001 + 2.2))
            elo[w] += k * mov_mult * (1 - w_exp)
            elo[l] -= k * mov_mult * (1 - w_exp)
        for team in elo:
            elo_by_season[(season, team)] = elo[team]
    return pd.DataFrame([{'Season': s, 'TeamID': t, 'Elo': e} for (s, t), e in elo_by_season.items()])


def get_massey(ordinals_df, season):
    so = ordinals_df[ordinals_df['Season'] == season]
    if len(so) == 0:
        return pd.DataFrame(columns=['TeamID', 'MasseyMean', 'MasseyMedian', 'MasseyMin', 'MasseyStd'])
    max_day = so['RankingDayNum'].max()
    late = so[so['RankingDayNum'] >= max_day - 14]
    agg = late.groupby('TeamID')['OrdinalRank'].agg(
        MasseyMean='mean', MasseyMedian='median', MasseyMin='min', MasseyStd='std'
    ).reset_index()
    agg['MasseyStd'] = agg['MasseyStd'].fillna(0)
    return agg


def compute_sos(compact_df, season, win_pcts):
    sg = compact_df[compact_df['Season'] == season]
    w = sg[['WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID', 'LTeamID': 'OppID'})
    l = sg[['LTeamID', 'WTeamID']].rename(columns={'LTeamID': 'TeamID', 'WTeamID': 'OppID'})
    m = pd.concat([w, l]).merge(
        win_pcts[['TeamID', 'WinPct']].rename(columns={'TeamID': 'OppID', 'WinPct': 'OppWP'}),
        on='OppID', how='left'
    )
    m['OppWP'] = m['OppWP'].fillna(0.5)
    return m.groupby('TeamID')['OppWP'].mean().reset_index().rename(columns={'OppWP': 'SOS'})


def compute_form(compact_df, season, n=10):
    sg = compact_df[compact_df['Season'] == season].sort_values('DayNum')
    w = sg[['DayNum', 'WTeamID']].copy(); w.columns = ['DayNum', 'TeamID']; w['Win'] = 1
    l = sg[['DayNum', 'LTeamID']].copy(); l.columns = ['DayNum', 'TeamID']; l['Win'] = 0
    ag = pd.concat([w, l]).sort_values(['TeamID', 'DayNum'])
    return ag.groupby('TeamID').tail(n).groupby('TeamID')['Win'].mean().reset_index().rename(columns={'Win': 'RecentWinPct'})


print("Computing stats...")
m_stats = compute_all_season_stats(m_season_detail, m_season_compact)
w_stats = compute_all_season_stats(w_season_detail, w_season_compact)

print("Computing Elo...")
m_elo = compute_elo(m_season_compact)
w_elo = compute_elo(w_season_compact)

print("Processing seeds...")
m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed)
w_seeds['SeedNum'] = w_seeds['Seed'].apply(parse_seed)

# Precompute massey for training seasons
print("Precomputing Massey ordinals...")
massey_cache = {}
for season in sorted(set(m_tourney['Season'].unique()) | {2026}):
    if season >= 2003:
        massey_cache[season] = get_massey(m_ordinals, season)

# Note: No Massey ordinals for women in this dataset, will handle separately


# ============================================================
# BUILD FEATURE VECTORS
# ============================================================

FEATURE_COLS = ['WinPctDiff', 'PointDiffDiff', 'AvgPtsDiff', 'AvgOppPtsDiff',
                'OffEffDiff', 'DefEffDiff', 'NetEffDiff', 'EFGPctDiff', 'TORateDiff',
                'ORPctDiff', 'FTRateDiff', 'FGPctDiff', 'FG3PctDiff', 'FTPctDiff',
                'OppFGPctDiff', 'AstPerGameDiff', 'TOPerGameDiff', 'StlPerGameDiff',
                'BlkPerGameDiff', 'ORPerGameDiff', 'DRPerGameDiff', 'TempoDiff',
                'OppEFGPctDiff', 'OppTORateDiff', 'DRPctDiff', 'OppFTRateDiff',
                'EloDiff', 'SeedDiff', 'Seed1', 'Seed2',
                'MasseyMeanDiff', 'MasseyMedianDiff', 'MasseyMinDiff',
                'SOSDiff', 'RecentFormDiff']


def build_team_profile(season, team_id, stats_df, elo_df, seeds_df, massey_df, sos_df, form_df):
    """Build a single team's profile for a given season."""
    s = stats_df[(stats_df['Season'] == season) & (stats_df['TeamID'] == team_id)]
    if len(s) == 0:
        return None
    s = s.iloc[0]

    profile = {}
    for col in ['WinPct', 'PointDiff', 'AvgPts', 'AvgOppPts',
                 'OffEff', 'DefEff', 'NetEff', 'EFGPct', 'TORate', 'ORPct',
                 'FTRate', 'FGPct', 'FG3Pct', 'FTPct', 'OppFGPct',
                 'AstPerGame', 'TOPerGame', 'StlPerGame', 'BlkPerGame',
                 'ORPerGame', 'DRPerGame', 'Tempo', 'OppEFGPct', 'OppTORate',
                 'DRPct', 'OppFTRate']:
        profile[col] = s[col] if col in s.index and pd.notna(s[col]) else 0

    e = elo_df[(elo_df['Season'] == season) & (elo_df['TeamID'] == team_id)]
    profile['Elo'] = e.iloc[0]['Elo'] if len(e) > 0 else 1500

    sd = seeds_df[(seeds_df['Season'] == season) & (seeds_df['TeamID'] == team_id)]
    profile['SeedNum'] = sd.iloc[0]['SeedNum'] if len(sd) > 0 else 17  # unseeded = 17

    if massey_df is not None and len(massey_df) > 0:
        m = massey_df[massey_df['TeamID'] == team_id]
        profile['MasseyMean'] = m.iloc[0]['MasseyMean'] if len(m) > 0 else 200
        profile['MasseyMedian'] = m.iloc[0]['MasseyMedian'] if len(m) > 0 else 200
        profile['MasseyMin'] = m.iloc[0]['MasseyMin'] if len(m) > 0 else 200
    else:
        profile['MasseyMean'] = 200
        profile['MasseyMedian'] = 200
        profile['MasseyMin'] = 200

    if sos_df is not None and len(sos_df) > 0:
        ss = sos_df[sos_df['TeamID'] == team_id]
        profile['SOS'] = ss.iloc[0]['SOS'] if len(ss) > 0 else 0.5
    else:
        profile['SOS'] = 0.5

    if form_df is not None and len(form_df) > 0:
        f = form_df[form_df['TeamID'] == team_id]
        profile['RecentWinPct'] = f.iloc[0]['RecentWinPct'] if len(f) > 0 else 0.5
    else:
        profile['RecentWinPct'] = 0.5

    return profile


def matchup_features(p1, p2):
    """Create feature vector from two team profiles."""
    return {
        'WinPctDiff': p1['WinPct'] - p2['WinPct'],
        'PointDiffDiff': p1['PointDiff'] - p2['PointDiff'],
        'AvgPtsDiff': p1['AvgPts'] - p2['AvgPts'],
        'AvgOppPtsDiff': p1['AvgOppPts'] - p2['AvgOppPts'],
        'OffEffDiff': p1['OffEff'] - p2['OffEff'],
        'DefEffDiff': p1['DefEff'] - p2['DefEff'],
        'NetEffDiff': p1['NetEff'] - p2['NetEff'],
        'EFGPctDiff': p1['EFGPct'] - p2['EFGPct'],
        'TORateDiff': p1['TORate'] - p2['TORate'],
        'ORPctDiff': p1['ORPct'] - p2['ORPct'],
        'FTRateDiff': p1['FTRate'] - p2['FTRate'],
        'FGPctDiff': p1['FGPct'] - p2['FGPct'],
        'FG3PctDiff': p1['FG3Pct'] - p2['FG3Pct'],
        'FTPctDiff': p1['FTPct'] - p2['FTPct'],
        'OppFGPctDiff': p1['OppFGPct'] - p2['OppFGPct'],
        'AstPerGameDiff': p1['AstPerGame'] - p2['AstPerGame'],
        'TOPerGameDiff': p1['TOPerGame'] - p2['TOPerGame'],
        'StlPerGameDiff': p1['StlPerGame'] - p2['StlPerGame'],
        'BlkPerGameDiff': p1['BlkPerGame'] - p2['BlkPerGame'],
        'ORPerGameDiff': p1['ORPerGame'] - p2['ORPerGame'],
        'DRPerGameDiff': p1['DRPerGame'] - p2['DRPerGame'],
        'TempoDiff': p1['Tempo'] - p2['Tempo'],
        'OppEFGPctDiff': p1['OppEFGPct'] - p2['OppEFGPct'],
        'OppTORateDiff': p1['OppTORate'] - p2['OppTORate'],
        'DRPctDiff': p1['DRPct'] - p2['DRPct'],
        'OppFTRateDiff': p1['OppFTRate'] - p2['OppFTRate'],
        'EloDiff': p1['Elo'] - p2['Elo'],
        'SeedDiff': p1['SeedNum'] - p2['SeedNum'],
        'Seed1': p1['SeedNum'],
        'Seed2': p2['SeedNum'],
        'MasseyMeanDiff': p1['MasseyMean'] - p2['MasseyMean'],
        'MasseyMedianDiff': p1['MasseyMedian'] - p2['MasseyMedian'],
        'MasseyMinDiff': p1['MasseyMin'] - p2['MasseyMin'],
        'SOSDiff': p1['SOS'] - p2['SOS'],
        'RecentFormDiff': p1['RecentWinPct'] - p2['RecentWinPct'],
    }


# ============================================================
# BUILD TRAINING DATA (men's tourney only, 2003+)
# ============================================================

print("Building training data...")

train_rows = []
for _, game in m_tourney[m_tourney['Season'] >= 2003].iterrows():
    season = game['Season']
    t1, t2 = min(game['WTeamID'], game['LTeamID']), max(game['WTeamID'], game['LTeamID'])
    result = 1 if game['WTeamID'] == t1 else 0

    massey_df = massey_cache.get(season)
    sos_df = compute_sos(m_season_compact, season, m_stats[m_stats['Season'] == season])
    form_df = compute_form(m_season_compact, season)

    p1 = build_team_profile(season, t1, m_stats, m_elo, m_seeds, massey_df, sos_df, form_df)
    p2 = build_team_profile(season, t2, m_stats, m_elo, m_seeds, massey_df, sos_df, form_df)

    if p1 and p2:
        feats = matchup_features(p1, p2)
        feats['Result'] = result
        train_rows.append(feats)

# Add women's tourney data too
print("Adding women's training data...")
for _, game in w_tourney[w_tourney['Season'] >= 2010].iterrows():
    season = game['Season']
    t1, t2 = min(game['WTeamID'], game['LTeamID']), max(game['WTeamID'], game['LTeamID'])
    result = 1 if game['WTeamID'] == t1 else 0

    sos_df = compute_sos(w_season_compact, season, w_stats[w_stats['Season'] == season])
    form_df = compute_form(w_season_compact, season)

    p1 = build_team_profile(season, t1, w_stats, w_elo, w_seeds, None, sos_df, form_df)
    p2 = build_team_profile(season, t2, w_stats, w_elo, w_seeds, None, sos_df, form_df)

    if p1 and p2:
        feats = matchup_features(p1, p2)
        feats['Result'] = result
        train_rows.append(feats)

train_df = pd.DataFrame(train_rows)
print(f"Training samples: {len(train_df)}")

X_train = train_df[FEATURE_COLS].values
y_train = train_df['Result'].values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

# Train models
print("Training models...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr.fit(X_scaled, y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
    use_label_encoder=False, random_state=42
)
xgb_model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
lr_brier = -cross_val_score(lr, X_scaled, y_train, cv=10, scoring='neg_brier_score').mean()
xgb_brier = -cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='neg_brier_score').mean()
print(f"  LR CV Brier: {lr_brier:.4f}")
print(f"  XGB CV Brier: {xgb_brier:.4f}")


# ============================================================
# PRECOMPUTE 2026 TEAM PROFILES
# ============================================================

print("\nPrecomputing 2026 team profiles...")

# Parse all team IDs from submission
sub_team_ids = set()
for game_id in sub['ID']:
    parts = game_id.split('_')
    sub_team_ids.add(int(parts[1]))
    sub_team_ids.add(int(parts[2]))

men_teams_2026 = sorted([t for t in sub_team_ids if t < 3000])
women_teams_2026 = sorted([t for t in sub_team_ids if t >= 3000])

massey_2026 = massey_cache.get(2026, pd.DataFrame())
m_sos_2026 = compute_sos(m_season_compact, 2026, m_stats[m_stats['Season'] == 2026])
m_form_2026 = compute_form(m_season_compact, 2026)
w_sos_2026 = compute_sos(w_season_compact, 2026, w_stats[w_stats['Season'] == 2026])
w_form_2026 = compute_form(w_season_compact, 2026)

# Build profiles
m_profiles = {}
for t in men_teams_2026:
    p = build_team_profile(2026, t, m_stats, m_elo, m_seeds, massey_2026, m_sos_2026, m_form_2026)
    if p:
        m_profiles[t] = p

w_profiles = {}
for t in women_teams_2026:
    p = build_team_profile(2026, t, w_stats, w_elo, w_seeds, None, w_sos_2026, w_form_2026)
    if p:
        w_profiles[t] = p

print(f"  Men's profiles: {len(m_profiles)}")
print(f"  Women's profiles: {len(w_profiles)}")


# ============================================================
# GENERATE ALL PREDICTIONS
# ============================================================

print("\nGenerating all predictions...")

LR_WEIGHT = 0.4
XGB_WEIGHT = 0.6

predictions = {}
batch_features = []
batch_ids = []

for _, row in sub.iterrows():
    game_id = row['ID']
    parts = game_id.split('_')
    t1, t2 = int(parts[1]), int(parts[2])

    profiles = m_profiles if t1 < 3000 else w_profiles

    p1 = profiles.get(t1)
    p2 = profiles.get(t2)

    if p1 and p2:
        feats = matchup_features(p1, p2)
        batch_features.append([feats[c] for c in FEATURE_COLS])
        batch_ids.append(game_id)
    else:
        predictions[game_id] = 0.5

# Batch predict
if batch_features:
    X = np.array(batch_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_s = scaler.transform(X)
    lr_preds = lr.predict_proba(X_s)[:, 1]
    xgb_preds = xgb_model.predict_proba(X)[:, 1]

    blended = LR_WEIGHT * lr_preds + XGB_WEIGHT * xgb_preds
    blended = np.clip(blended, 0.02, 0.98)

    for gid, pred in zip(batch_ids, blended):
        predictions[gid] = pred

print(f"  Model predictions: {len(batch_ids)}")
print(f"  Fallback (0.5): {len(predictions) - len(batch_ids)}")

# Write submission
submission = sub.copy()
submission['Pred'] = submission['ID'].map(predictions)
submission['Pred'] = submission['Pred'].fillna(0.5)
submission.to_csv(f'{OUT_DIR}/submission.csv', index=False)

print(f"\nSubmission stats:")
print(submission['Pred'].describe())
print(f"\nPreds == 0.5: {(submission['Pred'] == 0.5).sum()}")
print(f"Saved to {OUT_DIR}/submission.csv")
print("Done!")
