#!/usr/bin/env python3
"""
NBA Game Prediction Script

Predicts outcomes for games on a given date using trained XGBoost models.
Usage: python predict_games.py 2026-01-17 [--model regression]
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import sys
from datetime import datetime
import argparse

# Directories
MODEL_DIR = 'models'
ENHANCED_DIR = 'data_enhanced'
OUTPUT_DIR = 'predictions'
SCHEDULE_FILE = 'nba_schedule_2025-26.csv'

# Team name normalization (from merge_games.py)
TEAM_MAPPING = {
    'BRK': 'BKN',
    'CHO': 'CHA',
    'PHO': 'PHX',
}

REVERSE_TEAM_MAPPING = {v: k for k, v in TEAM_MAPPING.items()}

def normalize_team_name(team, to_standard=True):
    """Normalize team name between Basketball-Reference and standard formats."""
    if to_standard:
        return TEAM_MAPPING.get(team, team)
    else:
        return REVERSE_TEAM_MAPPING.get(team, team)


def load_enhanced_logs(season):
    """Load all enhanced game logs for a season."""
    season_dir = os.path.join(ENHANCED_DIR, season)
    
    if not os.path.exists(season_dir):
        return {}
    
    team_logs = {}
    csv_files = sorted([f for f in os.listdir(season_dir) if f.endswith('_enhanced.csv')])
    
    for csv_file in csv_files:
        team = csv_file.split('_')[0]
        filepath = os.path.join(season_dir, csv_file)
        
        try:
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            team_logs[team] = df
        except Exception as e:
            print(f"  ⚠️  Error loading {team} {season}: {e}")
    
    return team_logs


def create_differential_features(team_row, opp_row):
    """Create differential features between team and opponent (from merge_games.py)."""
    diffs = {}
    
    # Net Rating differentials
    if pd.notna(team_row.get('NetRtg_avg')) and pd.notna(opp_row.get('NetRtg_avg')):
        diffs['NetRtg_Diff'] = team_row['NetRtg_avg'] - opp_row['NetRtg_avg']
    
    # Offensive/Defensive Rating differentials
    if pd.notna(team_row.get('ORtg_avg')) and pd.notna(opp_row.get('ORtg_avg')):
        diffs['ORtg_Diff'] = team_row['ORtg_avg'] - opp_row['ORtg_avg']
    if pd.notna(team_row.get('DRtg_avg')) and pd.notna(opp_row.get('DRtg_avg')):
        diffs['DRtg_Diff'] = team_row['DRtg_avg'] - opp_row['DRtg_avg']
    
    # Win percentage differential
    if pd.notna(team_row.get('WinPct_avg')) and pd.notna(opp_row.get('WinPct_avg')):
        diffs['WinPct_Diff'] = team_row['WinPct_avg'] - opp_row['WinPct_avg']
    
    # Recent form differentials
    for window in [5, 10]:
        if pd.notna(team_row.get(f'NetRtg_last{window}')) and pd.notna(opp_row.get(f'NetRtg_last{window}')):
            diffs[f'NetRtg_Diff_last{window}'] = (
                team_row[f'NetRtg_last{window}'] - opp_row[f'NetRtg_last{window}']
            )
        if pd.notna(team_row.get(f'WinPct_last{window}')) and pd.notna(opp_row.get(f'WinPct_last{window}')):
            diffs[f'WinPct_Diff_last{window}'] = (
                team_row[f'WinPct_last{window}'] - opp_row[f'WinPct_last{window}']
            )
    
    # Four Factors differentials
    four_factors = ['eFG%', 'TOV%', 'ORB%', 'FT/FGA']
    for factor in four_factors:
        if pd.notna(team_row.get(f'{factor}_avg')) and pd.notna(opp_row.get(f'{factor}_avg')):
            diffs[f'{factor}_Diff'] = team_row[f'{factor}_avg'] - opp_row[f'{factor}_avg']
    
    # Pace differential
    if pd.notna(team_row.get('Pace_avg')) and pd.notna(opp_row.get('Pace_avg')):
        diffs['Pace_Diff'] = team_row['Pace_avg'] - opp_row['Pace_avg']
    
    # Rest advantage
    team_rest = team_row.get('Days_Rest', 0) if pd.notna(team_row.get('Days_Rest')) else 0
    opp_rest = opp_row.get('Days_Rest', 0) if pd.notna(opp_row.get('Days_Rest')) else 0
    diffs['Rest_Advantage'] = team_rest - opp_rest
    
    # Streak differential
    team_streak = (team_row.get('Current_Win_Streak', 0) if pd.notna(team_row.get('Current_Win_Streak')) else 0) - \
                  (team_row.get('Current_Loss_Streak', 0) if pd.notna(team_row.get('Current_Loss_Streak')) else 0)
    opp_streak = (opp_row.get('Current_Win_Streak', 0) if pd.notna(opp_row.get('Current_Win_Streak')) else 0) - \
                 (opp_row.get('Current_Loss_Streak', 0) if pd.notna(opp_row.get('Current_Loss_Streak')) else 0)
    diffs['Win_Streak_Diff'] = team_streak - opp_streak
    
    return diffs


def get_team_stats_before_date(team, season, before_date, team_logs):
    """
    Get a team's most recent stats from enhanced data before a given date.
    
    Args:
        team: Team abbreviation (standard format)
        season: Season string (e.g., '2025-26')
        before_date: Only get stats from games before this date
        team_logs: Dictionary of team -> DataFrame (enhanced logs)
    
    Returns:
        Series with team's most recent features, or None if no games found
    """
    # Try standard name first, then Basketball-Reference format
    team_br = normalize_team_name(team, to_standard=False)
    
    for team_name in [team, team_br]:
        if team_name not in team_logs:
            continue
        
        df = team_logs[team_name]
        
        # Get games before the prediction date
        df_before = df[df['Date'] < before_date].sort_values('Date')
        
        if len(df_before) > 0:
            # Return the most recent row (last game played)
            return df_before.iloc[-1]
    
    # If no games found, return None
    return None


def create_game_features(home_team, away_team, date, season, team_logs):
    """
    Create game-level features for prediction.
    Uses the same structure as merge_games.py
    """
    # Get team stats before the prediction date
    home_stats = get_team_stats_before_date(home_team, season, date, team_logs)
    away_stats = get_team_stats_before_date(away_team, season, date, team_logs)
    
    # If teams haven't played yet, use defaults
    if home_stats is None:
        home_stats = get_default_stats()
    if away_stats is None:
        away_stats = get_default_stats()
    
    # Calculate game numbers (how many games each team has played)
    home_game_num = home_stats.get('Game_Number', 0) + 1 if home_stats is not None and pd.notna(home_stats.get('Game_Number')) else 1
    away_game_num = away_stats.get('Game_Number', 0) + 1 if away_stats is not None and pd.notna(away_stats.get('Game_Number')) else 1
    
    # Calculate days rest for both teams
    home_days_rest = calculate_days_rest(home_stats, date)
    away_days_rest = calculate_days_rest(away_stats, date)
    
    # Create game row (similar to merge_games.py)
    game_features = {
        'Is_Home': 1,
        'Game_Number_Team': home_game_num,
        'Game_Number_Opp': away_game_num,
        
        # Home team features (Team_ prefix)
        'Team_ORtg_avg': home_stats.get('ORtg_avg', np.nan),
        'Team_DRtg_avg': home_stats.get('DRtg_avg', np.nan),
        'Team_NetRtg_avg': home_stats.get('NetRtg_avg', np.nan),
        'Team_Pace_avg': home_stats.get('Pace_avg', np.nan),
        'Team_WinPct_avg': home_stats.get('WinPct_avg', np.nan),
        'Team_eFG%_avg': home_stats.get('eFG%_avg', np.nan),
        'Team_TOV%_avg': home_stats.get('TOV%_avg', np.nan),
        'Team_ORB%_avg': home_stats.get('ORB%_avg', np.nan),
        'Team_FT/FGA_avg': home_stats.get('FT/FGA_avg', np.nan),
        'Team_TS%_avg': home_stats.get('TS%_avg', np.nan),
        'Team_TRB%_avg': home_stats.get('TRB%_avg', np.nan),
        'Team_AST%_avg': home_stats.get('AST%_avg', np.nan),
        'Team_STL%_avg': home_stats.get('STL%_avg', np.nan),
        'Team_BLK%_avg': home_stats.get('BLK%_avg', np.nan),
        'Team_NetRtg_last5': home_stats.get('NetRtg_last5', np.nan),
        'Team_NetRtg_last10': home_stats.get('NetRtg_last10', np.nan),
        'Team_WinPct_last5': home_stats.get('WinPct_last5', np.nan),
        'Team_WinPct_last10': home_stats.get('WinPct_last10', np.nan),
        'Team_Points_For_avg': home_stats.get('Points_For_avg', home_stats.get('Tm_avg', np.nan)),
        'Team_Points_Against_avg': home_stats.get('Points_Against_avg', home_stats.get('Opp_Score_avg', np.nan)),
        'Team_Current_Win_Streak': home_stats.get('Current_Win_Streak', 0) if pd.notna(home_stats.get('Current_Win_Streak')) else 0,
        'Team_Current_Loss_Streak': home_stats.get('Current_Loss_Streak', 0) if pd.notna(home_stats.get('Current_Loss_Streak')) else 0,
        'Team_Home_WinPct': home_stats.get('Home_WinPct', np.nan),
        'Team_Away_WinPct': np.nan,  # Home team, so away win pct not applicable
        'Team_Days_Rest': home_days_rest,
        'Team_Is_BackToBack': 1 if home_days_rest == 0 else 0,
        
        # Away team features (Opp_ prefix)
        'Opp_ORtg_avg': away_stats.get('ORtg_avg', np.nan),
        'Opp_DRtg_avg': away_stats.get('DRtg_avg', np.nan),
        'Opp_NetRtg_avg': away_stats.get('NetRtg_avg', np.nan),
        'Opp_Pace_avg': away_stats.get('Pace_avg', np.nan),
        'Opp_WinPct_avg': away_stats.get('WinPct_avg', np.nan),
        'Opp_eFG%_avg': away_stats.get('eFG%_avg', np.nan),
        'Opp_TOV%_avg': away_stats.get('TOV%_avg', np.nan),
        'Opp_ORB%_avg': away_stats.get('ORB%_avg', np.nan),
        'Opp_FT/FGA_avg': away_stats.get('FT/FGA_avg', np.nan),
        'Opp_TS%_avg': away_stats.get('TS%_avg', np.nan),
        'Opp_TRB%_avg': away_stats.get('TRB%_avg', np.nan),
        'Opp_AST%_avg': away_stats.get('AST%_avg', np.nan),
        'Opp_STL%_avg': away_stats.get('STL%_avg', np.nan),
        'Opp_BLK%_avg': away_stats.get('BLK%_avg', np.nan),
        'Opp_NetRtg_last5': away_stats.get('NetRtg_last5', np.nan),
        'Opp_NetRtg_last10': away_stats.get('NetRtg_last10', np.nan),
        'Opp_WinPct_last5': away_stats.get('WinPct_last5', np.nan),
        'Opp_WinPct_last10': away_stats.get('WinPct_last10', np.nan),
        'Opp_Points_For_avg': away_stats.get('Points_For_avg', away_stats.get('Tm_avg', np.nan)),
        'Opp_Points_Against_avg': away_stats.get('Points_Against_avg', away_stats.get('Opp_Score_avg', np.nan)),
        'Opp_Current_Win_Streak': away_stats.get('Current_Win_Streak', 0) if pd.notna(away_stats.get('Current_Win_Streak')) else 0,
        'Opp_Current_Loss_Streak': away_stats.get('Current_Loss_Streak', 0) if pd.notna(away_stats.get('Current_Loss_Streak')) else 0,
        'Opp_Home_WinPct': np.nan,  # Away team, so home win pct not applicable
        'Opp_Away_WinPct': away_stats.get('Away_WinPct', np.nan),
        'Opp_Days_Rest': away_days_rest,
        'Opp_Is_BackToBack': 1 if away_days_rest == 0 else 0,
    }
    
    # Add differential features
    diffs = create_differential_features(home_stats, away_stats)
    game_features.update(diffs)
    
    return game_features


def get_default_stats():
    """Return default stats for teams that haven't played yet."""
    return {
        'ORtg_avg': np.nan, 'DRtg_avg': np.nan, 'NetRtg_avg': np.nan,
        'Pace_avg': np.nan, 'WinPct_avg': np.nan, 'eFG%_avg': np.nan,
        'TOV%_avg': np.nan, 'ORB%_avg': np.nan, 'FT/FGA_avg': np.nan,
        'TS%_avg': np.nan, 'TRB%_avg': np.nan, 'AST%_avg': np.nan,
        'STL%_avg': np.nan, 'BLK%_avg': np.nan,
        'NetRtg_last5': np.nan, 'NetRtg_last10': np.nan,
        'WinPct_last5': np.nan, 'WinPct_last10': np.nan,
        'Points_For_avg': np.nan, 'Points_Against_avg': np.nan,
        'Current_Win_Streak': 0, 'Current_Loss_Streak': 0,
        'Home_WinPct': np.nan, 'Away_WinPct': np.nan,
        'Game_Number': 0, 'Date': None
    }


def calculate_days_rest(team_stats, game_date):
    """Calculate days rest since last game."""
    if team_stats is None or pd.isna(team_stats.get('Date')):
        return 1  # Default to 1 day rest
    
    last_game_date = pd.to_datetime(team_stats['Date'])
    days_rest = (game_date - last_game_date).days
    return max(1, days_rest)  # Minimum 1 day rest (can't be 0 or negative for upcoming games)


def load_model(model_type='classification'):
    """
    Load the saved XGBoost model and metadata.
    
    Args:
        model_type: 'classification' or 'regression'
    
    Returns:
        (booster, metadata) tuple
    """
    if model_type == 'regression':
        model_path = os.path.join(MODEL_DIR, 'xgb_regression_final.model')
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata_regression.pkl')
    else:
        model_path = os.path.join(MODEL_DIR, 'xgb_classifier_final.model')
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train_model.py (and train_model_regression.py) first to save the model."
        )
    
    # Load model
    booster = xgb.Booster()
    booster.load_model(model_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return booster, metadata


def predict_games_for_date(date_str, season='2025-26', model_type='classification'):
    """
    Main function to predict games for a given date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        season: Season string (e.g., '2025-26')
        model_type: 'classification' or 'regression'
    """
    # Load schedule
    if not os.path.exists(SCHEDULE_FILE):
        print(f"❌ Schedule file not found: {SCHEDULE_FILE}")
        print(f"   Run scrape_schedule.py first to create the schedule file.")
        return
    
    schedule = pd.read_csv(SCHEDULE_FILE)
    schedule['Date'] = pd.to_datetime(schedule['Date'])
    
    date = pd.to_datetime(date_str)
    games = schedule[schedule['Date'].dt.date == date.date()]
    
    if len(games) == 0:
        print(f"No games found for {date_str}")
        print(f"Make sure {SCHEDULE_FILE} has games for this date")
        return
    
    # Load enhanced team data
    print(f"\nLoading enhanced team data for {season}...")
    team_logs = load_enhanced_logs(season)
    
    if not team_logs:
        print(f"❌ No enhanced team data found for {season}")
        print(f"   Run feature_engineering.py first to create enhanced data.")
        return
    
    print(f"   Loaded data for {len(team_logs)} teams")
    
    # Load model
    print(f"\nLoading {model_type} model...")
    try:
        booster, metadata = load_model(model_type)
        feature_names = metadata['feature_names']
        print(f"   Model trained on seasons: {metadata['train_seasons']}")
        print(f"   Features: {len(feature_names)}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Prepare features for each game
    print(f"\nPreparing features for {len(games)} games on {date_str}...")
    game_features_list = []
    
    for idx, game in games.iterrows():
        home_team = normalize_team_name(game['Home_Team'], to_standard=True)
        away_team = normalize_team_name(game['Away_Team'], to_standard=True)
        
        print(f"  {home_team} vs {away_team}...", end=' ', flush=True)
        
        try:
            features = create_game_features(home_team, away_team, date, season, team_logs)
            features['Home_Team'] = home_team
            features['Away_Team'] = away_team
            features['Date'] = date_str
            game_features_list.append(features)
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")
            continue
    
    if not game_features_list:
        print("\n❌ No games could be processed")
        return
    
    # Create DataFrame
    df_features = pd.DataFrame(game_features_list)
    
    # Align with model's expected features
    X_pred = df_features.reindex(columns=feature_names, fill_value=0)
    
    # Fill NaN values
    X_pred = X_pred.fillna(0)
    
    # Make predictions
    print("\nMaking predictions...")
    dtest = xgb.DMatrix(X_pred)
    
    if model_type == 'regression':
        # Regression: predict point differential
        point_diff_pred = booster.predict(dtest)
        # Convert to win probability using sigmoid transformation
        scale = 10.0
        probabilities = 1 / (1 + np.exp(-point_diff_pred / scale))
        win_pred = (point_diff_pred > 0).astype(int)
    else:
        # Classification: direct win probability
        probabilities = booster.predict(dtest)
        win_pred = (probabilities >= 0.5).astype(int)
    
    # Create results
    results = pd.DataFrame({
        'Date': df_features['Date'],
        'Home_Team': df_features['Home_Team'],
        'Away_Team': df_features['Away_Team'],
        'Home_Win_Probability': probabilities,
        'Away_Win_Probability': 1 - probabilities,
        'Predicted_Winner': df_features['Home_Team'].where(win_pred == 1, df_features['Away_Team']),
        'Confidence': np.abs(probabilities - 0.5) * 2,  # 0 to 1 scale
    })
    
    if model_type == 'regression':
        results['Predicted_Point_Diff'] = point_diff_pred
    
    # Sort by confidence (most confident first)
    results = results.sort_values('Confidence', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print(f"PREDICTIONS FOR {date_str} ({model_type.upper()} MODEL)")
    print("="*80)
    
    if model_type == 'regression':
        print(f"\n{'Home':<6} vs {'Away':<6} | Winner | Home% | Away% | Point Diff | Confidence")
        print("-" * 80)
        for _, row in results.iterrows():
            home = row['Home_Team']
            away = row['Away_Team']
            winner = row['Predicted_Winner']
            home_pct = row['Home_Win_Probability'] * 100
            away_pct = row['Away_Win_Probability'] * 100
            point_diff = row['Predicted_Point_Diff']
            conf = row['Confidence'] * 100
            print(f"{home:<6} vs {away:<6} | {winner:<6} | {home_pct:5.1f}% | {away_pct:5.1f}% | {point_diff:+6.1f} | {conf:5.1f}%")
    else:
        print(f"\n{'Home':<6} vs {'Away':<6} | Winner | Home% | Away% | Confidence")
        print("-" * 80)
        for _, row in results.iterrows():
            home = row['Home_Team']
            away = row['Away_Team']
            winner = row['Predicted_Winner']
            home_pct = row['Home_Win_Probability'] * 100
            away_pct = row['Away_Win_Probability'] * 100
            conf = row['Confidence'] * 100
            print(f"{home:<6} vs {away:<6} | {winner:<6} | {home_pct:5.1f}% | {away_pct:5.1f}% | {conf:5.1f}%")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"predictions_{date_str.replace('-', '')}.csv")
    results.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict NBA games for a given date')
    parser.add_argument('date', type=str, nargs='?', help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--model', type=str, choices=['classification', 'regression'], 
                       default='classification', help='Model type to use (default: classification)')
    parser.add_argument('--season', type=str, default='2025-26', help='Season string (default: 2025-26)')
    
    args = parser.parse_args()
    
    # Get date from command line or use today
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n🏀 NBA Game Predictions")
    print(f"Date: {date_str}")
    print(f"Model: {args.model}")
    print(f"Season: {args.season}\n")
    
    predict_games_for_date(date_str, season=args.season, model_type=args.model)

