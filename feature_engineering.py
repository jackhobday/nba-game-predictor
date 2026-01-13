#!/usr/bin/env python3
"""
Feature Engineering Script for NBA Game Logs

Calculates rolling statistics, streaks, and other features for each team's game log.
Enhanced game logs are saved to a new directory for merging.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import glob

# Directories
DATA_DIR = 'data'
ENHANCED_DIR = 'data_enhanced'

# Columns to calculate rolling averages for
STAT_COLUMNS = [
    'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%',
    'eFG%', 'TOV%', 'ORB%', 'FT/FGA'
]

# Opponent stat columns (defensive stats)
OPP_STAT_COLUMNS = [
    'eFG%', 'TOV%', 'ORB%', 'FT/FGA'
]


def parse_date(date_str):
    """Parse date string to datetime object."""
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def calculate_rolling_features(df):
    """
    Calculate rolling statistics and features for a team's game log.
    
    Args:
        df: DataFrame with team game log
        
    Returns:
        DataFrame with added features
    """
    df = df.copy()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Parse result (W/L)
    df['Result'] = df['Rslt'].apply(lambda x: 1 if x == 'W' else 0)
    df['Is_Win'] = df['Result']
    
    # Determine home/away (blank = home, '@' = away)
    # Column index 3 is the location column (named 'nan' in CSV)
    if len(df.columns) > 3:
        location_col = df.columns[3]
        # Convert to string, strip whitespace: '@' = away (0), blank/NaN = home (1)
        df['Is_Home'] = df[location_col].astype(str).str.strip().apply(lambda x: 0 if x == '@' else 1)
    else:
        # Fallback: assume all home if we can't determine
        df['Is_Home'] = 1
    
    # Calculate Net Rating
    df['NetRtg'] = df['ORtg'] - df['DRtg']
    
    # Game number in season
    df['Game_Number'] = df['Gtm']
    
    # Days rest (days since last game)
    df['Days_Rest'] = 0
    for i in range(1, len(df)):
        days_diff = (df.loc[i, 'Date'] - df.loc[i-1, 'Date']).days
        df.loc[i, 'Days_Rest'] = days_diff
    
    # Is back-to-back (playing consecutive days)
    df['Is_BackToBack'] = (df['Days_Rest'] == 0).astype(int)
    
    # ===== SEASON-TO-DATE AVERAGES (Cumulative) =====
    for col in STAT_COLUMNS:
        if col in df.columns:
            df[f'{col}_avg'] = df[col].expanding().mean()
    
    # Net Rating average
    df['NetRtg_avg'] = df['NetRtg'].expanding().mean()
    
    # Win percentage (season-to-date)
    df['WinPct_avg'] = df['Result'].expanding().mean()
    
    # Points scored/allowed averages
    # Note: CSV has two 'Opp' columns - 'Opp' is opponent abbreviation (string), 'Opp.1' is opponent score (numeric)
    df['Tm_avg'] = df['Tm'].expanding().mean()
    
    # Use 'Opp.1' for opponent score (pandas renames duplicate columns)
    opp_score_col = 'Opp.1' if 'Opp.1' in df.columns else None
    if opp_score_col and pd.api.types.is_numeric_dtype(df[opp_score_col]):
        df['Opp_Score_avg'] = df[opp_score_col].expanding().mean()
        df['Point_Diff_avg'] = (df['Tm'] - df[opp_score_col]).expanding().mean()
    else:
        # Fallback if column structure is different
        df['Opp_Score_avg'] = np.nan
        df['Point_Diff_avg'] = np.nan
    
    # ===== RECENT FORM (Last N Games) =====
    windows = [5, 10]
    for window in windows:
        for col in STAT_COLUMNS:
            if col in df.columns:
                df[f'{col}_last{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        df[f'NetRtg_last{window}'] = df['NetRtg'].rolling(window=window, min_periods=1).mean()
        df[f'WinPct_last{window}'] = df['Result'].rolling(window=window, min_periods=1).mean()
        
        # Use 'Opp.1' for opponent score
        if opp_score_col and pd.api.types.is_numeric_dtype(df[opp_score_col]):
            df[f'Point_Diff_last{window}'] = (df['Tm'] - df[opp_score_col]).rolling(window=window, min_periods=1).mean()
        else:
            df[f'Point_Diff_last{window}'] = np.nan
    
    # ===== WIN/LOSS STREAKS =====
    df['Current_Win_Streak'] = 0
    df['Current_Loss_Streak'] = 0
    df['Longest_Win_Streak'] = 0
    df['Longest_Loss_Streak'] = 0
    
    current_win = 0
    current_loss = 0
    longest_win = 0
    longest_loss = 0
    
    for i in range(len(df)):
        if df.loc[i, 'Result'] == 1:  # Win
            current_win += 1
            current_loss = 0
            longest_win = max(longest_win, current_win)
        else:  # Loss
            current_loss += 1
            current_win = 0
            longest_loss = max(longest_loss, current_loss)
        
        df.loc[i, 'Current_Win_Streak'] = current_win
        df.loc[i, 'Current_Loss_Streak'] = current_loss
        df.loc[i, 'Longest_Win_Streak'] = longest_win
        df.loc[i, 'Longest_Loss_Streak'] = longest_loss
    
    # ===== HOME/AWAY SPLITS =====
    # Home stats (cumulative)
    home_mask = df['Is_Home'] == 1
    away_mask = df['Is_Home'] == 0
    
    df['Home_ORtg_avg'] = np.nan
    df['Home_DRtg_avg'] = np.nan
    df['Home_NetRtg_avg'] = np.nan
    df['Home_WinPct'] = np.nan
    df['Home_Games'] = 0
    
    df['Away_ORtg_avg'] = np.nan
    df['Away_DRtg_avg'] = np.nan
    df['Away_NetRtg_avg'] = np.nan
    df['Away_WinPct'] = np.nan
    df['Away_Games'] = 0
    
    # Calculate cumulative home/away stats
    home_wins = 0
    home_games = 0
    away_wins = 0
    away_games = 0
    
    home_ortg_sum = 0
    home_drtg_sum = 0
    away_ortg_sum = 0
    away_drtg_sum = 0
    
    for i in range(len(df)):
        if df.loc[i, 'Is_Home'] == 1:
            home_games += 1
            home_wins += df.loc[i, 'Result']
            home_ortg_sum += df.loc[i, 'ORtg']
            home_drtg_sum += df.loc[i, 'DRtg']
            
            df.loc[i, 'Home_Games'] = home_games
            df.loc[i, 'Home_WinPct'] = home_wins / home_games if home_games > 0 else 0
            df.loc[i, 'Home_ORtg_avg'] = home_ortg_sum / home_games if home_games > 0 else 0
            df.loc[i, 'Home_DRtg_avg'] = home_drtg_sum / home_games if home_games > 0 else 0
            df.loc[i, 'Home_NetRtg_avg'] = df.loc[i, 'Home_ORtg_avg'] - df.loc[i, 'Home_DRtg_avg']
        else:
            away_games += 1
            away_wins += df.loc[i, 'Result']
            away_ortg_sum += df.loc[i, 'ORtg']
            away_drtg_sum += df.loc[i, 'DRtg']
            
            df.loc[i, 'Away_Games'] = away_games
            df.loc[i, 'Away_WinPct'] = away_wins / away_games if away_games > 0 else 0
            df.loc[i, 'Away_ORtg_avg'] = away_ortg_sum / away_games if away_games > 0 else 0
            df.loc[i, 'Away_DRtg_avg'] = away_drtg_sum / away_games if away_games > 0 else 0
            df.loc[i, 'Away_NetRtg_avg'] = df.loc[i, 'Away_ORtg_avg'] - df.loc[i, 'Away_DRtg_avg']
    
    # Fill forward for games where split doesn't apply yet
    df['Home_ORtg_avg'] = df['Home_ORtg_avg'].ffill().fillna(0)
    df['Home_DRtg_avg'] = df['Home_DRtg_avg'].ffill().fillna(0)
    df['Home_NetRtg_avg'] = df['Home_NetRtg_avg'].ffill().fillna(0)
    df['Home_WinPct'] = df['Home_WinPct'].ffill().fillna(0)
    df['Away_ORtg_avg'] = df['Away_ORtg_avg'].ffill().fillna(0)
    df['Away_DRtg_avg'] = df['Away_DRtg_avg'].ffill().fillna(0)
    df['Away_NetRtg_avg'] = df['Away_NetRtg_avg'].ffill().fillna(0)
    df['Away_WinPct'] = df['Away_WinPct'].ffill().fillna(0)
    
    # ===== OPPONENT STATS (for later matchup analysis) =====
    # These are already in the data as defensive stats, but we'll keep them accessible
    # The last 4 columns are opponent offensive four factors
    
    return df


def process_team_file(filepath, season, team):
    """
    Process a single team's game log file.
    
    Args:
        filepath: Path to the CSV file
        season: Season string (e.g., '2024-25')
        team: Team abbreviation
        
    Returns:
        Success status
    """
    try:
        # Read the CSV
        df = pd.read_csv(filepath)
        
        if df.empty:
            print(f"  ⚠️  {team} {season}: Empty file")
            return False
        
        # Add team identifier
        df['Team'] = team
        df['Season'] = season
        
        # Calculate features
        df_enhanced = calculate_rolling_features(df)
        
        # Round all numeric columns to 2 decimal places
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        # Exclude integer columns (like Rk, Gtm, Game_Number, Result, Is_Home, etc.)
        integer_cols = ['Rk', 'Gtm', 'Game_Number', 'Result', 'Is_Win', 'Is_Home', 
                       'Days_Rest', 'Is_BackToBack', 'Current_Win_Streak', 
                       'Current_Loss_Streak', 'Longest_Win_Streak', 'Longest_Loss_Streak',
                       'Home_Games', 'Away_Games', 'OT']
        cols_to_round = [col for col in numeric_cols if col not in integer_cols]
        df_enhanced[cols_to_round] = df_enhanced[cols_to_round].round(2)
        
        # Create output directory
        output_dir = os.path.join(ENHANCED_DIR, season)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save enhanced file
        output_path = os.path.join(output_dir, f"{team}_{season}_enhanced.csv")
        df_enhanced.to_csv(output_path, index=False)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {team} {season}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_teams():
    """
    Process all team game log files and add features.
    """
    print("\n" + "="*70)
    print("  FEATURE ENGINEERING - NBA GAME LOGS")
    print("="*70 + "\n")
    
    # Get all seasons
    seasons = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    total_files = 0
    successful = 0
    failed = 0
    
    for season in seasons:
        season_dir = os.path.join(DATA_DIR, season)
        csv_files = sorted([f for f in os.listdir(season_dir) if f.endswith('.csv')])
        
        print(f"📅 Season: {season}")
        print("-" * 70)
        
        for csv_file in csv_files:
            total_files += 1
            
            # Extract team abbreviation from filename
            team = csv_file.split('_')[0]
            
            filepath = os.path.join(season_dir, csv_file)
            
            if process_team_file(filepath, season, team):
                successful += 1
                print(f"  ✓ {team} {season}: Enhanced")
            else:
                failed += 1
        
        print()
    
    # Summary
    print("="*70)
    print("  FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"Total files:  {total_files}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed:     {failed}")
    print(f"\nEnhanced files saved to: {ENHANCED_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    process_all_teams()

