#!/usr/bin/env python3
"""
Merge Team Game Logs into Game-Level Dataset

Matches team game logs to create game-level rows with both teams' features,
differential features, and target variable (winner).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Directories
ENHANCED_DIR = 'data_enhanced'
OUTPUT_DIR = 'data_merged'

# Team name mapping (Basketball-Reference to standard)
TEAM_MAPPING = {
    'BRK': 'BKN',  # Brooklyn Nets
    'CHO': 'CHA',  # Charlotte Hornets
    'PHO': 'PHX',  # Phoenix Suns
}

# Reverse mapping (standard to Basketball-Reference)
REVERSE_TEAM_MAPPING = {v: k for k, v in TEAM_MAPPING.items()}


def normalize_team_name(team, to_standard=True):
    """
    Normalize team name between standard and Basketball-Reference formats.
    
    Args:
        team: Team abbreviation
        to_standard: If True, convert to standard format (BKN, CHA, PHX)
                    If False, convert to Basketball-Reference format (BRK, CHO, PHO)
    
    Returns:
        Normalized team name
    """
    if to_standard:
        return TEAM_MAPPING.get(team, team)
    else:
        return REVERSE_TEAM_MAPPING.get(team, team)


def load_enhanced_logs(season):
    """
    Load all enhanced game logs for a season.
    
    Args:
        season: Season string (e.g., '2024-25')
        
    Returns:
        Dictionary mapping team -> DataFrame
    """
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
    """
    Create differential features between team and opponent.
    
    Args:
        team_row: Series with team's features
        opp_row: Series with opponent's features
        
    Returns:
        Dictionary of differential features
    """
    diffs = {}
    
    # Net Rating differentials
    if 'NetRtg_avg' in team_row and 'NetRtg_avg' in opp_row:
        diffs['NetRtg_Diff'] = team_row['NetRtg_avg'] - opp_row['NetRtg_avg']
    
    # Offensive/Defensive Rating differentials
    if 'ORtg_avg' in team_row and 'ORtg_avg' in opp_row:
        diffs['ORtg_Diff'] = team_row['ORtg_avg'] - opp_row['ORtg_avg']
    if 'DRtg_avg' in team_row and 'DRtg_avg' in opp_row:
        diffs['DRtg_Diff'] = team_row['DRtg_avg'] - opp_row['DRtg_avg']
    
    # Win percentage differential
    if 'WinPct_avg' in team_row and 'WinPct_avg' in opp_row:
        diffs['WinPct_Diff'] = team_row['WinPct_avg'] - opp_row['WinPct_avg']
    
    # Recent form differentials
    for window in [5, 10]:
        if f'NetRtg_last{window}' in team_row and f'NetRtg_last{window}' in opp_row:
            diffs[f'NetRtg_Diff_last{window}'] = (
                team_row[f'NetRtg_last{window}'] - opp_row[f'NetRtg_last{window}']
            )
        if f'WinPct_last{window}' in team_row and f'WinPct_last{window}' in opp_row:
            diffs[f'WinPct_Diff_last{window}'] = (
                team_row[f'WinPct_last{window}'] - opp_row[f'WinPct_last{window}']
            )
    
    # Four Factors differentials
    four_factors = ['eFG%', 'TOV%', 'ORB%', 'FT/FGA']
    for factor in four_factors:
        if f'{factor}_avg' in team_row and f'{factor}_avg' in opp_row:
            diffs[f'{factor}_Diff'] = team_row[f'{factor}_avg'] - opp_row[f'{factor}_avg']
    
    # Pace differential
    if 'Pace_avg' in team_row and 'Pace_avg' in opp_row:
        diffs['Pace_Diff'] = team_row['Pace_avg'] - opp_row['Pace_avg']
    
    # Rest advantage
    if 'Days_Rest' in team_row and 'Days_Rest' in opp_row:
        diffs['Rest_Advantage'] = team_row['Days_Rest'] - opp_row['Days_Rest']
    
    # Streak differential
    if 'Current_Win_Streak' in team_row and 'Current_Win_Streak' in opp_row:
        diffs['Win_Streak_Diff'] = (
            team_row['Current_Win_Streak'] - opp_row['Current_Win_Streak']
        )
    
    return diffs


def merge_season_games(season):
    """
    Merge all games for a single season.
    
    Args:
        season: Season string (e.g., '2024-25')
        
    Returns:
        DataFrame with merged games
    """
    print(f"📅 Merging games for {season}...")
    
    # Load all team logs
    team_logs = load_enhanced_logs(season)
    
    if not team_logs:
        print(f"  ⚠️  No enhanced logs found for {season}")
        return None
    
    merged_games = []
    games_processed = 0
    games_matched = 0
    games_failed = 0
    
    # Process each team's games
    for team, team_df in team_logs.items():
        # Normalize team name to standard format for consistent comparison
        team_standard = normalize_team_name(team, to_standard=True)
        
        for idx, team_row in team_df.iterrows():
            games_processed += 1
            
            date = team_row['Date']
            opp_abbr = team_row['Opp']
            
            # Normalize opponent name (might be in Basketball-Reference format)
            opp_standard = normalize_team_name(opp_abbr, to_standard=True)
            opp_br = normalize_team_name(opp_standard, to_standard=False)
            
            # Only process games where Team < Opponent alphabetically to avoid duplicates
            # This ensures we only create one row per game
            if team_standard >= opp_standard:
                continue  # Skip this game - it will be processed from the opponent's perspective
            
            # Try to find opponent's log (check both standard and BR formats)
            opp_log = None
            if opp_standard in team_logs:
                opp_log = team_logs[opp_standard]
            elif opp_br in team_logs:
                opp_log = team_logs[opp_br]
            
            if opp_log is None:
                games_failed += 1
                continue
            
            # Find opponent's row for this game (same date, opponent is this team)
            # Check both standard and BR formats for the team name
            team_br = normalize_team_name(team_standard, to_standard=False)
            opp_rows = opp_log[
                (opp_log['Date'] == date) & 
                (opp_log['Opp'].isin([team_standard, team_br, team]))
            ]
            
            if len(opp_rows) == 0:
                games_failed += 1
                continue
            
            opp_row = opp_rows.iloc[0]
            games_matched += 1
            
            # Create game row
            game_row = {
                # Game metadata
                'Season': season,
                'Date': date,
                'Team': team_standard,  # Use normalized team name for consistency
                'Opponent': opp_standard,
                'Is_Home': team_row['Is_Home'],
                'Game_Number_Team': team_row['Game_Number'],
                'Game_Number_Opp': opp_row['Game_Number'],
                
                # Team features (prefix with Team_)
                'Team_ORtg_avg': team_row.get('ORtg_avg', np.nan),
                'Team_DRtg_avg': team_row.get('DRtg_avg', np.nan),
                'Team_NetRtg_avg': team_row.get('NetRtg_avg', np.nan),
                'Team_Pace_avg': team_row.get('Pace_avg', np.nan),
                'Team_WinPct_avg': team_row.get('WinPct_avg', np.nan),
                'Team_eFG%_avg': team_row.get('eFG%_avg', np.nan),
                'Team_TOV%_avg': team_row.get('TOV%_avg', np.nan),
                'Team_ORB%_avg': team_row.get('ORB%_avg', np.nan),
                'Team_FT/FGA_avg': team_row.get('FT/FGA_avg', np.nan),
                'Team_TS%_avg': team_row.get('TS%_avg', np.nan),
                'Team_TRB%_avg': team_row.get('TRB%_avg', np.nan),
                'Team_AST%_avg': team_row.get('AST%_avg', np.nan),
                'Team_STL%_avg': team_row.get('STL%_avg', np.nan),
                'Team_BLK%_avg': team_row.get('BLK%_avg', np.nan),
                'Team_NetRtg_last5': team_row.get('NetRtg_last5', np.nan),
                'Team_NetRtg_last10': team_row.get('NetRtg_last10', np.nan),
                'Team_WinPct_last5': team_row.get('WinPct_last5', np.nan),
                'Team_WinPct_last10': team_row.get('WinPct_last10', np.nan),
                'Team_Points_For_avg': team_row.get('Points_For_avg', team_row.get('Tm_avg', np.nan)),
                'Team_Points_Against_avg': team_row.get('Points_Against_avg', team_row.get('Opp_Score_avg', np.nan)),
                'Team_Current_Win_Streak': team_row.get('Current_Win_Streak', 0),
                'Team_Current_Loss_Streak': team_row.get('Current_Loss_Streak', 0),
                'Team_Home_WinPct': team_row.get('Home_WinPct', np.nan) if team_row['Is_Home'] == 1 else np.nan,
                'Team_Away_WinPct': team_row.get('Away_WinPct', np.nan) if team_row['Is_Home'] == 0 else np.nan,
                'Team_Days_Rest': team_row.get('Days_Rest', 0),
                'Team_Is_BackToBack': team_row.get('Is_BackToBack', 0),
                
                # Opponent features (prefix with Opp_)
                'Opp_ORtg_avg': opp_row.get('ORtg_avg', np.nan),
                'Opp_DRtg_avg': opp_row.get('DRtg_avg', np.nan),
                'Opp_NetRtg_avg': opp_row.get('NetRtg_avg', np.nan),
                'Opp_Pace_avg': opp_row.get('Pace_avg', np.nan),
                'Opp_WinPct_avg': opp_row.get('WinPct_avg', np.nan),
                'Opp_eFG%_avg': opp_row.get('eFG%_avg', np.nan),
                'Opp_TOV%_avg': opp_row.get('TOV%_avg', np.nan),
                'Opp_ORB%_avg': opp_row.get('ORB%_avg', np.nan),
                'Opp_FT/FGA_avg': opp_row.get('FT/FGA_avg', np.nan),
                'Opp_TS%_avg': opp_row.get('TS%_avg', np.nan),
                'Opp_TRB%_avg': opp_row.get('TRB%_avg', np.nan),
                'Opp_AST%_avg': opp_row.get('AST%_avg', np.nan),
                'Opp_STL%_avg': opp_row.get('STL%_avg', np.nan),
                'Opp_BLK%_avg': opp_row.get('BLK%_avg', np.nan),
                'Opp_NetRtg_last5': opp_row.get('NetRtg_last5', np.nan),
                'Opp_NetRtg_last10': opp_row.get('NetRtg_last10', np.nan),
                'Opp_WinPct_last5': opp_row.get('WinPct_last5', np.nan),
                'Opp_WinPct_last10': opp_row.get('WinPct_last10', np.nan),
                'Opp_Points_For_avg': opp_row.get('Points_For_avg', opp_row.get('Tm_avg', np.nan)),
                'Opp_Points_Against_avg': opp_row.get('Points_Against_avg', opp_row.get('Opp_Score_avg', np.nan)),
                'Opp_Current_Win_Streak': opp_row.get('Current_Win_Streak', 0),
                'Opp_Current_Loss_Streak': opp_row.get('Current_Loss_Streak', 0),
                'Opp_Home_WinPct': opp_row.get('Home_WinPct', np.nan) if opp_row['Is_Home'] == 1 else np.nan,
                'Opp_Away_WinPct': opp_row.get('Away_WinPct', np.nan) if opp_row['Is_Home'] == 0 else np.nan,
                'Opp_Days_Rest': opp_row.get('Days_Rest', 0),
                'Opp_Is_BackToBack': opp_row.get('Is_BackToBack', 0),
                
                # Game outcome
                # Note: 'Opp' is opponent abbreviation, 'Opp.1' is opponent score
                'Team_Score': team_row['Tm'],
                'Opp_Score': team_row.get('Opp.1', team_row.get('Opp', np.nan)),  # Use Opp.1 for score, fallback to Opp
                'Result': team_row['Result'],  # 1 if Team wins, 0 if Opponent wins
                'OT': 1 if pd.notna(team_row.get('OT')) and str(team_row['OT']).strip() != '' else 0,
            }
            
            # Add differential features
            diffs = create_differential_features(team_row, opp_row)
            game_row.update(diffs)
            
            merged_games.append(game_row)
    
    if not merged_games:
        print(f"  ⚠️  No games merged for {season}")
        return None
    
    df_merged = pd.DataFrame(merged_games)
    
    print(f"  ✓ Processed {games_processed} team-game rows")
    print(f"  ✓ Matched {games_matched} games")
    print(f"  ✗ Failed to match {games_failed} games")
    print(f"  → Created {len(df_merged)} game-level rows")
    
    return df_merged


def merge_all_seasons():
    """
    Merge games for all seasons and create final dataset.
    """
    print("\n" + "="*70)
    print("  MERGING TEAM GAME LOGS INTO GAME-LEVEL DATASET")
    print("="*70 + "\n")
    
    # Get all seasons
    if not os.path.exists(ENHANCED_DIR):
        print(f"❌ Enhanced data directory not found: {ENHANCED_DIR}")
        print("   Please run feature_engineering.py first!")
        return
    
    seasons = sorted([d for d in os.listdir(ENHANCED_DIR) if os.path.isdir(os.path.join(ENHANCED_DIR, d))])
    
    if not seasons:
        print("❌ No seasons found in enhanced data directory")
        return
    
    all_games = []
    
    for season in seasons:
        df_season = merge_season_games(season)
        if df_season is not None and len(df_season) > 0:
            all_games.append(df_season)
        print()
    
    if not all_games:
        print("❌ No games were merged")
        return
    
    # Combine all seasons
    df_final = pd.concat(all_games, ignore_index=True)
    
    # Sort by date
    df_final = df_final.sort_values(['Season', 'Date']).reset_index(drop=True)
    
    # Round all numeric columns to 2 decimal places (excluding integer columns)
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    integer_cols = ['Result', 'Is_Home', 'Game_Number_Team', 'Game_Number_Opp', 
                   'Team_Current_Win_Streak', 'Team_Current_Loss_Streak',
                   'Opp_Current_Win_Streak', 'Opp_Current_Loss_Streak',
                   'Team_Days_Rest', 'Opp_Days_Rest', 'Team_Is_BackToBack', 
                   'Opp_Is_BackToBack', 'Team_Score', 'Opp_Score', 'OT']
    cols_to_round = [col for col in numeric_cols if col not in integer_cols]
    df_final[cols_to_round] = df_final[cols_to_round].round(2)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save individual season files (already rounded above)
    for season in seasons:
        df_season = df_final[df_final['Season'] == season]
        if len(df_season) > 0:
            output_path = os.path.join(OUTPUT_DIR, f"games_{season}.csv")
            df_season.to_csv(output_path, index=False)
            print(f"💾 Saved {len(df_season)} games to games_{season}.csv")
    
    # Save combined file
    output_path = os.path.join(OUTPUT_DIR, "games_all_seasons.csv")
    df_final.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("  MERGING COMPLETE")
    print("="*70)
    print(f"Total games: {len(df_final)}")
    print(f"Seasons: {len(seasons)}")
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print(f"  - games_all_seasons.csv ({len(df_final)} games)")
    for season in seasons:
        count = len(df_final[df_final['Season'] == season])
        if count > 0:
            print(f"  - games_{season}.csv ({count} games)")
    print("="*70 + "\n")
    
    # Feature summary
    print("📊 Feature Summary:")
    print(f"  - Total features: {len(df_final.columns)}")
    print(f"  - Team features: {len([c for c in df_final.columns if c.startswith('Team_')])}")
    print(f"  - Opponent features: {len([c for c in df_final.columns if c.startswith('Opp_')])}")
    print(f"  - Differential features: {len([c for c in df_final.columns if 'Diff' in c or 'Advantage' in c])}")
    print(f"  - Target variable: Result (1 = Team wins, 0 = Opponent wins)")
    print()


if __name__ == "__main__":
    merge_all_seasons()

