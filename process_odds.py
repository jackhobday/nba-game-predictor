#!/usr/bin/env python3
"""
Process Kaggle NBA Odds Data

Converts the Kaggle odds dataset (https://www.kaggle.com/datasets/christophertreasure/nba-odds-data)
into a format compatible with our game data pipeline.

The Kaggle data covers seasons from 2007-08 through mid-2022-23.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Input/Output
KAGGLE_ODDS_PATH = 'odds_data/oddsData.csv'
OUTPUT_PATH = 'odds_data/odds_processed.csv'

# Team name mapping: Kaggle full names -> Project 3-letter abbreviations
TEAM_NAME_TO_ABBR = {
    'Atlanta': 'ATL',
    'Boston': 'BOS',
    'Brooklyn': 'BKN',
    'Charlotte': 'CHA',
    'Chicago': 'CHI',
    'Cleveland': 'CLE',
    'Dallas': 'DAL',
    'Denver': 'DEN',
    'Detroit': 'DET',
    'Golden State': 'GSW',
    'Houston': 'HOU',
    'Indiana': 'IND',
    'LA Clippers': 'LAC',
    'LA Lakers': 'LAL',
    'Memphis': 'MEM',
    'Miami': 'MIA',
    'Milwaukee': 'MIL',
    'Minnesota': 'MIN',
    'New Jersey': 'NJN',  # Now Brooklyn
    'New Orleans': 'NOP',
    'New York': 'NYK',
    'Oklahoma City': 'OKC',
    'Orlando': 'ORL',
    'Philadelphia': 'PHI',
    'Phoenix': 'PHX',
    'Portland': 'POR',
    'Sacramento': 'SAC',
    'San Antonio': 'SAS',
    'Seattle': 'SEA',  # Now OKC (moved 2008)
    'Toronto': 'TOR',
    'Utah': 'UTA',
    'Washington': 'WAS',
    # Handle potential variations
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'New Orleans Hornets': 'NOP',
    'New Orleans Pelicans': 'NOP',
    'Charlotte Bobcats': 'CHA',
    'Charlotte Hornets': 'CHA',
}


def convert_season_format(kaggle_season: int) -> str:
    """
    Convert Kaggle season format (e.g., 2023) to project format (e.g., 2022-23).
    
    Kaggle uses the ending year, so 2023 means the 2022-23 season.
    """
    start_year = kaggle_season - 1
    end_year_short = str(kaggle_season)[-2:]
    return f"{start_year}-{end_year_short}"


def moneyline_to_implied_probability(ml: float) -> float:
    """
    Convert American moneyline odds to implied probability.
    
    For negative odds (favorites): probability = |odds| / (|odds| + 100)
    For positive odds (underdogs): probability = 100 / (odds + 100)
    """
    if pd.isna(ml) or ml == 0:
        return np.nan
    
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def process_kaggle_odds():
    """
    Process the Kaggle odds data and save in project format.
    """
    print("\n" + "="*70)
    print("  PROCESSING KAGGLE NBA ODDS DATA")
    print("="*70 + "\n")
    
    # Check if input file exists
    if not os.path.exists(KAGGLE_ODDS_PATH):
        print(f"❌ Kaggle odds file not found: {KAGGLE_ODDS_PATH}")
        print(f"   Please download from Kaggle and save to {KAGGLE_ODDS_PATH}")
        return None
    
    # Load data
    print(f"📂 Loading {KAGGLE_ODDS_PATH}...")
    df = pd.read_csv(KAGGLE_ODDS_PATH)
    print(f"   Loaded {len(df)} rows")
    
    # Convert date
    df['Date'] = pd.to_datetime(df['date'])
    
    # Convert season format
    df['Season'] = df['season'].apply(convert_season_format)
    
    # Map team names to abbreviations
    print("🔄 Mapping team names to abbreviations...")
    
    # Check for any unmapped teams
    all_teams = set(df['team'].unique()) | set(df['opponent'].unique())
    unmapped = [t for t in all_teams if t not in TEAM_NAME_TO_ABBR]
    if unmapped:
        print(f"   ⚠️  Unmapped teams found: {unmapped}")
        # Try to continue anyway
    
    df['Team'] = df['team'].map(TEAM_NAME_TO_ABBR)
    df['Opponent'] = df['opponent'].map(TEAM_NAME_TO_ABBR)
    
    # Determine home/away
    # In Kaggle data: "vs" = home, "@" = away
    df['Is_Home'] = df['home/visitor'].apply(lambda x: 1 if x == 'vs' else 0)
    
    # Keep only home team rows to avoid duplicates (each game appears twice in the data)
    df_home = df[df['Is_Home'] == 1].copy()
    print(f"   Filtered to {len(df_home)} home games")
    
    # Calculate implied probabilities
    print("📊 Calculating implied probabilities...")
    df_home['Home_ML'] = df_home['moneyLine']
    df_home['Away_ML'] = df_home['opponentMoneyLine']
    df_home['Home_Implied_Prob'] = df_home['Home_ML'].apply(moneyline_to_implied_probability)
    df_home['Away_Implied_Prob'] = df_home['Away_ML'].apply(moneyline_to_implied_probability)
    
    # Normalize implied probabilities (remove vig)
    total_prob = df_home['Home_Implied_Prob'] + df_home['Away_Implied_Prob']
    df_home['Home_Win_Prob'] = df_home['Home_Implied_Prob'] / total_prob
    df_home['Away_Win_Prob'] = df_home['Away_Implied_Prob'] / total_prob
    
    # Spread and totals
    df_home['Spread'] = df_home['spread']
    df_home['Total'] = df_home['total']
    
    # Actual scores and results
    df_home['Home_Score'] = df_home['score']
    df_home['Away_Score'] = df_home['opponentScore']
    df_home['Home_Win'] = (df_home['Home_Score'] > df_home['Away_Score']).astype(int)
    
    # Select and rename columns for output
    output_cols = [
        'Season', 'Date', 'Team', 'Opponent',
        'Home_ML', 'Away_ML', 
        'Home_Win_Prob', 'Away_Win_Prob',
        'Spread', 'Total',
        'Home_Score', 'Away_Score', 'Home_Win'
    ]
    
    df_output = df_home[output_cols].copy()
    
    # Rename Team/Opponent to Home_Team/Away_Team for clarity
    df_output = df_output.rename(columns={
        'Team': 'Home_Team',
        'Opponent': 'Away_Team'
    })
    
    # Sort by date
    df_output = df_output.sort_values(['Season', 'Date']).reset_index(drop=True)
    
    # Round probabilities
    df_output['Home_Win_Prob'] = df_output['Home_Win_Prob'].round(4)
    df_output['Away_Win_Prob'] = df_output['Away_Win_Prob'].round(4)
    
    # Drop rows with missing team mappings
    before_drop = len(df_output)
    df_output = df_output.dropna(subset=['Home_Team', 'Away_Team'])
    after_drop = len(df_output)
    if before_drop > after_drop:
        print(f"   Dropped {before_drop - after_drop} rows with unmapped teams")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save
    df_output.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved processed odds to {OUTPUT_PATH}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"Total games: {len(df_output)}")
    print(f"\nSeasons covered:")
    season_counts = df_output.groupby('Season').size()
    for season, count in season_counts.items():
        print(f"  {season}: {count} games")
    
    print(f"\nDate range: {df_output['Date'].min().date()} to {df_output['Date'].max().date()}")
    
    # Check overlap with our training data (2015-16 onwards)
    relevant_seasons = df_output[df_output['Season'] >= '2015-16']
    print(f"\nSeasons >= 2015-16 (overlaps with training data):")
    for season in sorted(relevant_seasons['Season'].unique()):
        count = len(relevant_seasons[relevant_seasons['Season'] == season])
        print(f"  {season}: {count} games")
    
    print("="*70 + "\n")
    
    return df_output


if __name__ == "__main__":
    process_kaggle_odds()
