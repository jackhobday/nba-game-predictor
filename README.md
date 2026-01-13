# NBA Game Outcome Prediction - ML Project

This project predicts NBA game outcomes using advanced statistics from Basketball-Reference.com. The final dataset includes rolling averages, win streaks, home/away splits, and other engineered features for machine learning.

## Project Goal

Build a machine learning model to predict the winner of NBA games based on:
- Team advanced statistics (season-to-date and recent form)
- Win/loss streaks and momentum
- Home court advantage
- Rest days and back-to-back games
- Matchup differentials between teams

## Getting Started

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Requirements:**
- Chrome browser (for Selenium scraper)
- Python 3.7+

### 2. Scrape Game Logs

Scrape advanced game logs from Basketball-Reference.com for all 30 NBA teams across 5 seasons:

```bash
python scrape_gamelogs_selenium.py --headless
```

**Options:**
- `--headless`: Run Chrome in background (recommended)
- Without flag: Watch the browser work (useful for debugging)

**Estimated time:** ~7-8 minutes for all 150 files (30 teams × 5 seasons)

### 3. Clean CSV Files

Remove duplicate header rows from scraped data:

```bash
python clean_csv_files.py
```

### 4. Engineer Features

Calculate rolling statistics, streaks, and other features for each team:

```bash
python feature_engineering.py
```

This creates enhanced game logs with:
- Season-to-date averages (ORtg, DRtg, NetRtg, Four Factors, etc.)
- Recent form (last 5 and last 10 games)
- Win/loss streaks
- Home/away splits
- Days rest and back-to-back indicators

### 5. Merge into Final Dataset

Combine team game logs into game-level rows with both teams' features:

```bash
python merge_games.py
```

**Output:** `data_merged/games_all_seasons.csv`
- 6,000 games (one row per game)
- 76 features (Team, Opponent, and differential features)
- Target variable: `Result` (1 = Team wins, 0 = Opponent wins)

## Final Dataset

The final dataset (`data_merged/games_all_seasons.csv`) is ready for machine learning with:
- **6,000 game rows** across 5 seasons
- **76 features** including:
  - Team and opponent season averages
  - Recent form metrics (last 5/10 games)
  - Win/loss streaks
  - Home/away splits
  - Rest days and back-to-back indicators
  - Matchup differentials
- **Target variable:** `Result` (binary classification)

## Data Source

Data is scraped from [Basketball-Reference.com](https://www.basketball-reference.com/).

> "Data provided by Basketball-Reference.com"

## Project Structure

```
.
├── data/                    # Raw game logs (generated, not committed)
├── data_enhanced/          # Enhanced logs with features (generated)
├── data_merged/            # Final ML dataset (generated)
├── scrape_gamelogs_selenium.py  # Web scraper
├── clean_csv_files.py      # Data cleaning script
├── feature_engineering.py  # Feature calculation script
├── merge_games.py          # Dataset merging script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```
