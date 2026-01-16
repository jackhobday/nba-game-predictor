# NBA Game Outcome Prediction - ML Project

This project predicts NBA game outcomes using advanced statistics from Basketball-Reference.com. The final dataset includes rolling averages, win streaks, home/away splits, and other engineered features for machine learning.

## Project Goal

Build a machine learning model to predict the winner of NBA games based on:
- Team advanced statistics (season-to-date and recent form)
- Win/loss streaks and momentum
- Home court advantage
- Rest days and back-to-back games
- Matchup differentials between teams

## Complete Pipeline

### Phase 1: Initial Setup (One-Time)

#### 1. Install Dependencies

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

#### 2. Scrape Historical Game Logs

Scrape advanced game logs from Basketball-Reference.com for all 30 NBA teams across multiple seasons:

```bash
python scrape_gamelogs_selenium.py --headless
```

**Options:**
- `--headless`: Run Chrome in background (recommended)
- `--season 2025-26`: Scrape specific season only
- Without flag: Watch the browser work (useful for debugging)

**Estimated time:** ~7-8 minutes per season (30 teams × ~82 games)

#### 3. Clean CSV Files

Remove duplicate header rows from scraped data:

```bash
python clean_csv_files.py
```

#### 4. Engineer Features

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

**Output:** `data_enhanced/[season]/[TEAM]_[season]_enhanced.csv`

#### 5. Merge into Final Dataset

Combine team game logs into game-level rows with both teams' features:

```bash
python merge_games.py
```

**Output:** `data_merged/games_all_seasons.csv`
- One row per game
- ~70+ features (Team, Opponent, and differential features)
- Target variable: `Result` (1 = Team wins, 0 = Opponent wins)

#### 6. Scrape NBA Schedule

Scrape the complete NBA schedule for the current season:

```bash
python scrape_schedule.py --season 2025-26 --headless
```

**Output:** `nba_schedule_2025-26.csv` with columns: `Game_Number`, `Date`, `Home_Team`, `Away_Team`

#### 7. Train Models

Train models using rolling season validation with recency weighting:

```bash
# XGBoost Classification (direct win/loss prediction)
python train_model.py

# XGBoost Regression (predicts point differential, then converts to win/loss)
python train_model_regression.py
```

**Training Approach:**
- Rolling validation: Train on historical seasons, test on next season
- Recency weighting: More recent seasons get higher weight (decay factor: 0.10)
- Early stopping: Uses last training season as validation set

**Outputs:**
- `models/xgb_classifier_final.model` - Classification model
- `models/xgb_regression_final.model` - Regression model
- `models/model_metadata.pkl` - Feature names and training info
- `model_results/cv_results.csv` - Cross-validation results

---

### Phase 2: Daily Predictions Workflow

Each day, run these steps to predict today's games:

#### Option A: Step-by-Step

```bash
# 1. Scrape latest games from current season (if games were played yesterday)
python scrape_gamelogs_selenium.py --season 2025-26 --headless

# 2. Clean new data
python clean_csv_files.py

# 3. Re-run feature engineering for current season (updates stats)
python feature_engineering.py

# 4. Make predictions for today
python predict_games.py
```

#### Option B: Quick Prediction (if data is already up-to-date)

```bash
python predict_games.py                    # Predict today's games
python predict_games.py 2026-01-17         # Predict specific date
python predict_games.py --model regression # Use regression model
```

**Prediction Output:**
- Console display with win probabilities and confidence levels
- CSV file: `predictions/predictions_YYYYMMDD.csv`

---

## Model Comparison

The project includes three models for comparison:

### 1. XGBoost Classification
- **Approach:** Direct binary classification (win/loss)
- **Output:** Win probability (0 to 1)
- **Best for:** Simple win/loss predictions

### 2. XGBoost Regression
- **Approach:** Predicts point differential, converts to win probability
- **Output:** Point differential + win probability
- **Best for:** When you want point spread predictions

### 3. Logistic Regression
- **Approach:** Simple baseline model
- **Output:** Win probability
- **Best for:** Baseline comparison, simpler alternative

## Features

The model uses ~70 engineered features including:

**Team-Level Features:**
- Season-to-date averages: ORtg, DRtg, NetRtg, Pace, Win%, Four Factors
- Recent form: Last 5 and last 10 game averages
- Win/loss streaks: Current and longest streaks
- Home/away splits: Win% and ratings by location
- Rest indicators: Days rest, back-to-back flags

**Game-Level Features:**
- Home/away status
- Days rest for both teams
- Game number in season

**Differential Features:**
- Net rating differential
- Offensive/defensive rating differentials
- Win percentage differential
- Recent form differentials
- Four factors differentials
- Pace differential
- Rest advantage
- Streak differential

## Confidence Analysis

The prediction system includes confidence scores:
- **Formula:** `|Probability - 0.5| × 2`
- **Range:** 0% (50/50) to 100% (certain)
- **Interpretation:** Higher confidence = further from even odds

When the model is highly confident (≥80%), predictions tend to be more reliable.

## Data Source

Data is scraped from [Basketball-Reference.com](https://www.basketball-reference.com/).

> "Data provided by Basketball-Reference.com"

## Project Structure

```
.
├── data/                          # Raw game logs (generated, not committed)
├── data_enhanced/                # Enhanced logs with features (generated)
├── data_merged/                  # Final ML dataset (generated)
├── models/                       # Trained models (generated)
│   ├── xgb_classifier_final.model
│   ├── xgb_regression_final.model
│   └── model_metadata.pkl
├── model_results/                # Training results and metrics
├── predictions/                  # Daily prediction outputs
├── scrape_gamelogs_selenium.py  # Game log scraper
├── scrape_schedule.py           # Schedule scraper
├── clean_csv_files.py           # Data cleaning script
├── feature_engineering.py       # Feature calculation script
├── merge_games.py               # Dataset merging script
├── train_model.py               # XGBoost classification training
├── train_model_regression.py    # XGBoost regression training
├── predict_games.py             # Prediction script
├── nba_schedule_2025-26.csv     # Current season schedule
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Reference

### Initial Setup (Run Once)
```bash
python scrape_gamelogs_selenium.py --headless
python clean_csv_files.py
python feature_engineering.py
python merge_games.py
python scrape_schedule.py --season 2025-26 --headless
python train_model.py
python train_model_regression.py
python train_model_logistic.py
```

### Daily Predictions
```bash
# Full workflow
python scrape_gamelogs_selenium.py --season 2025-26 --headless
python clean_csv_files.py
python feature_engineering.py
python predict_games.py

# Quick prediction (if data is already up-to-date)
python predict_games.py
```

### Making Predictions
```bash
python predict_games.py                    # Today's games (classification)
python predict_games.py 2026-01-17         # Specific date
python predict_games.py --model regression # Regression model
python predict_games.py --season 2025-26   # Specify season
```

## Results

Models are evaluated using rolling season validation:
- Each model is trained on historical seasons and tested on the next season
- Results are aggregated across all test seasons
- Average accuracy: ~60-65% (varies by model and season)
- Higher accuracy achieved on games where model is highly confident

## Future Enhancements

- Confidence-based accuracy analysis script
- Automated daily workflow script
- Model calibration analysis
- Feature importance visualization
- Prediction accuracy tracking over time
