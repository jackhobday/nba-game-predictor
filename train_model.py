#!/usr/bin/env python3
"""
NBA Game Outcome Prediction Model

Trains XGBoost model using rolling season splits for time-series cross-validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

# Configuration
DATA_FILE = 'data_merged/games_all_seasons.csv'
OUTPUT_DIR = 'model_results'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Define columns to exclude
leak_cols = ['Result', 'Team_Score', 'Opp_Score', 'OT']
id_cols = ['Season', 'Date', 'Team', 'Opponent']  # identifiers, not predictive

# Prepare features and target
X = df.drop(columns=[c for c in (leak_cols + id_cols) if c in df.columns]).copy()
y = df['Result'].astype(int)

# Get unique seasons
seasons = sorted(df['Season'].unique())
print(f"\nSeasons in dataset: {seasons}")
print(f"Total games: {len(df)}")
print(f"Features: {len(X.columns)}")

# Rolling season splits setup
# Start with 2015-2020 (5 seasons: 2015-16 through 2019-20), test on 2021+
# Find the index where we have accumulated 2015-2020 data (5 seasons)
min_train_seasons = 5  # Need at least 5 seasons for initial training (2015-2020)
start_test_idx = min_train_seasons  # Start testing from index 5 (2020-21 season)

results = []

for i in range(start_test_idx, len(seasons)):
    train_seasons = seasons[:i]
    test_season = seasons[i]

    print(f"\n{'='*70}")
    print(f"Train on {train_seasons} → Test on {test_season}")
    print(f"{'='*70}")

    # Split data by season
    train_idx = df["Season"].isin(train_seasons)
    test_idx = df["Season"] == test_season

    X_train = X.loc[train_idx].copy()
    y_train = y.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_test = y.loc[test_idx].copy()
    
    # Calculate sample weights: exponential decay for recency weighting
    # Most recent season gets weight 1.0, older seasons get progressively less weight
    # Decay factor: 0.10 per season (most recent = 1.0, one before = 0.90, two before = 0.80, etc.)
    # Lower values = less aggressive decay (older seasons keep more weight)
    DECAY_FACTOR = 0.10  # Adjust this to change how quickly weights decrease (0.0 = equal weights, higher = more decay)
    
    train_df_seasons = df.loc[train_idx, 'Season'].copy()
    season_weights = {}
    for j, season in enumerate(train_seasons):
        # Most recent season (last in list) gets weight 1.0
        # Older seasons get weight = 1.0 - (distance_from_most_recent * DECAY_FACTOR)
        distance_from_recent = len(train_seasons) - 1 - j
        weight = max(0.1, 1.0 - (distance_from_recent * DECAY_FACTOR))  # Minimum weight of 0.1
        season_weights[season] = weight
    
    sample_weights = train_df_seasons.map(season_weights).values
    
    print(f"  Season weights: {dict(sorted(season_weights.items(), key=lambda x: train_seasons.index(x[0])))}")

    # Align columns (handle missing columns in test set)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Early stopping setup
    val_season = train_seasons[-1]
    subtrain_seasons = train_seasons[:-1]

    # XGBoost parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 35,
    }

    if len(subtrain_seasons) == 0:
        # For first fold, no validation season available
        print("  No validation season available, using fixed n_estimators=50")
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=50
        )
        best_n_estimators = 50
    else:
        # Use last training season as validation
        sub_idx = df["Season"].isin(subtrain_seasons)
        val_idx = df["Season"] == val_season

        X_sub = X.loc[sub_idx].copy()
        y_sub = y.loc[sub_idx].copy()
        X_val = X.loc[val_idx].copy()
        y_val = y.loc[val_idx].copy()
        
        # Calculate weights for subtrain set
        subtrain_df_seasons = df.loc[sub_idx, 'Season'].copy()
        subtrain_season_weights = {}
        for j, season in enumerate(subtrain_seasons):
            distance_from_recent = len(subtrain_seasons) - 1 - j
            weight = max(0.1, 1.0 - (distance_from_recent * DECAY_FACTOR))
            subtrain_season_weights[season] = weight
        subtrain_sample_weights = subtrain_df_seasons.map(subtrain_season_weights).values

        X_sub, X_val = X_sub.align(X_val, join="left", axis=1, fill_value=0)

        dsub = xgb.DMatrix(X_sub, label=y_sub, weight=subtrain_sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        print(f"  Using {val_season} as validation set for early stopping...")
        booster = xgb.train(
            params=params,
            dtrain=dsub,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        best_n_estimators = booster.best_iteration + 1
        print(f"  Early stopping at {best_n_estimators} rounds")

        # Retrain on all training seasons with best_n_estimators (using full weights)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=best_n_estimators
        )

    # Evaluate on test set
    dtest = xgb.DMatrix(X_test)
    proba = booster.predict(dtest)
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print(f"  Best n_estimators: {best_n_estimators}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  ROC-AUC:  {auc:.3f}")

    results.append({
        "Train_Seasons": str(train_seasons),
        "Test_Season": test_season,
        "Best_n_estimators": best_n_estimators,
        "Accuracy": float(acc),
        "ROC_AUC": float(auc),
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Print summary
print("\n" + "="*70)
print("  CROSS-VALIDATION RESULTS")
print("="*70)
print(results_df.to_string(index=False))
print("\n" + "-"*70)
print(f"Average Accuracy: {results_df['Accuracy'].mean():.3f}")
print(f"Average ROC-AUC:  {results_df['ROC_AUC'].mean():.3f}")
print("="*70)

# Save results
results_path = os.path.join(OUTPUT_DIR, 'cv_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# Feature importance (from final model)
print("\n" + "="*70)
print("  FEATURE IMPORTANCE (Final Model)")
print("="*70)

score = booster.get_score(importance_type="gain")
fi = pd.Series(score).sort_values(ascending=False)

# Map feature indices to names if needed
if len(fi) > 0 and str(fi.index[0]).startswith("f"):
    feature_map = {f"f{i}": col for i, col in enumerate(X_train.columns)}
    fi.index = fi.index.map(feature_map)

# Display top features
print(f"\nTop 20 Features by Gain:")
print(fi.head(20).to_string())

# Save feature importance
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
fi.to_csv(fi_path, header=['Gain'])
print(f"\nFeature importance saved to: {fi_path}")

# Save the final trained model for predictions
import pickle

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Train a final model on all historical data (excluding current season if it exists)
# Use the last training configuration from cross-validation
print("\n" + "="*70)
print("  TRAINING FINAL MODEL FOR PREDICTIONS")
print("="*70)

# Train on all data up to the last complete season (exclude current/incomplete season)
# The last test season was seasons[-1], so train on all previous seasons
final_train_seasons = seasons[:-1] if len(seasons) > 1 else seasons
final_train_idx = df['Season'].isin(final_train_seasons)

X_final_train = X.loc[final_train_idx].copy()
y_final_train = y.loc[final_train_idx].copy()

print(f"Training final model on seasons: {final_train_seasons}")
print(f"Total training games: {len(X_final_train)}")

# Calculate recency weights for final model
DECAY_FACTOR = 0.10
final_train_df_seasons = df.loc[final_train_idx, 'Season'].copy()
final_season_weights = {}
for j, season in enumerate(final_train_seasons):
    distance_from_recent = len(final_train_seasons) - 1 - j
    weight = max(0.1, 1.0 - (distance_from_recent * DECAY_FACTOR))
    final_season_weights[season] = weight

final_sample_weights = final_train_df_seasons.map(final_season_weights).values

# Use same params as cross-validation
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 35,
}

# Train final model
dtrain_final = xgb.DMatrix(X_final_train, label=y_final_train, weight=final_sample_weights)
final_booster = xgb.train(params=params, dtrain=dtrain_final, num_boost_round=150)

# Save model
model_path = os.path.join(model_dir, 'xgb_classifier_final.model')
final_booster.save_model(model_path)

# Save metadata
model_metadata = {
    'feature_names': list(X_final_train.columns),
    'train_seasons': final_train_seasons,
    'decay_factor': DECAY_FACTOR,
    'model_type': 'classification'
}
metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(model_metadata, f)

print(f"✅ Final model saved to: {model_path}")
print(f"✅ Metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("  TRAINING COMPLETE")
print("="*70)

