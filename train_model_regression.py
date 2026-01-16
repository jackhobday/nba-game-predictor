#!/usr/bin/env python3
"""
NBA Game Outcome Prediction Model - Regression Approach

Trains XGBoost regression model to predict point differential,
then converts predictions to win/loss outcomes.
Uses rolling season splits for time-series cross-validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_absolute_error, 
    mean_squared_error, r2_score
)
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

# Prepare features
X = df.drop(columns=[c for c in (leak_cols + id_cols) if c in df.columns]).copy()

# Create point differential target (regression) and binary result (for classification metrics)
# Point differential: Team_Score - Opp_Score (positive = Team wins, negative = Team loses)
y_regression = (df['Team_Score'] - df['Opp_Score']).astype(float)
y_binary = df['Result'].astype(int)  # Keep for comparison metrics

# Get unique seasons
seasons = sorted(df['Season'].unique())
print(f"\nSeasons in dataset: {seasons}")
print(f"Total games: {len(df)}")
print(f"Features: {len(X.columns)}")
print(f"Point differential range: [{y_regression.min():.1f}, {y_regression.max():.1f}]")

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
    y_train_reg = y_regression.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_test_reg = y_regression.loc[test_idx].copy()
    y_test_binary = y_binary.loc[test_idx].copy()
    
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

    # XGBoost regression parameters
    params = {
        "objective": "reg:squarederror",  # Regression objective
        "eval_metric": "rmse",            # Root mean squared error
        "eta": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 35,
    }

    if len(subtrain_seasons) == 0:
        # For first fold, no validation season available
        print("  No validation season available, using fixed n_estimators=50")
        dtrain = xgb.DMatrix(X_train, label=y_train_reg, weight=sample_weights)
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
        y_sub_reg = y_regression.loc[sub_idx].copy()
        X_val = X.loc[val_idx].copy()
        y_val_reg = y_regression.loc[val_idx].copy()
        
        # Calculate weights for subtrain set
        subtrain_df_seasons = df.loc[sub_idx, 'Season'].copy()
        subtrain_season_weights = {}
        for j, season in enumerate(subtrain_seasons):
            distance_from_recent = len(subtrain_seasons) - 1 - j
            weight = max(0.1, 1.0 - (distance_from_recent * DECAY_FACTOR))
            subtrain_season_weights[season] = weight
        subtrain_sample_weights = subtrain_df_seasons.map(subtrain_season_weights).values

        X_sub, X_val = X_sub.align(X_val, join="left", axis=1, fill_value=0)

        dsub = xgb.DMatrix(X_sub, label=y_sub_reg, weight=subtrain_sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val_reg)

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
        dtrain = xgb.DMatrix(X_train, label=y_train_reg, weight=sample_weights)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=best_n_estimators
        )

    # Evaluate on test set - Regression predictions
    dtest = xgb.DMatrix(X_test)
    point_diff_pred = booster.predict(dtest)
    
    # Convert point differential to win/loss predictions
    win_pred = (point_diff_pred > 0).astype(int)
    
    # Calculate actual point differentials for test set
    test_team_scores = df.loc[test_idx, 'Team_Score'].values
    test_opp_scores = df.loc[test_idx, 'Opp_Score'].values
    actual_point_diff = test_team_scores - test_opp_scores

    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_test_reg, point_diff_pred))
    mae = mean_absolute_error(y_test_reg, point_diff_pred)
    r2 = r2_score(y_test_reg, point_diff_pred)
    
    # Classification metrics (converting point diff to win/loss)
    acc = accuracy_score(y_test_binary, win_pred)
    
    # For ROC-AUC, use point differential as probability-like score
    # Normalize point differential predictions to [0, 1] range for AUC calculation
    # Using sigmoid transformation: 1 / (1 + exp(-x/scale))
    # Scale factor of ~10 works well for NBA point differentials (typically -30 to +30)
    scale = 10.0
    proba_from_diff = 1 / (1 + np.exp(-point_diff_pred / scale))
    auc = roc_auc_score(y_test_binary, proba_from_diff)

    print(f"  Best n_estimators: {best_n_estimators}")
    print(f"  Regression Metrics:")
    print(f"    RMSE: {rmse:.3f} points")
    print(f"    MAE:  {mae:.3f} points")
    print(f"    R²:   {r2:.3f}")
    print(f"  Classification Metrics (from point diff > 0):")
    print(f"    Accuracy: {acc:.3f}")
    print(f"    ROC-AUC:  {auc:.3f}")

    results.append({
        "Train_Seasons": str(train_seasons),
        "Test_Season": test_season,
        "Best_n_estimators": best_n_estimators,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Accuracy": float(acc),
        "ROC_AUC": float(auc),
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Print summary
print("\n" + "="*70)
print("  CROSS-VALIDATION RESULTS (REGRESSION MODEL)")
print("="*70)
print(results_df.to_string(index=False))
print("\n" + "-"*70)
print("Regression Metrics:")
print(f"  Average RMSE: {results_df['RMSE'].mean():.3f} points")
print(f"  Average MAE:  {results_df['MAE'].mean():.3f} points")
print(f"  Average R²:   {results_df['R2'].mean():.3f}")
print("\nClassification Metrics (from point diff > 0):")
print(f"  Average Accuracy: {results_df['Accuracy'].mean():.3f}")
print(f"  Average ROC-AUC:  {results_df['ROC_AUC'].mean():.3f}")
print("="*70)

# Save results
results_path = os.path.join(OUTPUT_DIR, 'cv_results_regression.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# Feature importance (from final model)
print("\n" + "="*70)
print("  FEATURE IMPORTANCE (Final Regression Model)")
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
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_regression.csv')
fi.to_csv(fi_path, header=['Gain'])
print(f"\nFeature importance saved to: {fi_path}")

# Save the final trained model for predictions
import pickle

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Train a final model on all historical data (excluding current season if it exists)
print("\n" + "="*70)
print("  TRAINING FINAL REGRESSION MODEL FOR PREDICTIONS")
print("="*70)

# Train on all data up to the last complete season
final_train_seasons = seasons[:-1] if len(seasons) > 1 else seasons
final_train_idx = df['Season'].isin(final_train_seasons)

X_final_train = X.loc[final_train_idx].copy()
y_final_train_reg = y_regression.loc[final_train_idx].copy()

print(f"Training final regression model on seasons: {final_train_seasons}")
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
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 35,
}

# Train final model
dtrain_final = xgb.DMatrix(X_final_train, label=y_final_train_reg, weight=final_sample_weights)
final_booster = xgb.train(params=params, dtrain=dtrain_final, num_boost_round=150)

# Save model
model_path = os.path.join(model_dir, 'xgb_regression_final.model')
final_booster.save_model(model_path)

# Save metadata
model_metadata = {
    'feature_names': list(X_final_train.columns),
    'train_seasons': final_train_seasons,
    'decay_factor': DECAY_FACTOR,
    'model_type': 'regression'
}
metadata_path = os.path.join(model_dir, 'model_metadata_regression.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(model_metadata, f)

print(f"✅ Final regression model saved to: {model_path}")
print(f"✅ Metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("  REGRESSION TRAINING COMPLETE")
print("="*70)
print("\nNote: This model predicts point differential, then converts to win/loss.")
print("      Compare with classification model to see which approach works better!")

