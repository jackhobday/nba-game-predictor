#!/usr/bin/env python3
"""
Clean CSV files by removing duplicate header rows.

Each CSV should have:
- 1 header row at the top
- 82 game rows (for most seasons)
- 2020-21 season: 72 game rows (73 total with header)
"""

import os
import pandas as pd
from pathlib import Path

# Data directory
DATA_DIR = 'data'

# Expected row counts (including header)
EXPECTED_ROWS = {
    '2024-25': 83,  # 1 header + 82 games
    '2023-24': 83,  # 1 header + 82 games
    '2022-23': 83,  # 1 header + 82 games
    '2021-22': 83,  # 1 header + 82 games
    '2020-21': 73,  # 1 header + 72 games (shortened season)
}

# Header row pattern (first few columns that identify a header row)
HEADER_PATTERN = ['Rk', 'Gtm', 'Date', '', 'Opp', 'Rslt', 'Tm', 'Opp', 'OT']


def is_header_row(row):
    """
    Check if a row is a header row by comparing first few values.
    
    Args:
        row: Pandas Series or list representing a row
        
    Returns:
        True if this looks like a header row, False otherwise
    """
    if len(row) < len(HEADER_PATTERN):
        return False
    
    # Check first few columns
    for i, expected in enumerate(HEADER_PATTERN):
        if i >= len(row):
            return False
        row_val = str(row.iloc[i] if hasattr(row, 'iloc') else row[i]).strip()
        expected_str = str(expected).strip()
        
        # Handle empty string comparison
        if expected_str == '':
            # For empty column, check if row value is also empty or NaN
            if row_val and row_val.lower() != 'nan' and row_val != '':
                return False
        else:
            # For non-empty, must match exactly
            if row_val != expected_str:
                return False
    
    return True


def clean_csv_file(filepath, season):
    """
    Clean a single CSV file by removing duplicate header rows.
    
    Args:
        filepath: Path to the CSV file
        season: Season string (e.g., '2024-25')
        
    Returns:
        Tuple of (success: bool, rows_removed: int, final_row_count: int)
    """
    try:
        # Read the CSV file without headers to see raw structure
        df = pd.read_csv(filepath, header=None)
        
        if df.empty:
            return False, 0, 0
        
        # Find all header rows (simple format: Rk,Gtm,Date,,Opp,Rslt,Tm,Opp,OT,...)
        header_rows = []
        for idx in df.index:
            if is_header_row(df.iloc[idx]):
                header_rows.append(idx)
        
        if not header_rows:
            # Try reading with header to see if it's already properly formatted
            try:
                df_test = pd.read_csv(filepath)
                # Check if first row looks like data (numeric Rk column)
                if len(df_test) > 0:
                    first_col = df_test.columns[0]
                    if first_col == 'Rk' and pd.api.types.is_numeric_dtype(df_test.iloc[:, 0]):
                        # File is already clean
                        final_count = len(df_test) + 1  # +1 for header
                        return True, 0, final_count
            except:
                pass
            
            print(f"  ⚠️  No header row found in {os.path.basename(filepath)}")
            return False, 0, len(df)
        
        # Use the first simple header row as column names
        first_header_idx = header_rows[0]
        header_values = [str(val).strip() for val in df.iloc[first_header_idx].tolist()]
        
        # Remove all header rows (including multi-level ones at the top)
        rows_to_remove = header_rows
        
        # Create cleaned dataframe with only data rows
        df_cleaned = df.drop(rows_to_remove).reset_index(drop=True)
        
        # Set column names from the first simple header row
        df_cleaned.columns = header_values
        
        # Remove any rows that are completely empty or have all NaN values
        df_cleaned = df_cleaned.dropna(how='all').reset_index(drop=True)
        
        # Remove rows where Rk column is not numeric (likely leftover header or footer)
        if 'Rk' in df_cleaned.columns:
            df_cleaned = df_cleaned[pd.to_numeric(df_cleaned['Rk'], errors='coerce').notna()].reset_index(drop=True)
        
        # Save cleaned file
        df_cleaned.to_csv(filepath, index=False)
        
        final_count = len(df_cleaned) + 1  # +1 for header row
        
        return True, len(rows_to_remove) - 1, final_count  # -1 because we keep one header
        
    except Exception as e:
        print(f"  ✗ Error cleaning {os.path.basename(filepath)}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0


def clean_all_csvs():
    """
    Clean all CSV files in the data directory.
    """
    print("\n" + "="*70)
    print("  CLEANING CSV FILES")
    print("="*70 + "\n")
    
    total_files = 0
    successful = 0
    failed = 0
    total_rows_removed = 0
    
    # Process each season
    for season in sorted(EXPECTED_ROWS.keys()):
        season_dir = os.path.join(DATA_DIR, season)
        expected_count = EXPECTED_ROWS[season]
        
        if not os.path.exists(season_dir):
            print(f"⚠️  Season directory not found: {season_dir}")
            continue
        
        print(f"📅 Season: {season} (expected: {expected_count} rows per file)")
        print("-" * 70)
        
        # Get all CSV files in this season
        csv_files = sorted([f for f in os.listdir(season_dir) if f.endswith('.csv')])
        
        for csv_file in csv_files:
            filepath = os.path.join(season_dir, csv_file)
            total_files += 1
            
            # Clean the file
            success, rows_removed, final_count = clean_csv_file(filepath, season)
            
            if success:
                successful += 1
                total_rows_removed += rows_removed
                
                # Check if row count matches expected
                status = "✓"
                if final_count != expected_count:
                    status = f"⚠️  ({final_count} rows, expected {expected_count})"
                
                if rows_removed > 0:
                    print(f"  {status} {csv_file}: Removed {rows_removed} duplicate header(s), {final_count} rows total")
                else:
                    print(f"  {status} {csv_file}: Already clean, {final_count} rows total")
            else:
                failed += 1
                print(f"  ✗ {csv_file}: Failed to clean")
        
        print()
    
    # Summary
    print("="*70)
    print("  CLEANING COMPLETE")
    print("="*70)
    print(f"Total files:  {total_files}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed:     {failed}")
    print(f"Rows removed: {total_rows_removed}")
    print("="*70 + "\n")


if __name__ == "__main__":
    clean_all_csvs()

