#!/usr/bin/env python3
"""
Basketball-Reference Advanced Game Logs Scraper - Selenium Version

Uses Selenium WebDriver to scrape advanced game logs from Basketball-Reference.com
This version uses a real browser, making it much harder to block.
"""

import os
import time
import pandas as pd
from typing import Optional
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# Team abbreviations for Basketball-Reference URLs (some differ from standard abbreviations)
TEAMS = [
    'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Mapping from Basketball-Reference abbreviations to file naming abbreviations
# (Basketball-Reference uses BRK, CHO, PHO but we want BKN, CHA, PHX in filenames)
TEAM_NAME_MAPPING = {
    'BRK': 'BKN',  # Brooklyn Nets
    'CHO': 'CHA',  # Charlotte Hornets
    'PHO': 'PHX',  # Phoenix Suns
}

SEASONS = {
    '2024-25': '2025',
    '2023-24': '2024',
    '2022-23': '2023',
    '2021-22': '2022',
    '2020-21': '2021'
}

BASE_URL = "https://www.basketball-reference.com/teams/{team}/{year}/gamelog-advanced/"

# Rate limiting configuration
PAGE_LOAD_WAIT = 4  # seconds to wait for page to fully load (increased for stability)
REQUEST_DELAY = 3   # seconds between requests (increased to be more respectful)


def setup_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Set up and configure Chrome WebDriver.
    
    Args:
        headless: If True, run browser in headless mode (no visible window)
        
    Returns:
        Configured Chrome WebDriver instance
    """
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")
    
    # Add options to make Selenium less detectable
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Initialize driver
    import os
    import glob
    import stat
    
    # Get the base path from ChromeDriverManager
    driver_path = ChromeDriverManager().install()
    
    # Fix: webdriver-manager sometimes returns wrong file path
    # Navigate to the correct chromedriver executable
    if os.path.isfile(driver_path):
        driver_dir = os.path.dirname(driver_path)
        
        # Look for actual chromedriver executable in the directory
        possible_paths = [
            os.path.join(driver_dir, 'chromedriver'),
            os.path.join(driver_dir, 'chromedriver.exe'),
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                driver_path = path
                logger.debug(f"Using chromedriver at: {driver_path}")
                break
        else:
            # If not found directly, search recursively
            pattern = os.path.join(os.path.dirname(driver_dir), '**', 'chromedriver')
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Filter out non-executable files
                for match in matches:
                    if os.path.isfile(match):
                        driver_path = match
                        logger.debug(f"Found chromedriver at: {driver_path}")
                        break
    
    # Fix permissions on macOS - make chromedriver executable
    if os.path.isfile(driver_path):
        try:
            # Set execute permissions (chmod +x)
            current_perms = os.stat(driver_path).st_mode
            os.chmod(driver_path, current_perms | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            logger.debug(f"Set execute permissions on {driver_path}")
        except Exception as e:
            logger.warning(f"Could not set permissions on chromedriver: {e}")
    
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Remove webdriver property to avoid detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


def fetch_game_logs_selenium(driver: webdriver.Chrome, team: str, year: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """
    Fetch advanced game logs using Selenium WebDriver.
    
    Args:
        driver: Selenium WebDriver instance
        team: Team abbreviation (e.g., 'ATL', 'BOS')
        year: Basketball-Reference year (e.g., '2025' for 2024-25 season)
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame containing the game logs, or None if fetch fails
    """
    url = BASE_URL.format(team=team, year=year)
    
    for attempt in range(max_retries):
        try:
            # Check if driver is still alive
            try:
                driver.current_url
            except Exception:
                logger.warning(f"Driver session lost for {team} {year}, attempt {attempt + 1}")
                return None
            
            logger.debug(f"Navigating to {url} (attempt {attempt + 1}/{max_retries})")
            driver.get(url)
            
            # Wait for the table to load with longer timeout
            wait = WebDriverWait(driver, 15)
            table = wait.until(
                EC.presence_of_element_located((By.ID, "team_game_log_adv_reg"))
            )
            
            # Give the page extra time to fully render
            time.sleep(PAGE_LOAD_WAIT + 1)
            
            # Get the page source and parse with pandas
            page_source = driver.page_source
            
            # Use pandas to parse the HTML table (wrap in StringIO to avoid deprecation warning)
            from io import StringIO
            try:
                dfs = pd.read_html(StringIO(page_source), attrs={'id': 'team_game_log_adv_reg'})
                if dfs:
                    df = dfs[0]
                else:
                    # Fallback: try parsing all tables
                    dfs = pd.read_html(StringIO(page_source))
                    # Find the table with the game logs (usually the one with the most columns and ORtg)
                    df = None
                    for potential_df in dfs:
                        cols_str = str(potential_df.columns).lower()
                        if ('ortg' in cols_str or 'off_rtg' in cols_str) and len(potential_df.columns) > 20:
                            df = potential_df
                            break
            except Exception as e:
                logger.error(f"Error parsing HTML table: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            if df is None or df.empty:
                logger.warning(f"Could not find game log table for {team} {year}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching {team} {year} (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
            
            # Check if it's a recoverable error
            if "target window already closed" in error_msg or "target frame detached" in error_msg:
                logger.warning(f"Browser window closed, cannot retry for {team} {year}")
                return None
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return None
    
    return None


def clean_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format the game logs DataFrame to match the expected CSV structure.
    
    Args:
        df: Raw DataFrame from Basketball-Reference
        
    Returns:
        Cleaned DataFrame ready for export
    """
    if df is None or df.empty:
        return df
    
    # Handle multi-level column headers from Basketball-Reference
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-level columns
        df.columns = [' '.join(col).strip() if col[1] else col[0] for col in df.columns.values]
    
    # Find the rank column (could be 'Rk', 'Rank', or similar)
    rank_col = None
    for col in df.columns:
        if str(col).strip().lower() in ['rk', 'rank', 'ranker']:
            rank_col = col
            break
    
    # Remove header rows repeated in the middle
    if rank_col and rank_col in df.columns:
        df = df[df[rank_col] != rank_col].copy()
        # Remove footer rows (usually summaries or empty rows)
        df = df[df[rank_col].notna()].copy()
        df = df[df[rank_col] != ''].copy()
    else:
        # If no rank column, just remove completely empty rows
        df = df.dropna(how='all').copy()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Convert numeric columns to proper types
    numeric_columns = ['Rk', 'Gtm', 'Tm', 'Opp', 'ORtg', 'DRtg', 'Pace', 
                      'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%',
                      'eFG%', 'TOV%', 'ORB%', 'FT/FGA']
    
    for col in numeric_columns:
        # Handle potential multi-level column names
        matching_cols = [c for c in df.columns if col in str(c)]
        for matching_col in matching_cols:
            try:
                df[matching_col] = pd.to_numeric(df[matching_col], errors='ignore')
            except Exception:
                pass  # Skip if conversion fails
    
    return df


def save_to_csv(df: pd.DataFrame, team: str, season: str, data_dir: str = 'data') -> bool:
    """
    Save game logs DataFrame to CSV file in the appropriate directory.
    
    Args:
        df: DataFrame containing game logs
        team: Team abbreviation (Basketball-Reference format, e.g., 'BRK', 'CHO', 'PHO')
        season: Season string (e.g., '2024-25')
        data_dir: Base data directory
        
    Returns:
        True if save successful, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"No data to save for {team} {season}")
        return False
    
    try:
        # Create directory if it doesn't exist
        season_dir = os.path.join(data_dir, season)
        os.makedirs(season_dir, exist_ok=True)
        
        # Map team abbreviation to filename format (BRK->BKN, CHO->CHA, PHO->PHX)
        file_team = TEAM_NAME_MAPPING.get(team, team)
        
        # Construct file path
        filename = f"{file_team}_{season}_gamelogs.csv"
        filepath = os.path.join(season_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.debug(f"Saved {filename} ({len(df)} games)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving {team} {season}: {e}")
        return False


def scrape_all_teams(headless: bool = False, teams_to_scrape: Optional[list] = None):
    """
    Main function to scrape game logs for all teams and seasons using Selenium.
    
    Args:
        headless: If True, run browser in headless mode (no visible window)
        teams_to_scrape: Optional list of team abbreviations to scrape. If None, scrapes all teams.
                        Use Basketball-Reference abbreviations (BRK, CHO, PHO for Brooklyn, Charlotte, Phoenix)
    """
    # Use specified teams or all teams
    teams = teams_to_scrape if teams_to_scrape else TEAMS
    
    total_teams = len(teams) * len(SEASONS)
    successful = 0
    failed = 0
    current = 0
    
    print("\n" + "="*70)
    print(f"  NBA GAME LOGS SCRAPER - Selenium Version")
    print("="*70)
    if teams_to_scrape:
        print(f"Teams: {len(teams)} (specific: {', '.join(teams)}) | Seasons: {len(SEASONS)} | Total files: {total_teams}")
    else:
        print(f"Teams: {len(teams)} | Seasons: {len(SEASONS)} | Total files: {total_teams}")
    print(f"Estimated time: ~{(total_teams * (PAGE_LOAD_WAIT + REQUEST_DELAY) / 60):.1f} minutes")
    print(f"Browser mode: {'Headless' if headless else 'Visible'}")
    print("="*70 + "\n")
    
    logger.info(f"Starting Selenium scrape for {len(teams)} teams across {len(SEASONS)} seasons")
    logger.info(f"Total files to process: {total_teams}")
    
    # Set up Chrome driver
    print("🔧 Setting up Chrome WebDriver...")
    try:
        driver = setup_driver(headless=headless)
        print("✓ Chrome WebDriver ready\n")
    except Exception as e:
        print(f"✗ Failed to set up Chrome WebDriver: {e}")
        print("\nPlease ensure Chrome browser is installed on your system.")
        return
    
    try:
        for season, year in SEASONS.items():
            print(f"\n📅 SEASON: {season} (Basketball-Reference year: {year})")
            print("-" * 70)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing season: {season} (Basketball-Reference year: {year})")
            logger.info(f"{'='*60}\n")
            
            for team in teams:
                current += 1
                
                # Print progress indicator
                print(f"[{current:3d}/{total_teams}] {team} {season}...", end=" ", flush=True)
                
                # Check if driver is still alive, recreate if needed
                try:
                    driver.current_url
                except Exception:
                    logger.warning("Driver crashed, recreating...")
                    try:
                        driver.quit()
                    except:
                        pass
                    try:
                        driver = setup_driver(headless=headless)
                        logger.info("Driver recreated successfully")
                    except Exception as e:
                        logger.error(f"Failed to recreate driver: {e}")
                        print("✗ (driver error)")
                        failed += 1
                        continue
                
                # Fetch game logs
                df = fetch_game_logs_selenium(driver, team, year)
                
                if df is not None:
                    # Clean data
                    df_cleaned = clean_game_logs(df)
                    
                    # Save to CSV
                    if save_to_csv(df_cleaned, team, season):
                        print(f"✓ ({len(df_cleaned)} games)")
                        successful += 1
                    else:
                        print("✗ (save failed)")
                        failed += 1
                else:
                    print("✗ (fetch failed)")
                    failed += 1
                
                # Rate limiting - be respectful to Basketball-Reference
                time.sleep(REQUEST_DELAY)
        
    finally:
        # Always close the browser
        driver.quit()
        print("\n🔧 Chrome WebDriver closed")
    
    # Final summary
    print("\n" + "="*70)
    print("  SCRAPING COMPLETE!")
    print("="*70)
    print(f"Total files:  {total_teams}")
    print(f"✓ Successful: {successful} ({(successful/total_teams)*100:.1f}%)")
    print(f"✗ Failed:     {failed} ({(failed/total_teams)*100:.1f}%)")
    print("="*70 + "\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SCRAPING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {total_teams}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {(successful/total_teams)*100:.1f}%")


if __name__ == "__main__":
    import sys
    
    # Check for --headless flag
    headless_mode = "--headless" in sys.argv
    
    # Parse team arguments (everything that's not a flag)
    teams_to_scrape = None
    if len(sys.argv) > 1:
        # Get all arguments that aren't flags
        team_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
        if team_args:
            # Validate teams
            valid_teams = []
            for team in team_args:
                team_upper = team.upper()
                if team_upper in TEAMS:
                    valid_teams.append(team_upper)
                else:
                    print(f"⚠️  Warning: '{team}' is not a valid team abbreviation. Skipping.")
            if valid_teams:
                teams_to_scrape = valid_teams
                print(f"\n📋 Scraping specific teams: {', '.join(teams_to_scrape)}\n")
    
    if not headless_mode and not teams_to_scrape:
        print("\n💡 TIPS:")
        print("   - Run with --headless flag to hide the browser window")
        print("   - Specify teams: python scrape_gamelogs_selenium.py BRK CHO PHO")
        print("   - Combine: python scrape_gamelogs_selenium.py --headless BRK CHO PHO\n")
    
    scrape_all_teams(headless=headless_mode, teams_to_scrape=teams_to_scrape)

