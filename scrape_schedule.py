#!/usr/bin/env python3
"""
NBA Schedule Scraper

Scrapes the complete NBA schedule for a season from Basketball-Reference.com.
Extracts all scheduled games (completed and upcoming) for all 30 teams.
"""

import os
import time
import pandas as pd
from typing import Optional, Set, Tuple
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import glob
import stat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - same as game logs scraper
TEAMS = [
    'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Mapping from Basketball-Reference abbreviations to standard
TEAM_NAME_MAPPING = {
    'BRK': 'BKN',  # Brooklyn Nets
    'CHO': 'CHA',  # Charlotte Hornets
    'PHO': 'PHX',  # Phoenix Suns
}

# Reverse mapping (standard to Basketball-Reference)
REVERSE_TEAM_MAPPING = {v: k for k, v in TEAM_NAME_MAPPING.items()}

SEASONS = {
    '2025-26': '2026',
    '2024-25': '2025',
    '2023-24': '2024',
    '2022-23': '2023',
    '2021-22': '2022',
    '2020-21': '2021',
    '2019-20': '2020',
}

# URL for team schedule page
SCHEDULE_URL = "https://www.basketball-reference.com/teams/{team}/{year}_games.html"

# Rate limiting
PAGE_LOAD_WAIT = 3
REQUEST_DELAY = 1


def setup_driver(headless: bool = False) -> webdriver.Chrome:
    """Set up and configure Chrome WebDriver (same as game logs scraper)."""
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")
    
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Get driver path
    driver_path = ChromeDriverManager().install()
    
    if os.path.isfile(driver_path):
        driver_dir = os.path.dirname(driver_path)
        possible_paths = [
            os.path.join(driver_dir, 'chromedriver'),
            os.path.join(driver_dir, 'chromedriver.exe'),
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                driver_path = path
                break
        else:
            pattern = os.path.join(os.path.dirname(driver_dir), '**', 'chromedriver')
            matches = glob.glob(pattern, recursive=True)
            if matches:
                for match in matches:
                    if os.path.isfile(match):
                        driver_path = match
                        break
    
    # Fix permissions
    if os.path.isfile(driver_path):
        try:
            current_perms = os.stat(driver_path).st_mode
            os.chmod(driver_path, current_perms | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except Exception as e:
            logger.warning(f"Could not set permissions on chromedriver: {e}")
    
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


def normalize_team_name(team: str, to_standard: bool = True) -> str:
    """Normalize team name between Basketball-Reference and standard formats."""
    if to_standard:
        return TEAM_NAME_MAPPING.get(team, team)
    else:
        return REVERSE_TEAM_MAPPING.get(team, team)


def parse_date(date_str: str) -> Optional[str]:
    """
    Parse date string from Basketball-Reference format.
    Formats: "Mon, Oct 27, 2025" or "Oct 27, 2025" -> "2025-10-27"
    """
    try:
        # Try different date formats
        for fmt in ['%a, %b %d, %Y', '%b %d, %Y', '%Y-%m-%d']:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Fallback: try pandas parsing
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        return None


def extract_schedule_from_table(driver: webdriver.Chrome, team: str) -> pd.DataFrame:
    """
    Extract schedule data from the team's schedule table using Selenium directly.
    
    Args:
        driver: Selenium WebDriver instance
        team: Team abbreviation (Basketball-Reference format)
    
    Returns:
        DataFrame with columns: Game_Number, Date, Home_Team, Away_Team
    """
    try:
        # Wait for schedule table
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.ID, "games")))
        
        # Give page time to render
        time.sleep(PAGE_LOAD_WAIT)
        
        # Find all table rows (skip header rows)
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        schedule_rows = []
        team_standard = normalize_team_name(team, to_standard=True)
        
        for row in rows:
            # Skip header rows (they have class 'thead')
            row_class = row.get_attribute('class') or ''
            if 'thead' in row_class:
                continue
            
            # Get all cells in the row
            cells = row.find_elements(By.TAG_NAME, "td")
            th_cells = row.find_elements(By.TAG_NAME, "th")
            
            if len(cells) == 0 and len(th_cells) == 0:
                continue
            
            # Game number is in the first <th> (data-stat="g")
            game_num = None
            if len(th_cells) > 0:
                try:
                    game_num_text = th_cells[0].text.strip()
                    if game_num_text and game_num_text.isdigit():
                        game_num = int(game_num_text)
                except:
                    pass
            
            if game_num is None:
                continue  # Skip rows without valid game numbers
            
            # Date is in a <td> with data-stat="date_game" - get from link text
            date_str = None
            date_cell = None
            for cell in cells:
                if cell.get_attribute('data-stat') == 'date_game':
                    date_cell = cell
                    # Try to get date from link
                    try:
                        link = cell.find_element(By.TAG_NAME, "a")
                        date_str = link.text.strip()
                    except:
                        # Fallback: get from cell text or csk attribute
                        date_str = cell.text.strip()
                        if not date_str:
                            date_str = cell.get_attribute('csk')
                    break
            
            if not date_str:
                continue
            
            parsed_date = parse_date(date_str)
            if parsed_date is None:
                continue
            
            # Opponent is in a <td> with data-stat="opp_name" - get team abbreviation from link href
            opp_abbr = None
            is_home = True
            
            # First check game_location column (data-stat="game_location") for @ symbol
            for cell in cells:
                if cell.get_attribute('data-stat') == 'game_location':
                    location_text = cell.text.strip()
                    if location_text == '@':
                        is_home = False
                    break
            
            # Now get opponent abbreviation from opp_name cell
            for cell in cells:
                if cell.get_attribute('data-stat') == 'opp_name':
                    try:
                        # Get team abbreviation from link href (e.g., /teams/TOR/2026.html -> TOR)
                        link = cell.find_element(By.TAG_NAME, "a")
                        href = link.get_attribute('href')
                        if href and '/teams/' in href:
                            # Extract team abbreviation from URL
                            parts = href.split('/teams/')
                            if len(parts) > 1:
                                opp_abbr = parts[1].split('/')[0].upper()
                    except:
                        # Fallback: try to extract from cell text
                        cell_text = cell.text.strip()
                        # Could be full team name, try to match
                        pass
                    break
            
            if not opp_abbr:
                continue
            
            # Normalize opponent abbreviation
            opp_standard = normalize_team_name(opp_abbr, to_standard=True)
            
            # Skip if opponent not recognized
            if len(opp_standard) != 3 or opp_standard not in [normalize_team_name(t, True) for t in TEAMS]:
                continue
            
            # Determine home and away teams
            if is_home:
                home_team = team_standard
                away_team = opp_standard
            else:
                home_team = opp_standard
                away_team = team_standard
            
            # Create schedule entry
            schedule_rows.append({
                'Game_Number': game_num,
                'Date': parsed_date,
                'Home_Team': home_team,
                'Away_Team': away_team,
            })
        
        if schedule_rows:
            return pd.DataFrame(schedule_rows)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error extracting schedule for {team}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def fetch_team_schedule(driver: webdriver.Chrome, team: str, year: str) -> pd.DataFrame:
    """
    Fetch schedule for a single team.
    
    Args:
        driver: Selenium WebDriver instance
        team: Team abbreviation (Basketball-Reference format)
        year: Basketball-Reference year (e.g., '2026' for 2025-26 season)
    
    Returns:
        DataFrame with schedule data
    """
    url = SCHEDULE_URL.format(team=team, year=year)
    
    try:
        logger.debug(f"Fetching schedule: {url}")
        driver.get(url)
        
        # Extract schedule
        schedule_df = extract_schedule_from_table(driver, team)
        
        if schedule_df.empty:
            logger.warning(f"No schedule data found for {team} {year}")
        
        return schedule_df
        
    except Exception as e:
        logger.error(f"Error fetching schedule for {team} {year}: {e}")
        return pd.DataFrame()


def scrape_season_schedule(season: str, headless: bool = False, output_file: str = None) -> pd.DataFrame:
    """
    Scrape complete schedule for a season.
    
    Args:
        season: Season string (e.g., '2025-26')
        headless: If True, run browser in headless mode
        output_file: Optional path to save CSV (default: nba_schedule_{season}.csv)
    
    Returns:
        DataFrame with all games (deduplicated)
    """
    if season not in SEASONS:
        raise ValueError(f"Season {season} not in SEASONS dict")
    
    year = SEASONS[season]
    
    print(f"\n{'='*70}")
    print(f"  NBA SCHEDULE SCRAPER - {season}")
    print("="*70)
    print(f"Teams: {len(TEAMS)} | Season: {season} (BR year: {year})")
    print(f"Estimated time: ~{(len(TEAMS) * (PAGE_LOAD_WAIT + REQUEST_DELAY) / 60):.1f} minutes")
    print(f"Browser mode: {'Headless' if headless else 'Visible'}")
    print("="*70 + "\n")
    
    # Set up driver
    print("🔧 Setting up Chrome WebDriver...")
    try:
        driver = setup_driver(headless=headless)
        print("✓ Chrome WebDriver ready\n")
    except Exception as e:
        print(f"✗ Failed to set up Chrome WebDriver: {e}")
        return pd.DataFrame()
    
    all_schedules = []
    
    try:
        for i, team in enumerate(TEAMS, 1):
            print(f"[{i:2d}/{len(TEAMS)}] {team}...", end=" ", flush=True)
            
            schedule_df = fetch_team_schedule(driver, team, year)
            
            if not schedule_df.empty:
                all_schedules.append(schedule_df)
                print(f"✓ ({len(schedule_df)} games)")
            else:
                print("✗")
            
            # Rate limiting
            if i < len(TEAMS):
                time.sleep(REQUEST_DELAY)
    
    finally:
        driver.quit()
        print("\n🔧 Chrome WebDriver closed")
    
    if not all_schedules:
        print("\n❌ No schedule data collected")
        return pd.DataFrame()
    
    # Combine all schedules
    combined_df = pd.concat(all_schedules, ignore_index=True)
    
    # Deduplicate games (each game appears twice - once per team)
    # Keep unique combinations of Date, Home_Team, Away_Team
    combined_df = combined_df.drop_duplicates(subset=['Date', 'Home_Team', 'Away_Team'], keep='first')
    
    # Sort by date
    combined_df = combined_df.sort_values(['Date', 'Home_Team']).reset_index(drop=True)
    
    # Save to CSV
    if output_file is None:
        output_file = f'nba_schedule_{season}.csv'
    
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("  SCHEDULE SCRAPING COMPLETE")
    print("="*70)
    print(f"Total games found: {len(combined_df)}")
    print(f"Saved to: {output_file}")
    print("="*70 + "\n")
    
    return combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape NBA schedule from Basketball-Reference')
    parser.add_argument('--season', type=str, default='2025-26', help='Season to scrape (e.g., 2025-26)')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    schedule = scrape_season_schedule(args.season, headless=args.headless, output_file=args.output)
    
    if not schedule.empty:
        print("\nSample schedule:")
        print(schedule.head(10).to_string(index=False))
        print(f"\n... and {len(schedule) - 10} more games")

