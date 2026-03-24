# src/utils.py
import re
import numpy as np

TEAM_MAP = {
    "N.J": "NJD", "N.Y.R": "NYR", "N.Y.I": "NYI", "T.B": "TBL",
    "L.A": "LAK", "S.J": "SJS", "ARI": "ARI", "PHX": "ARI",
    "UTA": "UTA", "VEG": "VGK", "VGK": "VGK", "WPG": "WPG", "ATL": "WPG",
}

CAP_CEILINGS = {
    2018: 79.5,
    2019: 81.5,
    2020: 81.5,
    2021: 81.5,
    2022: 82.5,
    2023: 83.5,
    2024: 88.0,
}

def normalize_team(team_str):
    clean_name = team_str.strip().upper()
    return TEAM_MAP.get(clean_name, clean_name)


def clean_currency(value):
    clean_val = re.sub(r'[$,\s]', '', str(value))
    try:
        return float(clean_val) / 1_000_000
    except ValueError:
        return 0.0


def clean_player_name(name):
    name = name.replace('*', '')
    # Handle 'Last, First' -> 'First Last'
    if ',' in name:
        parts = name.split(',', 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()


def calculate_season_age(current_age, season_year):
    REFERENCE_YEAR = 2026
    years_to_subtract = REFERENCE_YEAR - int(season_year)
    return current_age - years_to_subtract


def calculate_correct_pct(row):
    ceiling_dollars = CAP_CEILINGS.get(row['Season'], 88.0) * 1_000_000
    val = row['Cap_Hit']
    # clean_currency stores Cap_Hit in millions, so convert to dollars
    if val < 1000:
        val = val * 1_000_000
    return (val / ceiling_dollars) * 100


def toi_to_float(toi_str):
    # Convert MM:SS string to decimal minutes.
    try:
        parts = str(toi_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except (TypeError, ValueError, IndexError):
        return np.nan
