import pandas as pd
import glob
import os
import re
import numpy as np
from src.utils import (
    normalize_team,
    clean_currency,
    clean_player_name,
    calculate_season_age,
    calculate_correct_pct,
    toi_to_float,
)
from src.renaming_map import apply_renaming_map

def process_pipeline():
    os.makedirs('data/processed', exist_ok=True)

    # Load in hockey reference data
    hr = pd.read_csv('data/raw/hr_player_stats.csv')
    hr['Player'] = hr['Player'].apply(clean_player_name)
    hr['Team'] = hr['Team'].apply(normalize_team)

    # Handle TOT by keeping aggregate row for traded players
    hr['is_tot'] = hr['Team'] == 'TOT'
    hr = hr.sort_values(['Player', 'Season', 'is_tot'], ascending=[True, True, False])
    hr = hr.drop_duplicates(subset=['Player', 'Season'], keep='first').drop(columns=['is_tot'])

    # Load in MoneyPuck data
    mp = pd.read_csv('data/raw/mp_skaters.csv')
    mp = mp[mp['situation'] == 'all'].copy()
    mp['name'] = mp['name'].apply(clean_player_name)
    mp['team'] = mp['team'].apply(normalize_team)

    # Load and process salary data
    salary_list = []
    cap_files = glob.glob('data/raw/player_caps_*.csv')

    for file in cap_files:
        match = re.search(r'(\d{2})-(\d{2})', file)
        if match:
            year_suffix = match.group(1)
            season_start_year = 2000 + int(year_suffix)

            df_sal = pd.read_csv(file)
            df_sal.columns = ['Rank', 'Player', 'Logo_Ignore', 'Current_Age', 'Pos', 'Term', 'Cap_Hit', 'Start', 'End']

            df_sal['Player'] = df_sal['Player'].apply(clean_player_name)

            # clean_currency returns millions
            df_sal['Cap_Hit'] = df_sal['Cap_Hit'].apply(clean_currency)

            df_sal['Age_at_Season'] = df_sal['Current_Age'].apply(
                lambda x: calculate_season_age(x, season_start_year)
            )
            df_sal['Season'] = season_start_year

            salary_list.append(df_sal[['Player', 'Age_at_Season', 'Season', 'Cap_Hit', 'Pos']])

    salaries = pd.concat(salary_list).drop_duplicates(
        subset=['Player', 'Season', 'Age_at_Season', 'Pos']
    )

    # merge all data sources together
    master = pd.merge(
        hr,
        mp,
        left_on=['Player', 'Season', 'Team'],
        right_on=['name', 'season', 'team'],
        how='inner'
    )

    # Merge on Name, Season, and Age to separate players with identical names
    final_df = pd.merge(
        master,
        salaries,
        left_on=['Player', 'Season', 'Age'],
        right_on=['Player', 'Season', 'Age_at_Season'],
        how='left'
    )

    '''
    Sometimes G / A / PTS are Zeroed-Out in HR but MP has values
    If HR scoring columns are zero but MoneyPuck has positive values,
    overwrite with the MP equivalents.
    '''
    mask_g = (
        final_df['G'].fillna(0).eq(0) &
        final_df['I_F_goals'].fillna(0).gt(0)
    )
    final_df.loc[mask_g, 'G'] = final_df.loc[mask_g, 'I_F_goals']

    mask_a = (
        final_df['A'].fillna(0).eq(0) &
        final_df['I_F_points'].fillna(0).gt(0)
    )
    final_df.loc[mask_a, 'A'] = (
        final_df.loc[mask_a, 'I_F_points'].fillna(0) -
        final_df.loc[mask_a, 'I_F_goals'].fillna(0)
    )

    mask_pts = (
        final_df['PTS'].fillna(0).eq(0) &
        final_df['I_F_points'].fillna(0).gt(0)
    )
    final_df.loc[mask_pts, 'PTS'] = final_df.loc[mask_pts, 'I_F_points']

    # Final Cleaning:

    # Only have salary data from 2018 onward
    final_df = final_df[final_df['Season'] >= 2018].copy()
    final_df = final_df.dropna(subset=['Cap_Hit'])

    # Fix Cap Pct Calculation
    final_df['Cap_Pct'] = final_df.apply(calculate_correct_pct, axis=1)

    # Convert TOI/60 from MM:SS to decimal minutes
    final_df['TOI/60'] = final_df['TOI/60'].apply(toi_to_float)
    final_df['PP TOI'] = final_df['PP TOI'].apply(toi_to_float)
    final_df['SH TOI'] = final_df['SH TOI'].apply(toi_to_float)
    final_df['TOI(EV)'] = final_df['TOI(EV)'].apply(toi_to_float)

    # Clean numeric nulls
    num_cols = final_df.select_dtypes(include=[np.number]).columns
    final_df[num_cols] = final_df[num_cols].fillna(0)

    # Set FO% to 0 for players with fewer than 20 total face-offs
    final_df.loc[(final_df['FOW'] + final_df['FOL']) < 20, 'FO%'] = 0

    # Drop irrelevant columns
    final_df = final_df.drop(
        columns=[
            'Awards', 'Logo_Ignore', 'Age_at_Season',
            'name', 'season', 'team',
            'position_y', 'Pos_y'
        ],
        errors='ignore'
    )

    final_df = final_df.rename(columns={'Pos_x': 'Pos'})

    # Remove duplicate columns by content
    final_df = final_df.loc[:, ~final_df.T.duplicated()]

    # Keep one row per player-season-team, preferring the most complete record
    final_df['_non_null_count'] = final_df.notna().sum(axis=1)
    final_df = final_df.sort_values(
        ['Player', 'Season', 'Team', '_non_null_count'],
        ascending=[True, True, True, False],
    )
    final_df = final_df.drop_duplicates(
        subset=['Player', 'Season', 'Team'],
        keep='first',
    ).drop(columns=['_non_null_count'])

    # Apply final rename map from renaming_map.py
    final_df = apply_renaming_map(final_df)

    duplicate_count = final_df.duplicated(['Player', 'Season', 'Team']).sum()
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate player-season-team rows after final dedupe."
        )

    final_df.to_csv('data/processed/cleaned_data.csv', index=False)
    print("Created 'cleaned_data.csv' with renamed columns applied.")

if __name__ == "__main__":
    process_pipeline()
