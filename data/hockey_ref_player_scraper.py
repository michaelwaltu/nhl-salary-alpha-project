import pandas as pd
import time

def get_nhl_data(start_year, end_year):
    all_seasons = []

    for year in range(start_year, end_year + 1):
        print(f"Fetching data for {year}...")
        
        simple_url = f"https://www.hockey-reference.com/leagues/NHL_{year}_skaters.html"
        adv_url = f"https://www.hockey-reference.com/leagues/NHL_{year}_skaters-advanced.html"
        toi_url = f"https://www.hockey-reference.com/leagues/NHL_{year}_skaters-time-on-ice.html"

        try:
            # 1. Read the tables
            df_simple = pd.read_html(simple_url)[0]
            time.sleep(3)
            df_adv = pd.read_html(adv_url)[0]
            time.sleep(3)
            df_toi = pd.read_html(toi_url)[0]

            # 2. Flatten MultiIndex headers for simple and advanced only
            if isinstance(df_simple.columns, pd.MultiIndex):
                df_simple.columns = df_simple.columns.get_level_values(-1)
            if isinstance(df_adv.columns, pd.MultiIndex):
                df_adv.columns = df_adv.columns.get_level_values(-1)

            # For TOI table, flatten first, then rename duplicate columns by position
            if isinstance(df_toi.columns, pd.MultiIndex):
                df_toi.columns = df_toi.columns.get_level_values(-1)

            # Rename TOI table columns so the three TOI columns are distinct
            toi_cols = list(df_toi.columns)
            if len(toi_cols) >= 21:
                toi_cols[2] = 'Team'      # Tm
                toi_cols[7] = 'EV TOI'
                toi_cols[12] = 'PP TOI'
                toi_cols[17] = 'SH TOI'
                df_toi.columns = toi_cols

            # 3. Standardize column names and clean repeating headers
            for df in [df_simple, df_adv, df_toi]:
                df.rename(columns={'Tm': 'Team', 'Tm_x': 'Team', 'Tm_y': 'Team'}, inplace=True)
                if '-9999' in df.columns:
                    df.rename(columns={'-9999': 'player_id'}, inplace=True)

            df_simple = df_simple[df_simple['Player'] != 'Player'].copy()
            df_adv = df_adv[df_adv['Player'] != 'Player'].copy()
            df_toi = df_toi[df_toi['Player'] != 'Player'].copy()

            # 4. Force Age and GP to numeric in the original two tables
            for df in [df_simple, df_adv]:
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
                df['GP'] = pd.to_numeric(df['GP'], errors='coerce')

            if 'GP' in df_toi.columns:
                df_toi['GP'] = pd.to_numeric(df_toi['GP'], errors='coerce')

            # 5. Merge advanced + simple exactly like before
            common_cols = ['Player', 'Age', 'Team', 'Pos', 'GP']
            merged_year = pd.merge(df_adv, df_simple, on=common_cols, how='left', suffixes=('', '_drop'))
            merged_year = merged_year[[c for c in merged_year.columns if not c.endswith('_drop')]]

            # 6. Pull only PP TOI and SH TOI from TOI page
            toi_keep = ['Player', 'Team', 'Pos', 'GP']
            if 'PP TOI' in df_toi.columns:
                toi_keep.append('PP TOI')
            if 'SH TOI' in df_toi.columns:
                toi_keep.append('SH TOI')

            df_toi_small = df_toi[toi_keep].copy()

            merged_year = pd.merge(
                merged_year,
                df_toi_small,
                on=['Player', 'Team', 'Pos', 'GP'],
                how='left',
                suffixes=('', '_toi_drop')
            )

            merged_year = merged_year[[c for c in merged_year.columns if not c.endswith('_toi_drop')]]

            merged_year['Season'] = year
            all_seasons.append(merged_year)

            print(f"Successfully processed {year}!")
            time.sleep(2)

        except Exception as e:
            print(f"Error on year {year}: {e}")
            continue

    if all_seasons:
        final_df = pd.concat(all_seasons, ignore_index=True)
        final_df.to_csv('hr_player_stats.csv', index=False)
        print("\n--- DONE! ---")
        print(f"Saved {len(final_df)} total rows to hr_player_stats.csv")
    else:
        print("No data was collected.")

if __name__ == "__main__":
    get_nhl_data(2015, 2025)