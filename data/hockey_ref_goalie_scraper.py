import pandas as pd
import time

def get_goalie_data(start_year, end_year):
    all_seasons = []

    for year in range(start_year, end_year + 1):
        print(f"Fetching Goalie data for {year}...")
        
        # Goalie stats URL
        url = f"https://www.hockey-reference.com/leagues/NHL_{year}_goalies.html"

        try:
            # 1. Read the table
            # Hockey-Ref usually has one main table for goalies
            df = pd.read_html(url)[0]
            
            # 2. Flatten MultiIndex headers (Goalies have "Record", "Goals", etc. headers)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)

            # 3. Standardize column names
            df.rename(columns={'Tm': 'Team', 'Tm_x': 'Team', 'Tm_y': 'Team'}, inplace=True)
            if '-9999' in df.columns:
                df.rename(columns={'-9999': 'player_id'}, inplace=True)

            # 4. Clean repeating headers and "Player" rows
            df = df[df['Player'] != 'Player'].copy()

            # 5. Fix Data Types for merging/sorting
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            df['GP'] = pd.to_numeric(df['GP'], errors='coerce')
            df['W'] = pd.to_numeric(df['W'], errors='coerce')
            df['SV%'] = pd.to_numeric(df['SV%'], errors='coerce')

            # 6. Add Season identifier
            df['Season'] = year
            
            all_seasons.append(df)
            print(f"Successfully processed {year}!")
            
            # Pause to be respectful to the server
            time.sleep(4)

        except Exception as e:
            print(f"Error on year {year}: {e}")
            continue

    if all_seasons:
        final_df = pd.concat(all_seasons, ignore_index=True)
        
        # Save to the specific filename you requested
        final_df.to_csv('hr_goalie_stats.csv', index=False)
        
        print("\n--- DONE! ---")
        print(f"Saved {len(final_df)} total goalie rows to hr_goalie_stats.csv")
    else:
        print("No goalie data was collected.")

if __name__ == "__main__":
    # Running from 2008 to 2025 as before
    get_goalie_data(2008, 2025)