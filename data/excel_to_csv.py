import pandas as pd
import glob
import os

def convert_cap_files():
    raw_path = 'data/raw/'
    files = glob.glob(os.path.join(raw_path, "player_caps_*.xlsx"))
    
    for file in files:
        filename = os.path.basename(file).replace('.xlsx', '.csv')
        print(f"Converting {file}...")
        # engine='openpyxl' handles .xlsx; we skip the logo column (Col C) if needed
        df = pd.read_excel(file, engine='openpyxl')
        
        # Save to raw folder as CSV for the cleaning script to pick up
        df.to_csv(os.path.join(raw_path, filename), index=False)
    print("✅ All Excel files converted to CSV.")

if __name__ == "__main__":
    convert_cap_files()