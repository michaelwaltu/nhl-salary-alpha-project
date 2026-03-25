import pandas as pd
import glob
import re
from pathlib import Path
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

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_hr_data() -> pd.DataFrame:
    """Load and clean Hockey Reference player data."""
    hr = pd.read_csv(RAW_DIR / "hr_player_stats.csv")

    hr["Player"] = hr["Player"].apply(clean_player_name)
    hr["Team"] = hr["Team"].apply(normalize_team)

    # Handle traded players (keep TOT aggregate row)
    hr["is_tot"] = hr["Team"] == "TOT"
    hr = hr.sort_values(["Player", "Season", "is_tot"], ascending=[True, True, False])
    hr = hr.drop_duplicates(subset=["Player", "Season"], keep="first").drop(columns=["is_tot"])

    return hr


def load_mp_data() -> pd.DataFrame:
    """Load and clean MoneyPuck skater data."""
    mp = pd.read_csv(RAW_DIR / "mp_skaters.csv")

    mp = mp[mp["situation"] == "all"].copy()
    mp["name"] = mp["name"].apply(clean_player_name)
    mp["team"] = mp["team"].apply(normalize_team)

    return mp


def load_salary_data() -> pd.DataFrame:
    """Load and process salary/cap data across seasons."""
    salary_list = []
    cap_files = glob.glob(str(RAW_DIR / "player_caps_*.csv"))

    for file in cap_files:
        match = re.search(r"(\d{2})-(\d{2})", file)
        if not match:
            continue

        season_start_year = 2000 + int(match.group(1))

        df_sal = pd.read_csv(file)
        df_sal.columns = [
            "Rank", "Player", "Logo_Ignore", "Current_Age",
            "Pos", "Term", "Cap_Hit", "Start", "End"
        ]

        df_sal["Player"] = df_sal["Player"].apply(clean_player_name)
        df_sal["Cap_Hit"] = df_sal["Cap_Hit"].apply(clean_currency)

        df_sal["Age_at_Season"] = df_sal["Current_Age"].apply(
            lambda x: calculate_season_age(x, season_start_year)
        )
        df_sal["Season"] = season_start_year

        salary_list.append(
            df_sal[["Player", "Age_at_Season", "Season", "Cap_Hit", "Pos"]]
        )

    if not salary_list:
        raise ValueError("No salary files found in data/raw")

    salaries = pd.concat(salary_list).drop_duplicates(
        subset=["Player", "Season", "Age_at_Season", "Pos"]
    )

    return salaries


def merge_datasets(
    hr: pd.DataFrame,
    mp: pd.DataFrame,
    salaries: pd.DataFrame,
) -> pd.DataFrame:
    """Merge HR, MoneyPuck, and salary datasets."""
    mp_and_hr = pd.merge(
        hr,
        mp,
        left_on=["Player", "Season", "Team"],
        right_on=["name", "season", "team"],
        how="inner",
    )

    final_df = pd.merge(
        mp_and_hr,
        salaries,
        left_on=["Player", "Season", "Age"],
        right_on=["Player", "Season", "Age_at_Season"],
        how="left",
    )

    return final_df


def fix_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Fix zeroed-out HR scoring stats using MoneyPuck data."""
    mask_g = df["G"].fillna(0).eq(0) & df["I_F_goals"].fillna(0).gt(0)
    df.loc[mask_g, "G"] = df.loc[mask_g, "I_F_goals"]

    mask_a = df["A"].fillna(0).eq(0) & df["I_F_points"].fillna(0).gt(0)
    df.loc[mask_a, "A"] = (
        df.loc[mask_a, "I_F_points"].fillna(0)
        - df.loc[mask_a, "I_F_goals"].fillna(0)
    )

    mask_pts = df["PTS"].fillna(0).eq(0) & df["I_F_points"].fillna(0).gt(0)
    df.loc[mask_pts, "PTS"] = df.loc[mask_pts, "I_F_points"]

    return df


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Perform final cleaning, filtering, and formatting."""
    df = df[df["Season"] >= 2018].copy()
    df = df.dropna(subset=["Cap_Hit"])

    # Cap percentage
    df["Cap_Pct"] = df.apply(calculate_correct_pct, axis=1)

    # Convert TOI columns
    df["TOI/60"] = df["TOI/60"].apply(toi_to_float)
    df["PP TOI"] = df["PP TOI"].apply(toi_to_float)
    df["SH TOI"] = df["SH TOI"].apply(toi_to_float)
    df["TOI(EV)"] = df["TOI(EV)"].apply(toi_to_float)

    # Fill numeric nulls
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Faceoff filter
    df.loc[(df["FOW"] + df["FOL"]) < 20, "FO%"] = 0

    # Drop unnecessary columns
    df = df.drop(
        columns=[
            "Awards", "Logo_Ignore", "Age_at_Season",
            "name", "season", "team",
            "position_y", "Pos_y",
        ],
        errors="ignore",
    )

    df = df.rename(columns={"Pos_x": "Pos"})

    # Remove duplicate columns
    df = df.loc[:, ~df.T.duplicated()]

    # Deduplicate rows (keep most complete)
    df["_non_null_count"] = df.notna().sum(axis=1)
    df = df.sort_values(
        ["Player", "Season", "Team", "_non_null_count"],
        ascending=[True, True, True, False],
    )

    df = df.drop_duplicates(
        subset=["Player", "Season", "Team"],
        keep="first",
    ).drop(columns=["_non_null_count"])

    # Apply rename map
    df = apply_renaming_map(df)

    return df


def save_output(df: pd.DataFrame) -> None:
    """Save cleaned dataset to processed directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "cleaned_data.csv", index=False)
    print("Created cleaned_data.csv with renamed columns.")


def process_pipeline() -> None:
    """Run full data cleaning pipeline."""
    hr = load_hr_data()
    mp = load_mp_data()
    salaries = load_salary_data()

    merged = merge_datasets(hr, mp, salaries)
    merged = fix_scoring(merged)
    final_df = final_cleanup(merged)

    save_output(final_df)


if __name__ == "__main__":
    process_pipeline()