from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import toi_to_float


MIN_GP = 10
MIN_TOTAL_TOI = 60

GP_COL = "Games Played"
FOW_COL = "Faceoffs Won"
FOL_COL = "Faceoffs Lost"
TOTAL_TOI_COL = "Total Ice Time"

DROP_COLUMNS = {
    "timeOnBench",
    "TK",
    "GV",
    "TAKE",
    "GIVE",
    "iceTimeRank",
    "Rk",
    "games_played",
    "icetime",
    "position",
    "faceoffsLost",
    "I_F_faceOffsWon",
    "OnIce_F_lowDangerShots",
    "OnIce_F_mediumDangerShots",
    "OnIce_F_highDangerShots",
    "OnIce_F_lowDangerxGoals",
    "OnIce_F_mediumDangerxGoals",
    "OnIce_F_highDangerxGoals",
    "OnIce_F_lowDangerGoals",
    "OnIce_F_mediumDangerGoals",
    "OnIce_F_highDangerGoals",
    "xGoalsForAfterShifts",
}

DROP_PREFIXES = ("OffIce_", "offIce_")

PER_GAME_COLS = [
    "Goals",
    "Assists",
    "Points",
    "Even Strength Goals",
    "Power Play Goals",
    "Individual xGoals_MP",
    "Individual Flurry Score Venue Adjusted xGoals_MP",
    "Primary Assists_MP",
    "Secondary Assists_MP",
    "Individual Shots On Goal_MP",
    "Individual Missed Shots_MP",
    "Individual Blocked Shot Attempts_MP",
    "highDangerShots",
    "mediumDangerShots",
    "lowDangerShots",
    "highDangerxGoals",
    "Individual xRebounds_MP",
    "Individual Hits_MP",
    "Takeaways",
    "Giveaways",
    "Individual DZone Giveaways_MP",
    "Shots Blocked By Player_MP",
    "Penalty Mins",
    "FO_total",
]

PER_60_FEATURE_SPECS = {
    "Goals": "Goals_per_60",
    "Assists": "Assists_per_60",
    "Points": "Points_per_60",
    "Individual Shots On Goal_MP": "Shots_on_Goal_per_60",
    "Individual xGoals_MP": "xGoals_per_60",
    "Individual Hits_MP": "Hits_per_60",
    "Shots Blocked By Player_MP": "Blocks_per_60",
}

# Notebook-driven drop decisions from feature validation / redundancy review
POST_VALIDATION_DROP_COLUMNS = {
    # first obvious pass
    "Points_per_60_alt",
    "Points_pg",
    "Goals_pg",
    "Assists_pg",
    "FO_total_pg",
    "Individual Hits_MP_pg",
    "Shots Blocked By Player_MP_pg",
    "On Ice Goals Against_MP",
    "Fenwick For",
    "Fenwick Against",

    # second pass
    "Individual Shots On Goal_MP_pg",
    "Individual Missed Shots_MP_pg",
    "lowDangerShots_pg",
    "mediumDangerShots_pg",
    "Individual xRebounds_MP_pg",
    "Individual xGoals_MP_pg",
    "Points_MP",
    "On Ice Rebounds For_MP",
    "On Ice Rebounds Against_MP",
    "lowDangerxGoals",
    "Goals_MP",
    "mediumDangerGoals",
    "Shifts",

    # final pass
    "On Ice Fenwick Pct_MP",
    "Individual Freeze_MP",
    "Even Strength Goals_pg",
    "Individual DZone Giveaways_MP_pg",
    "Fenwick For Pct",
    "Fenwick For Pct Relative",
}

PROCESSED_DIR = Path("data/processed")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series and fill divide-by-zero results with 0."""
    return numerator.div(denominator.replace(0, np.nan)).fillna(0)


def load_clean_data(input_path: str | Path) -> pd.DataFrame:
    """Load the cleaned player dataset."""
    return pd.read_csv(input_path)

def normalize_toi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TOI columns from MM:SS strings to numeric minutes."""
    df = df.copy()

    toi_cols = [
        "Avg TOI",
        "Total Ice Time",
        "Power Play Time On Ice",
        "Short Handed Time On Ice",
        "PP TOI",
        "SH TOI",
        "TOI(EV)",
    ]

    for col in toi_cols:
        if col in df.columns:
            df[col] = df[col].apply(toi_to_float)

    return df

def drop_excluded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns removed earlier in feature exploration."""
    cols_to_drop = [
        col for col in df.columns
        if col in DROP_COLUMNS or col.startswith(DROP_PREFIXES)
    ]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop, errors="ignore")
    return df


def add_faceoff_total(df: pd.DataFrame) -> pd.DataFrame:
    """Create total faceoff volume."""
    df = df.copy()
    if {FOW_COL, FOL_COL}.issubset(df.columns):
        df["FO_total"] = df[FOW_COL].fillna(0) + df[FOL_COL].fillna(0)
    return df


def add_per_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-game rate features."""
    df = df.copy()

    if GP_COL not in df.columns:
        return df

    gp_safe = df[GP_COL].replace(0, np.nan)

    for source_col in PER_GAME_COLS:
        if source_col in df.columns:
            df[f"{source_col}_pg"] = df[source_col] / gp_safe

    new_pg_cols = [c for c in df.columns if c.endswith("_pg")]
    if new_pg_cols:
        df[new_pg_cols] = df[new_pg_cols].fillna(0)

    return df


def add_per_60_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-60 features using season total TOI in minutes."""
    df = df.copy()

    if TOTAL_TOI_COL not in df.columns:
        return df

    total_toi_minutes = df[TOTAL_TOI_COL]

    for source_col, feature_name in PER_60_FEATURE_SPECS.items():
        if source_col in df.columns:
            df[feature_name] = safe_divide(df[source_col] * 60, total_toi_minutes)

    return df


def add_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create usage-share features from PP, SH, and EV ice time."""
    df = df.copy()

    if TOTAL_TOI_COL not in df.columns:
        return df

    total_toi_minutes = df[TOTAL_TOI_COL]

    usage_map = {
        "PP TOI": "PP_TOI_share",
        "SH TOI": "SH_TOI_share",
        "TOI(EV)": "EV_TOI_share",
    }

    for source_col, feature_name in usage_map.items():
        if source_col in df.columns:
            source_minutes = df[source_col]
            df[feature_name] = safe_divide(source_minutes, total_toi_minutes)

    return df


def add_archetype_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deployment and on-ice impact features."""
    df = df.copy()

    if {"On Ice For xGoals_MP", "On Ice Against xGoals_MP"}.issubset(df.columns):
        df["On Ice xGoals Diff_MP"] = (
            df["On Ice For xGoals_MP"] - df["On Ice Against xGoals_MP"]
        )
        df["On Ice xGoals Ratio_MP"] = safe_divide(
            df["On Ice For xGoals_MP"],
            df["On Ice Against xGoals_MP"],
        )

    if {"Individual OZone Shift Starts_MP", "Individual DZone Shift Starts_MP"}.issubset(df.columns):
        df["Individual Zone Start Balance_MP"] = (
            df["Individual OZone Shift Starts_MP"] - df["Individual DZone Shift Starts_MP"]
        )

    if {"FO_total_pg", "Faceoff Pct"}.issubset(df.columns):
        faceoff_pct_scale = 100.0 if df["Faceoff Pct"].dropna().max() > 1.5 else 1.0
        df["Faceoff Impact_pg"] = df["FO_total_pg"] * (df["Faceoff Pct"] / faceoff_pct_scale)

    return df


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create finishing and scoring efficiency features."""
    df = df.copy()

    if {"Goals", "Individual xGoals_MP"}.issubset(df.columns):
        df["Goals_minus_xGoals"] = df["Goals"] - df["Individual xGoals_MP"]

    if {"Goals", "Individual Shots On Goal_MP"}.issubset(df.columns):
        shots = df["Individual Shots On Goal_MP"]
        goals = df["Goals"]

        shooting_pct = safe_divide(goals, shots)

        # Keep only players with a reasonable shot sample
        shooting_pct = np.where(shots >= 20, shooting_pct, np.nan)

        # Hard cap to avoid impossible or noisy extremes
        shooting_pct = np.clip(shooting_pct, 0, 0.5)

        df["Shooting_Pct"] = shooting_pct
        df["Shooting_Pct_valid"] = (~df["Shooting_Pct"].isna()).astype(int)

    if {"highDangerGoals", "highDangerxGoals"}.issubset(df.columns):
        df["highDanger_Conversion"] = safe_divide(
            df["highDangerGoals"],
            df["highDangerxGoals"],
        )
    elif {"highDangerGoals", "highDangerxGoals_pg"}.issubset(df.columns):
        df["highDanger_Conversion"] = safe_divide(
            df["highDangerGoals"],
            df["highDangerxGoals_pg"],
        )

    if {"Points", TOTAL_TOI_COL}.issubset(df.columns):
        total_toi_minutes = df[TOTAL_TOI_COL]
        df["Points_per_60_alt"] = safe_divide(df["Points"] * 60, total_toi_minutes)

    return df


def add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create salary-efficiency and salary-normalized features."""
    df = df.copy()

    if {"Cap Hit", "Points"}.issubset(df.columns):
        df["CapHit_per_Point"] = safe_divide(df["Cap Hit"], df["Points"])

    if {"Cap Hit", "Individual xGoals_MP"}.issubset(df.columns):
        df["CapHit_per_xG"] = safe_divide(df["Cap Hit"], df["Individual xGoals_MP"])

    if {"Cap Hit", TOTAL_TOI_COL}.issubset(df.columns):
        total_toi_minutes = df[TOTAL_TOI_COL]
        df["CapHit_per_TOI"] = safe_divide(df["Cap Hit"], total_toi_minutes)

    if {"Points", "Cap Pct"}.issubset(df.columns):
        df["Points_per_CapPct"] = safe_divide(df["Points"], df["Cap Pct"])

    if {"Individual xGoals_MP", "Cap Pct"}.issubset(df.columns):
        df["xG_per_CapPct"] = safe_divide(df["Individual xGoals_MP"], df["Cap Pct"])

    return df


def apply_player_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to players with minimum games played and total ice time."""
    df = df.copy()


    if GP_COL not in df.columns or TOTAL_TOI_COL not in df.columns:
        return df

    total_toi_minutes = df[TOTAL_TOI_COL]
    df = df[(df[GP_COL] >= MIN_GP) & (total_toi_minutes >= MIN_TOTAL_TOI)].copy()

    return df


def prune_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop selected high-collinearity or low-priority columns."""
    df = df.copy()
    drop_cols = set()

    # Existing broad cleanup
    drop_cols.update({
        "Dzone Start Pct",
        "TOI Even Strength",
        "Cap Hit",
        "Individual Penalty Minutes_MP",
        "Penalty Minutes Drawn_MP",
        "Faceoffs Won",
        "Faceoffs Lost",
        "Total Shot Attempts",
        "Shots On Goal",
        "Individual Fly Shift Starts_MP",
        "Individual Fly Shift Ends_MP",
        "Individual Neutral Zone Shift Starts_MP",
        "Individual Neutral Zone Shift Ends_MP",
        "Individual Play Continued In Zone_MP",
        "Individual Play Continued Outside Zone_MP",
        "Individual Unblocked Shot Attempts_MP",
        "On Ice Goals For_MP",
        "Plus Minus",
        "EV Plus_Minus",
        "highDangerShots_pg",
    })

    on_ice_keep = {
        "On Ice For xGoals_MP",
        "On Ice Against xGoals_MP",
    }

    for col in df.columns:
        col_l = col.lower()

        if "corsi" in col_l:
            drop_cols.add(col)

        if any(
            token in col_l
            for token in [
                "xon goal",
                "xfreeze",
                "xplay stopped",
                "xplay continued",
                "saved shots",
                "saved unblocked",
            ]
        ):
            drop_cols.add(col)

        if any(
            token in col_l
            for token in [
                "score adjusted",
                "score venue adjusted",
                "xgoals from xrebounds",
                "xgoals from actual rebounds",
                "rebound xgoals",
                "xgoals with earned rebounds",
            ]
        ):
            drop_cols.add(col)

        if col.startswith("On Ice For ") or col.startswith("On Ice Against "):
            if col not in on_ice_keep:
                drop_cols.add(col)

        if col in {"On Ice xGoals Pct_MP"}:
            drop_cols.add(col)

    drop_cols.update({c for c in df.columns if "flurry" in c.lower()})

    # If a per-game feature exists, drop the raw counting version
    for col in df.columns:
        if col.endswith("_pg"):
            base_col = col[:-3]
            if base_col in df.columns:
                drop_cols.add(base_col)

    # Notebook-driven validated drops
    drop_cols.update(POST_VALIDATION_DROP_COLUMNS)

    df = df.drop(columns=sorted(drop_cols), errors="ignore")
    print(f"Dropped {len(drop_cols)} columns")

    return df


def save_output(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save engineered features to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_features_pipeline(
    input_path: str | Path = "data/processed/cleaned_data.csv",
    output_path: str | Path = "data/processed/engineered_features.csv",
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    print("Loading cleaned player data...")
    df = load_clean_data(input_path)
    df = normalize_toi_columns(df)
    df = drop_excluded_columns(df)
    df = add_faceoff_total(df)
    df = add_per_game_features(df)
    df = add_per_60_features(df)
    df = add_usage_features(df)
    df = add_archetype_features(df)
    df = add_efficiency_features(df)
    df = add_salary_features(df)
    df = apply_player_filters(df)
    df = prune_model_columns(df)

    save_output(df, output_path)

    print("✅ DONE!")
    print(f"Saved engineered feature dataset to: {output_path}")
    print(f"Rows kept with GP >= {MIN_GP} and total TOI >= {MIN_TOTAL_TOI}: {len(df):,}")
    print(f"Columns in output: {df.shape[1]}")

    return df


if __name__ == "__main__":
    build_features_pipeline()