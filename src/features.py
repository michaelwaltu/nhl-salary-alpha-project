import os
import numpy as np
import pandas as pd
from utils import toi_to_float

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


def drop_excluded_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        col for col in df.columns
        if col in DROP_COLUMNS or col.startswith(DROP_PREFIXES)
    ]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop, errors="ignore")
    return df

def prepare_modeling_features(
    input_path="data/processed/cleaned_data.csv",
    output_path="data/processed/modeling_ready_data.csv",
):
    os.makedirs("data/processed", exist_ok=True)

    print("Loading cleaned modeling data...")
    df = pd.read_csv(input_path)
    df = drop_excluded_columns(df)

    # FO_total
    df["FO_total"] = df[FOW_COL].fillna(0) + df[FOL_COL].fillna(0)

    # Per-game features
    # Avoid divide-by-zero
    gp_safe = df[GP_COL].replace(0, np.nan)

    for source_col in PER_GAME_COLS:
        df[f"{source_col}_pg"] = df[source_col] / gp_safe

    # Fill any new NaNs from GP=0 or missing source columns
    new_pg_cols = [c for c in df.columns if c.endswith("_pg")]
    df[new_pg_cols] = df[new_pg_cols].fillna(0)

    # Engineered archetype features
    if {"On Ice For xGoals_MP", "On Ice Against xGoals_MP"}.issubset(df.columns):
        # Net expected-goal impact while on ice (positive = better on-ice territorial outcomes)
        df["On Ice xGoals Diff_MP"] = df["On Ice For xGoals_MP"] - df["On Ice Against xGoals_MP"]
        # Expected-goal dominance ratio while on ice (stabilized denominator avoids divide-by-zero)
        df["On Ice xGoals Ratio_MP"] = df["On Ice For xGoals_MP"] / (df["On Ice Against xGoals_MP"] + 1e-6)

    if {"highDangerGoals", "highDangerxGoals"}.issubset(df.columns):
        # High-danger finishing efficiency relative to expected finishing quality
        df["highDanger Conversion"] = df["highDangerGoals"] / (df["highDangerxGoals"] + 1e-6)
    elif {"highDangerGoals", "highDangerxGoals_pg"}.issubset(df.columns):
        # Same efficiency using per-game xGoals when total highDangerxGoals is unavailable
        df["highDanger Conversion"] = df["highDangerGoals"] / (df["highDangerxGoals_pg"] + 1e-6)

    if {"Individual OZone Shift Starts_MP", "Individual DZone Shift Starts_MP"}.issubset(df.columns):
        # Deployment usage tilt (positive = more offensive-zone deployment)
        df["Individual Zone Start Balance_MP"] = (
            df["Individual OZone Shift Starts_MP"] - df["Individual DZone Shift Starts_MP"]
        )

    if {"FO_total_pg", "Faceoff Pct"}.issubset(df.columns):
        # Faceoff impact combining opportunity volume and win skill (auto-handles pct scale)
        faceoff_pct_scale = 100.0 if df["Faceoff Pct"].dropna().max() > 1.5 else 1.0
        df["Faceoff Impact_pg"] = df["FO_total_pg"] * (df["Faceoff Pct"] / faceoff_pct_scale)

    # GP filter from notebook workflow plus a very low-usage TOI screen
    total_toi_minutes = df[TOTAL_TOI_COL].apply(toi_to_float)
    df_model = df[(df[GP_COL] >= MIN_GP) & (total_toi_minutes >= MIN_TOTAL_TOI)].copy()

    # Prune high-collinearity feature families
    drop_cols = set()

    # 1) Keep one zone-start metric: keep Ozone Start Pct, drop Dzone Start Pct
    drop_cols.add("Dzone Start Pct")

    # 2) Keep TOI Per 60, drop TOI Even Strength
    drop_cols.add("TOI Even Strength")

    # Explicit removals requested
    drop_cols.update({"Cap Hit", "Individual Penalty Minutes_MP", "Penalty Minutes Drawn_MP"})

    on_ice_keep = {"On Ice For xGoals_MP", "On Ice Against xGoals_MP", "On Ice Goals Against_MP"}

    # Explicit additional removals requested
    drop_cols.update(
        {
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
        }
    )

    for col in df_model.columns:
        col_l = col.lower()

        # Drop Corsi family (keeping Fenwick-based possession features)
        if "corsi" in col_l:
            drop_cols.add(col)

        # 3) Remove raw shot decomposition branches
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

        # 4) Drop score/venue/rebound variant duplicates
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

        # 5) For On Ice For/Against keep only the compact core
        if col.startswith("On Ice For ") or col.startswith("On Ice Against "):
            if col not in on_ice_keep:
                drop_cols.add(col)

        # Keep Fenwick and xGoals percent; drop On-Ice Fenwick relative and xGoals pct duplicates
        if col in {"On Ice xGoals Pct_MP"}:
            drop_cols.add(col)

    # Remove flurry remnants
    drop_cols.update({c for c in df_model.columns if "flurry" in c.lower()})

    # Drop total/count columns when a per-game version exists
    for col in df_model.columns:
        if col.endswith("_pg"):
            base_col = col[:-3]
            if base_col in df_model.columns:
                drop_cols.add(base_col)

    df_model = df_model.drop(columns=sorted(drop_cols), errors="ignore")
    print(f"Dropped {len(drop_cols)} columns")


    df_model.to_csv(output_path, index=False)

    print("✅ DONE!")
    print(f"Saved filtered modeling dataset to: {output_path}")
    print(f"Rows kept with GP >= {MIN_GP} and total TOI >= {MIN_TOTAL_TOI}: {len(df_model):,}")
    print(f"Columns in output: {df_model.shape[1]}")


if __name__ == "__main__":
    prepare_modeling_features()
