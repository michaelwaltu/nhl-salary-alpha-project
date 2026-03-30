from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler



INPUT_PATH = Path("data/processed/engineered_features.csv")
OUTPUT_DIR = Path("data/processed")

FORWARD_OUTPUT = OUTPUT_DIR / "forward_clusters.csv"
DEFENSE_OUTPUT = OUTPUT_DIR / "defense_clusters.csv"
FORWARD_SUMMARY_OUTPUT = OUTPUT_DIR / "forward_cluster_summary.csv"
DEFENSE_SUMMARY_OUTPUT = OUTPUT_DIR / "defense_cluster_summary.csv"


FORWARD_FEATURES = [
    "Age",
    "Avg TOI",
    "Ozone Start Pct",
    "Faceoff Pct",
    "Goals_per_60",
    "Assists_per_60",
    "Points_per_60",
    "Shots_on_Goal_per_60",
    "xGoals_per_60",
    "Hits_per_60",
    "Blocks_per_60",
    "Takeaways_pg",
    "Giveaways_pg",
    "Faceoff Impact_pg",
    "On Ice xGoals Diff_MP",
    "On Ice xGoals Ratio_MP",
    "Individual Zone Start Balance_MP",
    "Shooting Pct",
    "highDanger_Conversion",
]

DEFENSE_FEATURES = [
    "Age",
    "Avg TOI",
    "Ozone Start Pct",
    "Goals_per_60",
    "Assists_per_60",
    "Points_per_60",
    "Shots_on_Goal_per_60",
    "xGoals_per_60",
    "Hits_per_60",
    "Blocks_per_60",
    "Takeaways_pg",
    "Giveaways_pg",
    "On Ice xGoals Diff_MP",
    "On Ice xGoals Ratio_MP",
    "Individual Zone Start Balance_MP",
    "Shooting Pct",
    "highDanger_Conversion",
]

FORWARD_CLUSTER_LABELS = {
    0: "Offensive Specialist",
    1: "Primary Scorer",
    2: "Defensive / Depth Forward",
    3: "Elite Offensive Driver",
    4: "Playmaking / Transition Forward",
}

DEFENSE_CLUSTER_LABELS = {
    0: "Defensive / Two-Way Top-4 Defenseman",
    1: "Offensive Defenseman / PP QB",
    2: "Physical Shutdown Defenseman",
    3: "Depth Defenseman",
}


@dataclass
class ClusterConfig:
    min_gp: int = 10
    min_toi_fwd: float = 8.0
    min_toi_def: float = 10.0
    n_clusters_fwd: int = 5
    n_clusters_def: int = 4
    random_state: int = 42


def classify_position(pos: str) -> str:
    pos = str(pos).strip()

    if pos == "D":
        return "D"
    if pos == "C":
        return "F"
    if pos in {"LW", "RW", "W", "F", "LW/RW"}:
        return "F"
    return "EXCLUDE"


def split_position_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["Position Group"] = df["Position"].apply(classify_position)

    df_fwd = df[df["Position Group"] == "F"].copy()
    df_def = df[df["Position Group"] == "D"].copy()
    excluded = df[df["Position Group"] == "EXCLUDE"].copy()

    return df_fwd, df_def, excluded


def filter_clustering_population(
    df_fwd: pd.DataFrame,
    df_def: pd.DataFrame,
    config: ClusterConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_fwd_filt = df_fwd[
        (df_fwd["Games Played"] >= config.min_gp) &
        (df_fwd["Avg TOI"] >= config.min_toi_fwd)
    ].copy()

    df_def_filt = df_def[
        (df_def["Games Played"] >= config.min_gp) &
        (df_def["Avg TOI"] >= config.min_toi_def)
    ].copy()

    return df_fwd_filt, df_def_filt


def validate_feature_columns(df: pd.DataFrame, feature_cols: list[str], group_name: str) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {group_name} feature columns: {missing}")

    na_cols = df[feature_cols].isna().sum()
    na_cols = na_cols[na_cols > 0]
    if not na_cols.empty:
        raise ValueError(f"{group_name} feature set contains missing values:\n{na_cols}")


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str], group_name: str) -> pd.DataFrame:
    validate_feature_columns(df, feature_cols, group_name)
    return df[feature_cols].copy()


def scale_feature_matrix(X: pd.DataFrame) -> Tuple[StandardScaler, object]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def fit_gmm_and_assign_clusters(
    df: pd.DataFrame,
    X_scaled,
    n_clusters: int,
    cluster_labels: Dict[int, str],
    random_state: int,
) -> Tuple[pd.DataFrame, GaussianMixture]:
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    clusters = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    out = df.copy()
    out["Cluster"] = clusters
    out["Cluster Label"] = out["Cluster"].map(cluster_labels)

    for i in range(n_clusters):
        out[f"Cluster_{i}_prob"] = probs[:, i]

    out["Assigned_Cluster_Prob"] = probs.max(axis=1)

    return out, gmm


def summarize_clusters(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    return df.groupby(["Cluster", "Cluster Label"])[feature_cols].mean().sort_index()


def get_top_cluster_examples(
    df: pd.DataFrame,
    n_clusters: int,
    top_n: int = 5,
) -> Dict[int, pd.DataFrame]:
    examples = {}

    cols_to_show = ["Player", "Team", "Season"]

    for k in range(n_clusters):
        prob_col = f"Cluster_{k}_prob"
        examples[k] = (
            df.sort_values(prob_col, ascending=False)[cols_to_show + [prob_col]]
            .head(top_n)
            .reset_index(drop=True)
        )

    return examples


def clustering_pipeline():
    config = ClusterConfig()
    df = pd.read_csv(INPUT_PATH)
    df_fwd, df_def, excluded = split_position_groups(df)
    df_fwd_filt, df_def_filt = filter_clustering_population(df_fwd, df_def, config)
    X_fwd = build_feature_matrix(df_fwd_filt, FORWARD_FEATURES, "forward")
    X_def = build_feature_matrix(df_def_filt, DEFENSE_FEATURES, "defense")
    _, X_fwd_scaled = scale_feature_matrix(X_fwd)
    _, X_def_scaled = scale_feature_matrix(X_def)

    df_fwd_out, _ = fit_gmm_and_assign_clusters(
        df=df_fwd_filt,
        X_scaled=X_fwd_scaled,
        n_clusters=config.n_clusters_fwd,
        cluster_labels=FORWARD_CLUSTER_LABELS,
        random_state=config.random_state,
    )

    df_def_out, _ = fit_gmm_and_assign_clusters(
        df=df_def_filt,
        X_scaled=X_def_scaled,
        n_clusters=config.n_clusters_def,
        cluster_labels=DEFENSE_CLUSTER_LABELS,
        random_state=config.random_state,
    )

    fwd_summary = summarize_clusters(df_fwd_out, FORWARD_FEATURES)
    def_summary = summarize_clusters(df_def_out, DEFENSE_FEATURES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_fwd_out.to_csv(FORWARD_OUTPUT, index=False)
    df_def_out.to_csv(DEFENSE_OUTPUT, index=False)
    fwd_summary.to_csv(FORWARD_SUMMARY_OUTPUT)
    def_summary.to_csv(DEFENSE_SUMMARY_OUTPUT)

    print(f"Saved: {FORWARD_OUTPUT}")
    print(f"Saved: {DEFENSE_OUTPUT}")