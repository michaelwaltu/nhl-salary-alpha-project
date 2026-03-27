from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


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