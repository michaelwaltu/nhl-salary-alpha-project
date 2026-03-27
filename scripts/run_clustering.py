from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clustering import (
    ClusterConfig,
    DEFENSE_CLUSTER_LABELS,
    DEFENSE_FEATURES,
    FORWARD_CLUSTER_LABELS,
    FORWARD_FEATURES,
    build_feature_matrix,
    filter_clustering_population,
    fit_gmm_and_assign_clusters,
    get_top_cluster_examples,
    scale_feature_matrix,
    split_position_groups,
    summarize_clusters,
)


INPUT_PATH = Path("data/processed/engineered_features.csv")
OUTPUT_DIR = Path("data/processed")

FORWARD_OUTPUT = OUTPUT_DIR / "forward_clusters.csv"
DEFENSE_OUTPUT = OUTPUT_DIR / "defense_clusters.csv"
FORWARD_SUMMARY_OUTPUT = OUTPUT_DIR / "forward_cluster_summary.csv"
DEFENSE_SUMMARY_OUTPUT = OUTPUT_DIR / "defense_cluster_summary.csv"


def main() -> None:
    config = ClusterConfig()
    df = pd.read_csv(INPUT_PATH)
    df_fwd, df_def, excluded = split_position_groups(df)
    df_fwd_filt, df_def_filt = filter_clustering_population(df_fwd, df_def, config)
    X_fwd = build_feature_matrix(df_fwd_filt, FORWARD_FEATURES, "forward")
    X_def = build_feature_matrix(df_def_filt, DEFENSE_FEATURES, "defense")
    _, X_fwd_scaled = scale_feature_matrix(X_fwd)
    _, X_def_scaled = scale_feature_matrix(X_def)

    print("Fitting GMM...")
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

    print("\nDone.")
    print(f"Saved: {FORWARD_OUTPUT}")
    print(f"Saved: {DEFENSE_OUTPUT}")

if __name__ == "__main__":
    main()