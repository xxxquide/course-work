"""Module C: detect anomalous Telegram users and cluster bot candidates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config


LOGGER = logging.getLogger(__name__)
NO_CLUSTER_ID = -1


def load_features(input_path: Path = config.FEATURES_PATH) -> pd.DataFrame:
    """Load engineered features from CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    frame = pd.read_csv(input_path, dtype={"user_id": str})
    required = {"user_id", *config.FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature file is missing required columns: {sorted(missing)}")
    return frame


def preprocess_features(features: pd.DataFrame) -> np.ndarray:
    """Impute missing feature values and apply standard scaling."""
    matrix = features.loc[:, config.FEATURE_COLUMNS].to_numpy(dtype=float)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    return StandardScaler().fit_transform(imputed)


def run_isolation_forest(scaled_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run IsolationForest and return labels plus anomaly scores."""
    if scaled_features.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    labels = model.fit_predict(scaled_features)
    scores = model.decision_function(scaled_features)
    return labels, scores


def count_non_noise_clusters(labels: np.ndarray) -> int:
    """Count DBSCAN-style clusters while excluding noise label -1."""
    return len({int(label) for label in labels if int(label) != NO_CLUSTER_ID})


def robust_01(values: pd.Series, invert: bool = False) -> pd.Series:
    """Scale numeric values to 0..1 using 5th/95th percentiles."""
    series = pd.to_numeric(values, errors="coerce").astype(float)
    if series.dropna().empty:
        scaled = pd.Series(0.0, index=series.index)
    else:
        filled = series.fillna(series.median())
        low = float(filled.quantile(0.05))
        high = float(filled.quantile(0.95))
        if high <= low:
            scaled = pd.Series(0.0, index=series.index)
        else:
            scaled = ((filled - low) / (high - low)).clip(0, 1)
    return 1 - scaled if invert else scaled


def bot_neighbor_ratio(features: pd.DataFrame, is_bot: np.ndarray) -> pd.Series:
    """Return ratio of graph neighbours that are also bot candidates."""
    if not config.GRAPH_PATH.exists():
        return pd.Series(0.0, index=features.index)
    try:
        graph = nx.read_graphml(config.GRAPH_PATH)
    except Exception as exc:  # pragma: no cover - defensive for corrupt GraphML
        LOGGER.warning("Could not load graph for risk score: %s", exc)
        return pd.Series(0.0, index=features.index)
    bot_ids = set(features.loc[is_bot, "user_id"].astype(str))
    ratios: list[float] = []
    for user_id in features["user_id"].astype(str):
        if user_id not in graph:
            ratios.append(0.0)
            continue
        neighbours = list(graph.neighbors(user_id))
        if not neighbours:
            ratios.append(0.0)
            continue
        ratios.append(sum(1 for neighbour in neighbours if str(neighbour) in bot_ids) / len(neighbours))
    return pd.Series(ratios, index=features.index)


def build_risk_explanations(
    features: pd.DataFrame,
    components: dict[str, pd.Series],
    is_bot: np.ndarray,
) -> pd.Series:
    """Build concise Ukrainian explanation labels for each user's risk score."""
    explanations: list[str] = []
    for position, (_, row) in enumerate(features.iterrows()):
        reasons: list[str] = []
        if bool(is_bot[position]):
            reasons.append("аномальна поведінка")
        if components["fast_reaction"].iat[position] >= 0.65:
            reasons.append("швидка реакція")
        if float(row.get("duplicate_ratio", 0.0)) >= 0.35:
            reasons.append("повтори тексту")
        if components["graph"].iat[position] >= 0.65:
            reasons.append("багато графових зв'язків")
        if components["bot_neighbour"].iat[position] >= 0.25:
            reasons.append("зв'язки з бот-кандидатами")
        if components["category"].iat[position] >= 0.25:
            reasons.append("чутливі категорії")
        explanations.append(", ".join(reasons) if reasons else "низький ризик")
    return pd.Series(explanations, index=features.index)


def compute_risk_scores(features: pd.DataFrame, scores: np.ndarray, is_bot: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Compute an explainable 0..100 suspicion score from ML and evidence signals."""
    anomaly_rank = pd.Series(scores, index=features.index).rank(pct=True, ascending=True)
    anomaly_component = (1 - anomaly_rank).clip(0, 1)
    category_columns = [column for column in features.columns if column.startswith("category_") and column.endswith("_ratio")]
    category_component = features[category_columns].max(axis=1).fillna(0.0) if category_columns else pd.Series(0.0, index=features.index)
    components = {
        "anomaly": anomaly_component,
        "graph": robust_01(features["graph_degree"]),
        "duplicate": pd.to_numeric(features["duplicate_ratio"], errors="coerce").fillna(0.0).clip(0, 1),
        "fast_reaction": robust_01(features["reaction_speed_mean"], invert=True),
        "category": category_component.clip(0, 1),
        "bot_neighbour": bot_neighbor_ratio(features, is_bot).clip(0, 1),
    }
    risk = (
        0.40 * components["anomaly"]
        + 0.15 * components["graph"]
        + 0.10 * components["duplicate"]
        + 0.10 * components["fast_reaction"]
        + 0.10 * components["category"]
        + 0.15 * components["bot_neighbour"]
    ) * 100
    explanations = build_risk_explanations(features, components, is_bot)
    return risk.round(2), explanations


def cluster_bot_candidates(bot_features: np.ndarray) -> np.ndarray:
    """Cluster anomalous users with DBSCAN and KMeans fallback."""
    n_candidates = bot_features.shape[0]
    if n_candidates == 0:
        return np.array([], dtype=int)

    if n_candidates >= 3:
        dbscan_labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(bot_features)
        if count_non_noise_clusters(dbscan_labels) >= 2:
            return dbscan_labels.astype(int)
        LOGGER.info(
            "DBSCAN found fewer than 2 clusters on %s candidates; falling back to KMeans.",
            n_candidates,
        )
    else:
        LOGGER.info(
            "Only %s bot candidate(s); skipping DBSCAN and using KMeans fallback.",
            n_candidates,
        )

    n_clusters = min(5, n_candidates)
    try:
        return (
            KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            .fit_predict(bot_features)
            .astype(int)
        )
    except ValueError as exc:
        LOGGER.warning("KMeans fallback failed: %s. Saving results without cluster_id.", exc)
        return np.full(n_candidates, NO_CLUSTER_ID, dtype=int)


def detect_bot_farms() -> tuple[Path, dict[str, float]]:
    """Run anomaly detection, cluster bot candidates, and save ML results."""
    config.configure_logging()
    config.ensure_directories()
    features = load_features()
    scaled = preprocess_features(features)
    labels, scores = run_isolation_forest(scaled)
    is_bot = labels == -1
    cluster_ids = np.full(features.shape[0], NO_CLUSTER_ID, dtype=int)
    bot_indices = np.flatnonzero(is_bot)
    cluster_ids[bot_indices] = cluster_bot_candidates(scaled[bot_indices])
    risk_scores, risk_reasons = compute_risk_scores(features, scores, is_bot)

    results = pd.DataFrame(
        {
            "user_id": features["user_id"],
            "anomaly_score": scores,
            "risk_score": risk_scores,
            "risk_reasons": risk_reasons,
            "is_bot": is_bot.astype(int),
            "cluster_id": cluster_ids,
        }
    )
    config.ML_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.ML_RESULTS_PATH, index=False)

    total_users = int(features.shape[0])
    bot_count = int(is_bot.sum())
    bot_percentage = (bot_count / total_users * 100) if total_users else 0.0
    cluster_count = count_non_noise_clusters(cluster_ids[bot_indices])

    print(f"[C] Total users: {total_users}")
    print(f"[C] Bot candidates detected: {bot_count} ({bot_percentage:.1f}%)")
    print(f"[C] Bot farm clusters found: {cluster_count}")
    return config.ML_RESULTS_PATH, {
        "total_users": float(total_users),
        "bot_count": float(bot_count),
        "bot_percentage": bot_percentage,
        "cluster_count": float(cluster_count),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the Module C command-line parser."""
    parser = argparse.ArgumentParser(description="Detect bot candidates and bot-farm clusters.")
    return parser


def main() -> None:
    """Run ML detection from the command line."""
    build_parser().parse_args()
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    output_path, _ = detect_bot_farms()
    print(f"Module C output saved → {output_path}")


if __name__ == "__main__":
    main()
