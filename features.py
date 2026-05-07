"""Module B: build co-activity graph and per-user feature vectors."""

from __future__ import annotations

import argparse
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance

import categories
import config


LOGGER = logging.getLogger(__name__)


def load_comments(input_path: Path = config.RAW_COMMENTS_PATH) -> pd.DataFrame:
    """Load raw Telegram comments and normalize expected columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Raw comments file not found: {input_path}")
    frame = pd.read_csv(input_path, dtype={"user_id": str, "post_id": str, "reply_to_msg_id": str})
    required = {"user_id", "message_text", "timestamp", "post_id", "reply_to_msg_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Raw comments file is missing required columns: {sorted(missing)}")
    if "channel" not in frame.columns:
        LOGGER.warning("Raw comments file has no channel column; using a single unknown channel.")
        frame["channel"] = "unknown"
    if "post_timestamp" not in frame.columns:
        LOGGER.warning("Raw comments file has no post_timestamp column; reaction speeds will be missing.")
        frame["post_timestamp"] = pd.NaT
    if "category_tags" not in frame.columns:
        frame = categories.tag_comments(frame, mode="tag")
    if "category_match_terms" not in frame.columns:
        frame["category_match_terms"] = ""
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["post_timestamp"] = pd.to_datetime(frame["post_timestamp"], utc=True, errors="coerce")
    frame["message_text"] = frame["message_text"].fillna("").astype(str)
    frame["category_tags"] = frame["category_tags"].fillna("").astype(str)
    frame["category_match_terms"] = frame["category_match_terms"].fillna("").astype(str)
    frame = frame.dropna(subset=["user_id", "timestamp"])
    return frame


def build_coactivity_graph(
    comments: pd.DataFrame,
    window_seconds: int = config.CO_ACTIVITY_WINDOW_SECONDS,
) -> nx.Graph:
    """Build weighted user graph from same-post comments within a time window."""
    graph = nx.Graph()
    for user_id in comments["user_id"].dropna().astype(str).unique():
        graph.add_node(user_id)

    grouped = comments.sort_values("timestamp").groupby(["channel", "post_id"], dropna=False)
    for _, post_comments in grouped:
        rows = post_comments[["user_id", "timestamp"]].drop_duplicates().to_dict("records")
        for left_index, left in enumerate(rows):
            left_user = str(left["user_id"])
            left_time = left["timestamp"]
            for right in rows[left_index + 1 :]:
                right_time = right["timestamp"]
                delta = abs((right_time - left_time).total_seconds())
                if delta > window_seconds:
                    break
                right_user = str(right["user_id"])
                if left_user == right_user:
                    continue
                if graph.has_edge(left_user, right_user):
                    graph[left_user][right_user]["weight"] += 1
                else:
                    graph.add_edge(left_user, right_user, weight=1)
    return graph


def compute_reaction_speeds(comments: pd.DataFrame) -> pd.Series:
    """Compute seconds between post publish time and each comment."""
    speeds = (comments["timestamp"] - comments["post_timestamp"]).dt.total_seconds()
    return speeds.where(speeds >= 0)


def compute_duplicate_ratio(messages: list[str]) -> float:
    """Compute fraction of exact or near-exact duplicate messages for one user."""
    if len(messages) <= 1:
        return 0.0
    duplicate_indices: set[int] = set()
    normalized = [message.strip().casefold() for message in messages]
    for left_index, right_index in itertools.combinations(range(len(normalized)), 2):
        if left_index in duplicate_indices and right_index in duplicate_indices:
            continue
        if levenshtein_distance(normalized[left_index], normalized[right_index]) < 5:
            duplicate_indices.add(left_index)
            duplicate_indices.add(right_index)
    return len(duplicate_indices) / len(messages)


def observation_period_days(comments: pd.DataFrame) -> float:
    """Return the global observation period in days with a one-day lower bound."""
    if comments.empty:
        return 1.0
    total_seconds = (comments["timestamp"].max() - comments["timestamp"].min()).total_seconds()
    return max(total_seconds / 86_400, 1.0)


def category_counts(user_comments: pd.DataFrame) -> dict[str, int]:
    """Count category-tagged comments for one user."""
    counts = {category: 0 for category in categories.CATEGORY_PATTERNS}
    for raw_tags in user_comments["category_tags"].fillna("").astype(str):
        tags = {tag for tag in raw_tags.split(",") if tag}
        for tag in tags:
            if tag in counts:
                counts[tag] += 1
    return counts


def compute_features(comments: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    """Compute the required seven-feature matrix for every observed user."""
    comments = comments.copy()
    comments["reaction_speed_seconds"] = compute_reaction_speeds(comments)
    period_days = observation_period_days(comments)
    clustering = nx.clustering(graph, weight="weight")
    rows: list[dict[str, object]] = []

    for user_id, user_comments in comments.groupby("user_id"):
        reaction_speeds = user_comments["reaction_speed_seconds"].dropna()
        messages = user_comments["message_text"].tolist()
        counts = category_counts(user_comments)
        comment_total = max(len(user_comments), 1)
        rows.append(
            {
                "user_id": str(user_id),
                "reaction_speed_mean": reaction_speeds.mean() if not reaction_speeds.empty else np.nan,
                "reaction_speed_std": reaction_speeds.std(ddof=0) if len(reaction_speeds) > 1 else 0.0,
                "comment_count_per_day": len(user_comments) / period_days,
                "duplicate_ratio": compute_duplicate_ratio(messages),
                "graph_degree": graph.degree(str(user_id)),
                "clustering_coefficient": clustering.get(str(user_id), 0.0),
                "unique_channels": user_comments["channel"].nunique(),
                "category_diversity": sum(1 for value in counts.values() if value > 0),
                **{
                    f"category_{category}_ratio": counts[category] / comment_total
                    for category in categories.CATEGORY_PATTERNS
                },
            }
        )

    return pd.DataFrame(rows, columns=("user_id", *config.FEATURE_COLUMNS))


def save_graph(graph: nx.Graph, output_path: Path = config.GRAPH_PATH) -> Path:
    """Save the co-activity graph in GraphML format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
    return output_path


def build_features() -> tuple[Path, Path]:
    """Run graph construction and feature engineering."""
    config.configure_logging()
    config.ensure_directories()
    comments = load_comments()
    graph = build_coactivity_graph(comments)
    features = compute_features(comments, graph)
    config.FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.FEATURES_PATH, index=False)
    save_graph(graph)
    print(f"[B] Feature matrix shape: {features.shape[0]} users × {len(config.FEATURE_COLUMNS)} features")
    return config.FEATURES_PATH, config.GRAPH_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the Module B command-line parser."""
    parser = argparse.ArgumentParser(description="Build co-activity graph and user features.")
    return parser


def main() -> None:
    """Run feature engineering from the command line."""
    build_parser().parse_args()
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    features_path, graph_path = build_features()
    print(f"Module B outputs saved → {features_path}, {graph_path}")


if __name__ == "__main__":
    main()
