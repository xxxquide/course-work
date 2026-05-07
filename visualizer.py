"""Module D: render bot-farm detection results as an interactive PyVis dashboard."""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

import categories
import config


LOGGER = logging.getLogger(__name__)
NO_CLUSTER_ID = -1
CLUSTER_PALETTE = (
    "#ef4444",
    "#8b5cf6",
    "#0ea5e9",
    "#22c55e",
    "#f59e0b",
    "#ec4899",
    "#14b8a6",
    "#6366f1",
)
NORMAL_NODE_COLOR = "rgba(148, 163, 184, 0.46)"
BOT_CANDIDATE_COLOR = "#f97316"
GRAPH_BACKGROUND = "#0b1020"

UKRAINIAN_COLUMNS = {
    "cluster_id": "Кластер",
    "bot_count": "К-сть бот-кандидатів",
    "mean_anomaly_score": "Середній anomaly score",
    "mean_risk_score": "Середній рівень підозрілості",
    "min_anomaly_score": "Найнижчий anomaly score",
    "mean_graph_degree": "Середній ступінь у графі",
    "max_graph_degree": "Макс. ступінь у графі",
    "user_ids": "ID користувачів",
    "user_id": "ID користувача",
    "anomaly_score": "Anomaly score",
    "risk_score": "Рівень підозрілості",
    "risk_reasons": "Пояснення ризику",
    "comment_count_per_day": "Коментарів на день",
    "duplicate_ratio": "Частка повторів",
    "graph_degree_actual": "Зв'язків у графі",
    "clustering_coefficient": "Коефіцієнт кластеризації",
    "unique_channels": "Каналів",
}


def load_graph(input_path: Path = config.GRAPH_PATH) -> nx.Graph:
    """Load the co-activity graph from GraphML."""
    if not input_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_path}")
    return nx.read_graphml(input_path)


def load_results(input_path: Path = config.ML_RESULTS_PATH) -> pd.DataFrame:
    """Load ML bot-detection results from CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"ML results file not found: {input_path}")
    frame = pd.read_csv(input_path, dtype={"user_id": str})
    required = {"user_id", "anomaly_score", "is_bot", "cluster_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ML results file is missing required columns: {sorted(missing)}")
    return frame


def load_features(input_path: Path = config.FEATURES_PATH) -> pd.DataFrame:
    """Load feature rows for report tables."""
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    frame = pd.read_csv(input_path, dtype={"user_id": str})
    required = {"user_id", *config.FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature file is missing required columns: {sorted(missing)}")
    return frame


def load_comments(input_path: Path = config.RAW_COMMENTS_PATH) -> pd.DataFrame:
    """Load raw comments for user profile panels."""
    if not input_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    if "category_tags" not in frame.columns:
        frame = categories.tag_comments(frame, mode="tag")
    return frame


def result_lookup(results: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Convert ML results to a lookup keyed by user ID."""
    return results.set_index("user_id").to_dict(orient="index")


def cluster_color(cluster_id: int) -> str:
    """Return a stable color for a bot-farm cluster."""
    if cluster_id == NO_CLUSTER_ID:
        return BOT_CANDIDATE_COLOR
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


def node_style(
    user_id: str,
    lookup: dict[str, dict[str, object]],
    profiles: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return PyVis style attributes for a graph node."""
    result = lookup.get(user_id, {})
    is_bot = int(result.get("is_bot", 0)) == 1
    cluster_id = int(result.get("cluster_id", NO_CLUSTER_ID))
    risk_score = float(result.get("risk_score", 0.0))
    profile = (profiles or {}).get(user_id, {})
    top_category = str(profile.get("top_category_label") or "немає")
    title = (
        f"ID: {user_id}\n"
        f"Рівень підозрілості: {risk_score:.0f}/100\n"
        f"Кластер: {cluster_id if cluster_id != NO_CLUSTER_ID else 'немає'}\n"
        f"Топ категорія: {top_category}"
    )

    if is_bot and cluster_id != NO_CLUSTER_ID:
        color = cluster_color(cluster_id)
        return {
            "label": f"{user_id}\nкластер {cluster_id}",
            "color": {"background": color, "border": "#fef2f2", "highlight": {"background": color, "border": "#ffffff"}},
            "size": 20 + min(risk_score / 10, 10),
            "borderWidth": 3,
            "font": {"color": "#f8fafc", "size": 15, "face": "Arial"},
            "title": title,
        }
    if is_bot:
        return {
            "label": user_id,
            "color": {"background": BOT_CANDIDATE_COLOR, "border": "#ffedd5"},
            "size": 16 + min(risk_score / 12, 8),
            "borderWidth": 2,
            "font": {"color": "#f8fafc", "size": 13, "face": "Arial"},
            "title": title,
        }
    return {
        "label": "",
        "color": NORMAL_NODE_COLOR,
        "size": 5,
        "title": title,
    }


def is_bot_node(user_id: str, lookup: dict[str, dict[str, object]]) -> bool:
    """Return whether a node is an ML bot candidate."""
    return int(lookup.get(user_id, {}).get("is_bot", 0)) == 1


def node_cluster_id(user_id: str, lookup: dict[str, dict[str, object]]) -> int:
    """Return cluster ID for a node, or -1 when absent/noise."""
    return int(lookup.get(user_id, {}).get("cluster_id", NO_CLUSTER_ID))


def edge_weight(graph: nx.Graph, source: str, target: str) -> float:
    """Return numeric edge weight with a safe default."""
    return float(graph[source][target].get("weight", 1))


def strongest_neighbors(
    graph: nx.Graph,
    node: str,
    min_edge_weight: int,
    limit: int,
) -> list[str]:
    """Return strongest neighbours for a node, sorted by co-activity weight."""
    neighbours = [
        (neighbor, edge_weight(graph, node, neighbor), graph.degree(neighbor))
        for neighbor in graph.neighbors(node)
        if edge_weight(graph, node, neighbor) >= min_edge_weight
    ]
    neighbours.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [neighbor for neighbor, _, _ in neighbours[:limit]]


def filter_graph_for_viz(
    graph: nx.Graph,
    lookup: dict[str, dict[str, object]],
    min_edge_weight: int = config.VIZ_MIN_EDGE_WEIGHT,
    max_nodes: int = config.VIZ_MAX_NODES,
) -> nx.Graph:
    """Return a browser-friendly investigation graph.

    The full co-activity graph is useful as data, but it is too dense for a
    readable browser view. The dashboard therefore keeps all bot candidates,
    adds only their strongest neighbours, and drops normal-normal edges.
    """
    bot_nodes = {str(n) for n in graph.nodes if is_bot_node(str(n), lookup)}
    keep = set(bot_nodes)
    for bot in bot_nodes:
        keep.update(
            strongest_neighbors(
                graph,
                bot,
                min_edge_weight=min_edge_weight,
                limit=config.VIZ_NEIGHBORS_PER_BOT,
            )
        )

    if len(keep) > max_nodes:
        normal_nodes = keep - bot_nodes
        ranked_normals = sorted(
            normal_nodes,
            key=lambda node: (
                -max(edge_weight(graph, node, bot) for bot in graph.neighbors(node) if bot in bot_nodes),
                -graph.degree(node),
                node,
            ),
        )
        keep = bot_nodes | set(ranked_normals[: max(0, max_nodes - len(bot_nodes))])
        LOGGER.warning("Dashboard capped at %s nodes for browser performance.", len(keep))

    edge_candidates: list[tuple[str, str, float]] = []
    for source, target, attributes in graph.edges(data=True):
        source = str(source)
        target = str(target)
        if source not in keep or target not in keep:
            continue
        source_is_bot = source in bot_nodes
        target_is_bot = target in bot_nodes
        if not source_is_bot and not target_is_bot:
            continue
        weight = float(attributes.get("weight", 1))
        source_cluster = node_cluster_id(source, lookup)
        target_cluster = node_cluster_id(target, lookup)
        same_cluster = (
            source_cluster != NO_CLUSTER_ID
            and source_cluster == target_cluster
        )
        if weight >= min_edge_weight or source_is_bot and target_is_bot or same_cluster:
            edge_candidates.append((source, target, weight))

    edge_candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    limited_edges = edge_candidates[: config.VIZ_MAX_EDGES]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(keep)
    for source, target, _ in limited_edges:
        subgraph.add_edge(source, target, **graph[source][target])

    isolated_normals = [n for n in subgraph.nodes if subgraph.degree(n) == 0 and n not in bot_nodes]
    subgraph.remove_nodes_from(isolated_normals)
    return subgraph


def physics_options() -> str:
    """Return JSON physics options that stabilise quickly and stop simulating."""
    return json.dumps({
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -18000,
                "centralGravity": 0.18,
                "springLength": 170,
                "springConstant": 0.025,
                "damping": 0.72,
                "avoidOverlap": 0.55,
            },
            "stabilization": {
                "enabled": True,
                "iterations": config.VIZ_STABILIZATION_ITERATIONS,
                "fit": True,
            },
            "minVelocity": 0.75,
            "timestep": 0.5,
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": True,
            "hideEdgesOnDrag": True,
            "multiselect": True,
        },
        "edges": {
            "smooth": False,
            "selectionWidth": 2,
            "hoverWidth": 1.5,
        },
        "nodes": {
            "shape": "dot",
            "font": {"size": 12, "face": "Arial"},
        },
    })


def edge_style(source: str, target: str, weight: float, lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    """Return visual edge style based on bot/cluster relationship."""
    source_cluster = node_cluster_id(source, lookup)
    target_cluster = node_cluster_id(target, lookup)
    same_cluster = source_cluster != NO_CLUSTER_ID and source_cluster == target_cluster
    if same_cluster:
        color = cluster_color(source_cluster)
        return {
            "color": {"color": color, "highlight": "#ffffff", "hover": color},
            "width": min(1.5 + weight, 6),
            "value": min(weight, 6),
            "title": f"Спільна активність: {weight:g}<br>Один кластер: {source_cluster}",
        }
    if is_bot_node(source, lookup) and is_bot_node(target, lookup):
        return {
            "color": {"color": "rgba(248, 113, 113, 0.75)", "highlight": "#fecaca", "hover": "#fecaca"},
            "width": min(1 + weight, 5),
            "value": min(weight, 5),
            "title": f"Спільна активність між бот-кандидатами: {weight:g}",
        }
    return {
        "color": {"color": "rgba(148, 163, 184, 0.34)", "highlight": "#cbd5e1", "hover": "#cbd5e1"},
        "width": min(0.5 + weight, 4),
        "value": min(weight, 4),
        "title": f"Спільна активність: {weight:g}",
    }


def split_tags(raw_tags: object) -> list[str]:
    """Split comma-separated category tags."""
    return [tag for tag in str(raw_tags or "").split(",") if tag]


def top_category_label(counts: dict[str, int]) -> str:
    """Return the most frequent category label for a user."""
    if not counts:
        return ""
    category, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    if count <= 0:
        return ""
    return categories.CATEGORY_LABELS_UK.get(category, category)


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert value to float with a default."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = NO_CLUSTER_ID) -> int:
    """Convert value to int with a default."""
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_user_profiles(
    visible_graph: nx.Graph,
    full_graph: nx.Graph,
    results: pd.DataFrame,
    candidate_table: pd.DataFrame,
    comments: pd.DataFrame,
    message_limit: int = 20,
) -> dict[str, dict[str, object]]:
    """Build compact user profiles for graph click panels."""
    lookup = result_lookup(results)
    table_lookup = candidate_table.set_index("user_id").to_dict(orient="index")
    visible_users = {str(node) for node in visible_graph.nodes}
    visible_users.update(results.loc[results["is_bot"] == 1, "user_id"].astype(str))
    comments_by_user = {
        str(user_id): group.sort_values("timestamp", ascending=False)
        for user_id, group in comments.groupby("user_id")
    } if not comments.empty and "user_id" in comments.columns else {}

    profiles: dict[str, dict[str, object]] = {}
    for user_id in sorted(visible_users):
        result = lookup.get(user_id, {})
        row = table_lookup.get(user_id, {})
        user_comments = comments_by_user.get(user_id, pd.DataFrame())
        channel_counts = (
            user_comments["channel"].value_counts().head(8).to_dict()
            if not user_comments.empty and "channel" in user_comments.columns
            else {}
        )
        category_counts = {category: 0 for category in categories.CATEGORY_PATTERNS}
        if not user_comments.empty and "category_tags" in user_comments.columns:
            for raw_tags in user_comments["category_tags"]:
                for tag in split_tags(raw_tags):
                    if tag in category_counts:
                        category_counts[tag] += 1
        max_weight = max(
            (edge_weight(full_graph, user_id, neighbour) for neighbour in full_graph.neighbors(user_id)),
            default=1.0,
        ) if user_id in full_graph else 1.0
        cluster_id = safe_int(result.get("cluster_id"))
        top_connections: list[dict[str, object]] = []
        if user_id in full_graph:
            for neighbour in sorted(full_graph.neighbors(user_id), key=lambda n: -edge_weight(full_graph, user_id, n))[:12]:
                weight = edge_weight(full_graph, user_id, neighbour)
                neighbour_result = lookup.get(str(neighbour), {})
                neighbour_cluster = safe_int(neighbour_result.get("cluster_id"))
                relationship_score = min(
                    100.0,
                    (weight / max_weight * 70.0)
                    + (20.0 if int(neighbour_result.get("is_bot", 0)) == 1 else 0.0)
                    + (10.0 if cluster_id != NO_CLUSTER_ID and cluster_id == neighbour_cluster else 0.0),
                )
                top_connections.append(
                    {
                        "user_id": str(neighbour),
                        "shared_coactivity_count": int(weight),
                        "relationship_score": round(relationship_score, 1),
                        "is_bot": int(neighbour_result.get("is_bot", 0)),
                        "cluster_id": neighbour_cluster,
                    }
                )
        messages: list[dict[str, object]] = []
        if not user_comments.empty:
            for _, comment in user_comments.head(message_limit).iterrows():
                messages.append(
                    {
                        "timestamp": str(comment.get("timestamp", "")),
                        "channel": str(comment.get("channel", "")),
                        "post_id": str(comment.get("post_id", "")),
                        "text": str(comment.get("message_text", ""))[:420],
                        "category_tags": split_tags(comment.get("category_tags", "")),
                    }
                )
        profiles[user_id] = {
            "user_id": user_id,
            "is_bot": int(result.get("is_bot", 0)),
            "cluster_id": cluster_id,
            "anomaly_score": round(safe_float(result.get("anomaly_score")), 4),
            "risk_score": round(safe_float(result.get("risk_score")), 2),
            "risk_reasons": str(result.get("risk_reasons", "")),
            "top_category_label": top_category_label(category_counts),
            "features": {
                "comment_count_per_day": round(safe_float(row.get("comment_count_per_day")), 3),
                "duplicate_ratio": round(safe_float(row.get("duplicate_ratio")), 3),
                "graph_degree": int(safe_float(row.get("graph_degree_actual", row.get("graph_degree", 0)))),
                "clustering_coefficient": round(safe_float(row.get("clustering_coefficient")), 4),
                "unique_channels": int(safe_float(row.get("unique_channels"))),
                "category_diversity": int(safe_float(row.get("category_diversity"))),
            },
            "channels": {str(key): int(value) for key, value in channel_counts.items()},
            "categories": {
                categories.CATEGORY_LABELS_UK.get(key, key): int(value)
                for key, value in category_counts.items()
                if value > 0
            },
            "messages": messages,
            "connections": top_connections,
        }
    return profiles


def build_network(
    graph: nx.Graph,
    results: pd.DataFrame,
    profiles: dict[str, dict[str, object]] | None = None,
) -> Network:
    """Build a PyVis network with bot-specific node and edge styling."""
    network = Network(height="920px", width="100%", bgcolor=GRAPH_BACKGROUND, font_color="#f8fafc")
    network.set_options(physics_options())
    lookup = result_lookup(results)

    for user_id in graph.nodes:
        network.add_node(str(user_id), **node_style(str(user_id), lookup, profiles))

    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes.get("weight", 1))
        style = edge_style(str(source), str(target), weight, lookup)
        network.add_edge(
            str(source),
            str(target),
            **style,
        )
    return network


def legend_html(
    node_count: int,
    edge_count: int,
    total_nodes: int,
    total_edges: int,
    report_href: str = "analysis_report.html",
) -> str:
    """Return the HTML legend overlay inserted into the dashboard."""
    cluster_items = "".join(
        f'<span style="color:{color};">●</span> Кластер {index}<br>'
        for index, color in enumerate(CLUSTER_PALETTE[:4])
    )
    return f"""
<div style="
  position: fixed;
  top: 18px;
  right: 18px;
  z-index: 9999;
  max-width: 360px;
  padding: 16px 18px;
  color: #f9fafb;
  background: rgba(15, 23, 42, 0.94);
  border: 1px solid rgba(249, 250, 251, 0.18);
  border-radius: 8px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.32);
  font-family: Arial, sans-serif;
  font-size: 14px;
  line-height: 1.45;
">
  <strong style="font-size:16px;">Виявлення ботоферм</strong><br>
  <span style="color:#94a3b8;">●</span> Звичайний користувач<br>
  <span style="color:{BOT_CANDIDATE_COLOR};">●</span> Окремий бот-кандидат<br>
  {cluster_items}
  <small>Показано бот-кандидатів та їхні найсильніші зв'язки.</small><br>
  <small>Товщина ребра = частота спільної активності.</small><br>
  <small>На екрані: {node_count} / {total_nodes} вузлів, {edge_count} / {total_edges} зв'язків.</small><br>
  <small><a href="{html.escape(report_href)}" style="color:#93c5fd;">Відкрити аналітичний звіт</a></small>
</div>
"""


def panel_assets(profiles: dict[str, dict[str, object]] | None = None) -> str:
    """Return HTML/CSS/JS for the graph user detail panel."""
    profiles_json = json.dumps(profiles or {}, ensure_ascii=False).replace("</", "<\\/")
    return f"""
<style>
  #userProfilePanel {{
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: 410px;
    max-width: min(410px, 92vw);
    z-index: 10000;
    transform: translateX(102%);
    transition: transform 180ms ease;
    color: #0f172a;
    background: #f8fafc;
    border-left: 1px solid #cbd5e1;
    box-shadow: -24px 0 60px rgba(2, 6, 23, 0.35);
    font-family: Arial, sans-serif;
    overflow-y: auto;
  }}
  #userProfilePanel.open {{ transform: translateX(0); }}
  .profile-header {{ padding: 18px; color: #f8fafc; background: #0b1020; }}
  .profile-header button {{ float: right; border: 0; color: #f8fafc; background: transparent; font-size: 22px; cursor: pointer; }}
  .profile-body {{ padding: 16px; }}
  .profile-stat-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 12px 0; }}
  .profile-stat {{ padding: 10px; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; }}
  .profile-stat strong {{ display: block; font-size: 18px; }}
  .profile-section {{ margin: 18px 0; }}
  .profile-section h3 {{ margin: 0 0 8px; font-size: 15px; }}
  .bar-row {{ display: grid; grid-template-columns: 130px 1fr 36px; gap: 8px; align-items: center; margin: 7px 0; font-size: 12px; }}
  .bar-track {{ height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: #2563eb; }}
  .message-card {{ padding: 10px; margin: 8px 0; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; }}
  .message-meta {{ color: #64748b; font-size: 12px; margin-bottom: 5px; }}
  .pill {{ display: inline-block; margin: 2px 4px 2px 0; padding: 2px 7px; border-radius: 999px; background: #e0e7ff; color: #3730a3; font-size: 11px; }}
</style>
<aside id="userProfilePanel" aria-live="polite">
  <div class="profile-header">
    <button type="button" id="closeProfilePanel" aria-label="Закрити">×</button>
    <div id="profileTitle">Оберіть користувача</div>
    <small>Клікніть на вузол графа, щоб побачити деталі.</small>
  </div>
  <div class="profile-body" id="profileBody"></div>
</aside>
<script>
  const USER_PROFILES = {profiles_json};
  function escapeHtml(value) {{
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }}[char]));
  }}
  function bars(title, data) {{
    const entries = Object.entries(data || {{}}).slice(0, 8);
    if (!entries.length) return `<div class="profile-section"><h3>${{title}}</h3><p>Немає даних.</p></div>`;
    const maxValue = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
    return `<div class="profile-section"><h3>${{title}}</h3>` + entries.map(([label, value]) => {{
      const width = Math.round((Number(value) || 0) / maxValue * 100);
      return `<div class="bar-row"><span>${{escapeHtml(label)}}</span><div class="bar-track"><div class="bar-fill" style="width:${{width}}%"></div></div><span>${{value}}</span></div>`;
    }}).join("") + `</div>`;
  }}
  function renderProfile(userId) {{
    const profile = USER_PROFILES[String(userId)];
    const panel = document.getElementById("userProfilePanel");
    const title = document.getElementById("profileTitle");
    const body = document.getElementById("profileBody");
    if (!profile) {{
      title.textContent = `Користувач ${{userId}}`;
      body.innerHTML = "<p>Для цього вузла немає розширеного профілю.</p>";
      panel.classList.add("open");
      return;
    }}
    title.textContent = `Користувач ${{profile.user_id}}`;
    const status = profile.is_bot ? "Бот-кандидат" : "Звичайний користувач";
    const cluster = profile.cluster_id === -1 ? "немає" : profile.cluster_id;
    const messages = (profile.messages || []).map((message) => {{
      const tags = (message.category_tags || []).map((tag) => `<span class="pill">${{escapeHtml(tag)}}</span>`).join("");
      return `<div class="message-card"><div class="message-meta">@${{escapeHtml(message.channel)}} · post ${{escapeHtml(message.post_id)}} · ${{escapeHtml(message.timestamp)}}</div><div>${{escapeHtml(message.text)}}</div><div>${{tags}}</div></div>`;
    }}).join("") || "<p>Повідомлення не знайдені.</p>";
    const connections = (profile.connections || []).map((item) =>
      `<div class="message-card"><strong>${{escapeHtml(item.user_id)}}</strong><br>Сила зв'язку: ${{item.relationship_score}}/100 · спільна активність: ${{item.shared_coactivity_count}} · кластер: ${{item.cluster_id === -1 ? "немає" : item.cluster_id}}</div>`
    ).join("") || "<p>Сильні зв'язки не знайдені.</p>";
    body.innerHTML = `
      <div class="profile-stat-grid">
        <div class="profile-stat"><strong>${{profile.risk_score}}/100</strong>Рівень підозрілості</div>
        <div class="profile-stat"><strong>${{status}}</strong>Статус</div>
        <div class="profile-stat"><strong>${{cluster}}</strong>Кластер</div>
        <div class="profile-stat"><strong>${{profile.anomaly_score}}</strong>Anomaly score</div>
      </div>
      <div class="profile-section"><h3>Чому підозрілий</h3><p>${{escapeHtml(profile.risk_reasons || "низький ризик")}}</p></div>
      <div class="profile-stat-grid">
        <div class="profile-stat"><strong>${{profile.features.comment_count_per_day}}</strong>Коментарів на день</div>
        <div class="profile-stat"><strong>${{profile.features.duplicate_ratio}}</strong>Частка повторів</div>
        <div class="profile-stat"><strong>${{profile.features.graph_degree}}</strong>Зв'язків у графі</div>
        <div class="profile-stat"><strong>${{profile.features.unique_channels}}</strong>Каналів</div>
      </div>
      ${{bars("Активність за каналами", profile.channels)}}
      ${{bars("Категорії повідомлень", profile.categories)}}
      <div class="profile-section"><h3>Найсильніші зв'язки</h3>${{connections}}</div>
      <div class="profile-section"><h3>Останні повідомлення</h3>${{messages}}</div>
    `;
    panel.classList.add("open");
  }}
  document.getElementById("closeProfilePanel").addEventListener("click", () => {{
    document.getElementById("userProfilePanel").classList.remove("open");
  }});
  if (typeof network !== "undefined") {{
    network.once("stabilizationIterationsDone", function () {{
      network.setOptions({{ physics: false }});
    }});
    network.on("click", function (params) {{
      if (params.nodes && params.nodes.length) renderProfile(params.nodes[0]);
    }});
  }}
</script>
"""


def inject_legend(
    html_path: Path,
    legend: str,
    profiles: dict[str, dict[str, object]] | None = None,
) -> None:
    """Insert the legend and profile panel into the generated PyVis HTML file."""
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<body>", f"<body>\n{legend}", 1)
    html = html.replace("</body>", f"{panel_assets(profiles)}\n</body>", 1)
    html_path.write_text(html, encoding="utf-8")


def build_candidate_table(graph: nx.Graph, results: pd.DataFrame) -> pd.DataFrame:
    """Merge ML results with features and graph context for review."""
    features = load_features()
    results = results.copy()
    if "risk_score" not in results.columns:
        results["risk_score"] = 0.0
    if "risk_reasons" not in results.columns:
        results["risk_reasons"] = ""
    table = results.merge(features, on="user_id", how="left")
    table["graph_degree_actual"] = table["user_id"].map(lambda user_id: graph.degree(str(user_id)) if str(user_id) in graph else 0)
    table = table.sort_values(["is_bot", "cluster_id", "anomaly_score"], ascending=[False, True, True])
    return table


def save_candidate_outputs(candidate_table: pd.DataFrame) -> tuple[Path, Path]:
    """Save CSV files for bot candidates and cluster summary."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bot_candidates = candidate_table[candidate_table["is_bot"] == 1].copy()
    bot_candidates.to_csv(config.BOT_CANDIDATES_PATH, index=False)

    clustered = bot_candidates[bot_candidates["cluster_id"] != NO_CLUSTER_ID]
    if clustered.empty:
        cluster_summary = pd.DataFrame(
            columns=(
                "cluster_id",
                "bot_count",
                "mean_anomaly_score",
                "mean_risk_score",
                "min_anomaly_score",
                "mean_graph_degree",
                "max_graph_degree",
                "user_ids",
            )
        )
    else:
        cluster_summary = (
            clustered.groupby("cluster_id")
            .agg(
                bot_count=("user_id", "size"),
                mean_anomaly_score=("anomaly_score", "mean"),
                mean_risk_score=("risk_score", "mean"),
                min_anomaly_score=("anomaly_score", "min"),
                mean_graph_degree=("graph_degree_actual", "mean"),
                max_graph_degree=("graph_degree_actual", "max"),
                user_ids=("user_id", lambda values: ", ".join(map(str, values))),
            )
            .reset_index()
            .sort_values(["bot_count", "mean_anomaly_score"], ascending=[False, True])
        )
    cluster_summary.to_csv(config.CLUSTER_SUMMARY_PATH, index=False)
    return config.BOT_CANDIDATES_PATH, config.CLUSTER_SUMMARY_PATH


def localized_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a DataFrame with Ukrainian column labels and compact numeric formatting."""
    display = frame.copy() if columns is None else frame.loc[:, columns].copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}".rstrip("0").rstrip("."))
    display = display.rename(columns=UKRAINIAN_COLUMNS)
    return display.to_html(index=False, escape=True, classes="data-table")


def render_report_html(
    candidate_table: pd.DataFrame,
    cluster_paths: list[Path],
    graph: nx.Graph,
    filtered: nx.Graph,
) -> Path:
    """Render a static HTML report with tables and links to graph views."""
    bot_candidates = candidate_table[candidate_table["is_bot"] == 1].copy()
    cluster_summary = pd.read_csv(config.CLUSTER_SUMMARY_PATH)
    top_candidates = bot_candidates.sort_values("anomaly_score").head(50)
    display_columns = [
        "user_id",
        "risk_score",
        "risk_reasons",
        "anomaly_score",
        "cluster_id",
        "comment_count_per_day",
        "duplicate_ratio",
        "graph_degree_actual",
        "category_diversity",
        "clustering_coefficient",
        "unique_channels",
    ]
    links = "\n".join(
        f'<li><a href="{html.escape(path.relative_to(config.OUTPUT_DIR).as_posix())}">{html.escape(path.name)}</a></li>'
        for path in cluster_paths
    ) or "<li>Окремі сторінки кластерів не створені, бо модель не знайшла кластерних ботоферм.</li>"
    report = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Аналітичний звіт щодо ботоферм</title>
  <style>
    body {{ margin: 0; padding: 32px; font-family: Arial, sans-serif; color: #111827; background: #eef2f7; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    .hero {{ padding: 26px 28px; color: #f8fafc; background: #0b1020; border-radius: 8px; }}
    h1, h2 {{ margin: 0 0 14px; }}
    p {{ max-width: 920px; line-height: 1.55; }}
    section {{ margin: 0 0 28px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; }}
    .metric {{ padding: 16px; background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; }}
    .metric strong {{ display: block; font-size: 24px; margin-bottom: 4px; }}
    .panel {{ padding: 18px; background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border: 1px solid #e5e7eb; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; white-space: nowrap; }}
    a {{ color: #1d4ed8; }}
    .hero a {{ color: #bfdbfe; }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Аналітичний звіт щодо виявлення ботоферм</h1>
    <p>Звіт зведено з результатів некерованого навчання та аналізу графа спільної активності. Дані варто трактувати як аналітичні індикатори для перевірки, а не як автоматичний доказ зловмисної поведінки.</p>
    <p><a href="botfarm_graph.html">Відкрити оптимізований інтерактивний граф</a></p>
  </section>
  <section class="summary">
    <div class="metric"><strong>{graph.number_of_nodes()}</strong>Усього користувачів</div>
    <div class="metric"><strong>{graph.number_of_edges()}</strong>Зв'язків спільної активності</div>
    <div class="metric"><strong>{len(bot_candidates)}</strong>Бот-кандидатів</div>
    <div class="metric"><strong>{filtered.number_of_nodes()}</strong>Вузлів у головному графі</div>
  </section>
  <section class="panel">
    <h2>Зведення кластерів</h2>
    {localized_table(cluster_summary)}
  </section>
  <section class="panel">
    <h2>Найсильніші бот-кандидати</h2>
    {localized_table(top_candidates, display_columns)}
  </section>
  <section class="panel">
    <h2>Окремі графи кластерів</h2>
    <ul>{links}</ul>
  </section>
</main>
</body>
</html>
"""
    config.REPORT_PATH.write_text(report, encoding="utf-8")
    return config.REPORT_PATH


def render_cluster_dashboards(
    graph: nx.Graph,
    results: pd.DataFrame,
    candidate_table: pd.DataFrame,
    comments: pd.DataFrame,
    output_dir: Path = config.CLUSTERS_DIR,
) -> list[Path]:
    """Render one drill-down HTML per bot-farm cluster."""
    if output_dir.exists():
        for old_file in output_dir.glob("cluster_*.html"):
            old_file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_paths: list[Path] = []
    bot_results = results[(results["is_bot"] == 1) & (results["cluster_id"] != NO_CLUSTER_ID)]
    if bot_results.empty:
        return cluster_paths

    for cluster_id, cluster_users in bot_results.groupby("cluster_id"):
        members = {str(uid) for uid in cluster_users["user_id"]}
        members_in_graph = members.intersection(graph.nodes)
        if not members_in_graph:
            continue
        neighbours: set[str] = set()
        for member in members_in_graph:
            neighbours.update(
                strongest_neighbors(
                    graph,
                    member,
                    min_edge_weight=1,
                    limit=config.VIZ_CLUSTER_NEIGHBORS_PER_BOT,
                )
            )
        keep = members_in_graph | neighbours
        subgraph = graph.subgraph(keep).copy()
        sub_results = results[results["user_id"].isin(subgraph.nodes)]
        sub_profiles = build_user_profiles(subgraph, graph, results, candidate_table, comments)
        network = build_network(subgraph, sub_results, sub_profiles)
        cluster_path = output_dir / f"cluster_{int(cluster_id)}.html"
        network.save_graph(str(cluster_path))
        legend = legend_html(
            node_count=subgraph.number_of_nodes(),
            edge_count=subgraph.number_of_edges(),
            total_nodes=subgraph.number_of_nodes(),
            total_edges=subgraph.number_of_edges(),
            report_href="../analysis_report.html",
        )
        inject_legend(cluster_path, legend, sub_profiles)
        cluster_paths.append(cluster_path)
    return cluster_paths


def render_dashboard() -> Path:
    """Render the graph dashboard to output/botfarm_graph.html."""
    config.configure_logging()
    config.ensure_directories()
    graph = load_graph()
    results = load_results()
    comments = load_comments()
    lookup = result_lookup(results)

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    filtered = filter_graph_for_viz(graph, lookup)
    candidate_table = build_candidate_table(graph, results)
    profiles = build_user_profiles(filtered, graph, results, candidate_table, comments)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.USER_PROFILES_PATH.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    network = build_network(filtered, results, profiles)
    config.DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    network.save_graph(str(config.DASHBOARD_PATH))
    legend = legend_html(
        node_count=filtered.number_of_nodes(),
        edge_count=filtered.number_of_edges(),
        total_nodes=total_nodes,
        total_edges=total_edges,
    )
    inject_legend(config.DASHBOARD_PATH, legend, profiles)
    print(
        f"[D] Dashboard saved → output/botfarm_graph.html "
        f"({filtered.number_of_nodes()}/{total_nodes} nodes, "
        f"{filtered.number_of_edges()}/{total_edges} edges shown)"
    )

    cluster_paths = render_cluster_dashboards(graph, results, candidate_table, comments)
    for cluster_path in cluster_paths:
        print(f"[D] Cluster dashboard saved → {cluster_path.relative_to(config.BASE_DIR)}")

    bot_candidates_path, cluster_summary_path = save_candidate_outputs(candidate_table)
    report_path = render_report_html(candidate_table, cluster_paths, graph, filtered)
    print(f"[D] Bot candidates table saved → {bot_candidates_path.relative_to(config.BASE_DIR)}")
    print(f"[D] Cluster summary saved → {cluster_summary_path.relative_to(config.BASE_DIR)}")
    print(f"[D] User profiles saved → {config.USER_PROFILES_PATH.relative_to(config.BASE_DIR)}")
    print(f"[D] Static report saved → {report_path.relative_to(config.BASE_DIR)}")

    return config.DASHBOARD_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the Module D command-line parser."""
    parser = argparse.ArgumentParser(description="Render bot-farm graph dashboard.")
    return parser


def main() -> None:
    """Run dashboard rendering from the command line."""
    build_parser().parse_args()
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    output_path = render_dashboard()
    print(f"Module D output saved → {output_path}")


if __name__ == "__main__":
    main()
