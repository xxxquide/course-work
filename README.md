# Telegram Botfarm Detector

## Project overview

Telegram Botfarm Detector is an unsupervised graph-analysis pipeline for finding coordinated comment behavior in public Telegram news channels. It collects recent post comments, builds a user co-activity graph, engineers behavioral and graph features, detects anomalous users with machine learning, and renders an interactive HTML investigation dashboard. The project is designed as a coursework prototype that runs the same way on macOS Apple Silicon and Windows x64. It does not require GPU libraries or supervised labels.

## System requirements

- Python 3.10, 3.11, or 3.12
- macOS Apple Silicon M2, arm64
- Windows x64
- Telegram API credentials from https://my.telegram.org/apps

## Installation

### macOS M2

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### Windows x64

```bat
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

## Telegram API setup

1. Open https://my.telegram.org/apps and sign in with the Telegram account that will collect public-channel comments.
2. Create a Telegram application if you do not already have one.
3. Copy the generated `api_id` and `api_hash`.
4. Copy `.env.example` to `.env` in the project root.
5. Fill in `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `TELEGRAM_PHONE`.
6. Optionally set `TELEGRAM_CHANNELS=vinnicatruexa,vn20minut,truexanewsua,truexakyiv` or pass `--channels` at runtime.
7. On first collection, Telethon may prompt for a login code sent by Telegram.

## Usage

Run the full pipeline with the default 200-post limit:

```bash
python pipeline.py --all
```

Run everything with explicit channels and a larger post cap:

```bash
python pipeline.py --all --limit 500 --channels ukraine_now,suspilne_news,pravda_ua
```

Run the expanded Ukrainian news-channel set from the default configuration:

```bash
python pipeline.py --all --limit 100
```

Subsequent runs are incremental by default: collected comments are deduplicated into `data/comments_cache.csv`, channel progress is stored in `data/collection_state.json`, and `data/raw_comments.csv` is regenerated from the cache for the ML pipeline.

Force a rescan while still deduplicating into the cache:

```bash
python pipeline.py --all --limit 100 --full-refresh
```

Retry channels previously marked as inactive or invalid:

```bash
python pipeline.py --collect --retry-inactive-channels
```

Optionally cap comments per post when you need a faster exploratory run:

```bash
python pipeline.py --all --limit 100 --comments-per-post 50 --comment-sample latest
```

Use category filtering for a focused export and report:

```bash
python pipeline.py --all --limit 100 --category-mode filter --categories ipso,military
```

Run a custom channel set. Both `@channel` and `channel` formats are accepted:

```bash
python pipeline.py --all --limit 100 --channels @vinnicatruexa,@vn20minut,@chesnavinnytsia,@vinnytsiarealll,@vn_right_now,@truexanewsua,@voynareal,@kievreal1,@u_now,@novynu_ukraina,@tgsn_ua,@novini_ukrtg,@truexakyiv
```

Run only feature engineering after raw comments already exist:

```bash
python pipeline.py --features
```

Expected output includes progress lines such as:

```text
[A] Collected 1234 comments from ukraine_now
✅ Module A complete — data/raw_comments.csv
[B] Feature matrix shape: 842 users × 14 features
✅ Module B complete — data/features.csv
[C] Total users: 842
[C] Bot candidates detected: 42 (5.0%)
[C] Bot farm clusters found: 3
✅ Module C complete — data/ml_results.csv
[D] Dashboard saved → output/botfarm_graph.html
✅ Module D complete — output/botfarm_graph.html
```

## Output files

| Filename | Location | Description |
| --- | --- | --- |
| `raw_comments.csv` | `data/` | Raw collected comments with user, post, reply, channel, and timestamp fields. |
| `comments_cache.csv` | `data/` | Persistent deduplicated cache used by incremental collection. |
| `collection_state.json` | `data/` | Per-channel progress, status, last seen post, and collection counters. |
| `features.csv` | `data/` | Per-user feature matrix used by the ML engine, including timing, graph, text duplication, and category signals. |
| `graph.graphml` | `data/` | Weighted co-activity graph where users are connected by near-simultaneous same-post commenting. |
| `ml_results.csv` | `data/` | User anomaly scores, bot labels, and bot-farm cluster IDs. |
| `botfarm_graph.html` | `output/` | Optimized interactive PyVis dashboard focused on bot candidates and their strongest neighbours. |
| `analysis_report.html` | `output/` | Static summary report with cluster metrics, top suspicious users, and links to cluster graph pages. |
| `bot_candidates.csv` | `output/` | Table of all ML bot candidates with anomaly scores, cluster IDs, and feature values. |
| `cluster_summary.csv` | `output/` | Compact summary of detected bot-farm clusters. |
| `user_profiles.json` | `output/` | Per-user details embedded into graph click panels. |
| `cluster_*.html` | `output/clusters/` | Small per-cluster graph pages for focused visual inspection. |

## How it works

**Module A — collector.py:** The collector connects to Telegram with Telethon credentials loaded from `.env`, iterates over recent posts in configurable public channels, and saves non-anonymous comments to `data/raw_comments.csv`. It retries transient network and RPC failures, handles Telegram flood waits, and prints per-channel collection progress.

**Module B — features.py:** The feature builder loads raw comments, constructs a weighted co-activity graph with NetworkX, and connects users who comment on the same post within the configured time window. It computes reaction speed, posting rate, duplicate-content ratio, graph degree, clustering coefficient, channel diversity, and category-signal features for each user.

**Module C — ml_engine.py:** The ML engine imputes missing values, scales all features, and uses Isolation Forest to identify anomalous users without requiring labeled training data. It then clusters bot candidates with DBSCAN, falls back to KMeans when density-based clustering cannot identify multiple groups, and adds an explainable 0–100 suspicion score.

**Module D — visualizer.py:** The visualizer combines the GraphML graph and ML results into an optimized Ukrainian-language PyVis dashboard plus static review tables. The main dashboard intentionally shows only bot candidates and their strongest neighbours so the browser does not attempt to render the full dense co-activity graph. Normal users are grey and unlabeled, unclustered bot candidates are orange, clustered bot-farm members use distinct cluster colors, and edge width reflects repeated co-activity.

## Interpreting results

Start with `output/analysis_report.html` after running the pipeline. It gives a readable summary, the strongest candidate accounts, cluster-level metrics, and links to focused cluster graph pages. Use `output/botfarm_graph.html` for the optimized overview graph, and use `output/clusters/cluster_*.html` when you need to inspect one suspected farm at a time.

The raw `data/graph.graphml` file is not meant to be opened directly in a browser. It is a machine-readable NetworkX/GraphML file for tools such as Python, Gephi, or Cytoscape. For coursework review, the most useful human-readable files are `analysis_report.html`, `bot_candidates.csv`, `cluster_summary.csv`, and the HTML graph pages.

Grey nodes are users classified as normal, orange nodes are isolated bot candidates, and colored cluster nodes are anomalous users assigned to a bot-farm cluster. Clusters indicate groups of accounts that share suspicious timing, content, and co-activity patterns; they should be treated as investigation leads rather than automatic proof of malicious automation.
