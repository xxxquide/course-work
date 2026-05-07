# Architecture

```text
┌────────────────────┐
│ A: collector.py    │
│ Telegram comments  │
└─────────┬──────────┘
          │ data/raw_comments.csv
          ▼
┌────────────────────┐
│ B: features.py     │
│ Graph + features   │
└─────────┬──────────┘
          │ data/features.csv + data/graph.graphml
          ▼
┌────────────────────┐
│ C: ml_engine.py    │
│ Anomaly + clusters │
└─────────┬──────────┘
          │ data/ml_results.csv
          ▼
┌────────────────────┐
│ D: visualizer.py   │
│ PyVis dashboard    │
└────────────────────┘
          output/botfarm_graph.html
```

The system is a four-stage batch pipeline. Module A authenticates with Telegram through Telethon, reads configured public news channels, and writes comment-level observations to CSV. The raw schema preserves user identity, message text, UTC timestamp, post ID, reply ID, channel, and post timestamp so later stages can compute timing and graph features without calling Telegram again. Module B transforms comments into a co-activity graph: users become nodes, and edges represent pairs of users commenting on the same post within the configured time window. Module C converts graph and behavioral metrics into a scaled numeric matrix, detects outliers, and groups suspected bots. Module D merges graph topology and ML labels into a browser-based PyVis dashboard for analyst review.

The feature set intentionally mixes timing, content, volume, topology, and category evidence. `reaction_speed_mean` captures accounts that respond unusually fast after publication. `reaction_speed_std` separates consistently automated behavior from irregular human activity. `comment_count_per_day` measures posting intensity over the observation period. `duplicate_ratio` identifies repeated or near-repeated text using Levenshtein distance, a common signal in copy-paste campaigns. `graph_degree` counts how many other users an account repeatedly appears with in the co-activity graph. `clustering_coefficient` measures whether an account sits inside a tightly connected coordination group rather than a random crowd. `unique_channels` detects accounts active across several news channels, while category-ratio features capture 18+, military, IPSO, violence/threat, political agitation, and spam/scam signals as explainable evidence.

Isolation Forest is used because the task lacks reliable labels and bot farms are expected to be rare relative to ordinary commenters. It isolates unusual feature combinations without assuming a Gaussian distribution or linear boundary. DBSCAN is then applied only to anomalous users because bot farms are better modeled as dense coordination groups than as globally spherical clusters. DBSCAN can mark isolated anomalies as noise, which is useful for separating lone suspicious accounts from farms. KMeans is only a fallback when DBSCAN finds fewer than two non-noise clusters; this keeps output usable for small or weakly separated candidate sets while preserving the unsupervised design.

Cross-platform compatibility is handled by avoiding GPU frameworks and using dependencies with mature wheels for macOS arm64 and Windows x64. NumPy, pandas, scikit-learn, NetworkX, PyVis, Telethon, python-dotenv, tqdm, and python-Levenshtein all provide pip-installable releases in the pinned major-version ranges. The project uses `pathlib.Path` for every filesystem path, so directory separators and output locations remain portable. The code targets Python 3.10 through 3.12 and avoids platform-specific shell commands in runtime modules.

Known limitations remain important. Telegram comment availability depends on channel settings and account permissions, and API rate limits can interrupt large collections. Reaction speed is only as accurate as available post timestamps and may be missing for imported datasets. Levenshtein duplicate detection catches simple near-copies but not semantic paraphrases. Isolation Forest contamination is fixed at five percent, which may over- or under-estimate bot prevalence for some channels. Future improvements could add language-aware text embeddings, temporal burst detection, adaptive contamination tuning, persistent collection state, richer HTML filtering controls, and human-in-the-loop review labels for semi-supervised refinement.
