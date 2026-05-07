"""CLI orchestrator that runs Telegram bot-farm detection modules A through D."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import collector
import config
import features
import ml_engine
import visualizer


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the pipeline command-line parser."""
    parser = argparse.ArgumentParser(description="Run the Telegram bot-farm detection pipeline.")
    parser.add_argument("--collect", action="store_true", help="Run Module A data collection.")
    parser.add_argument("--features", action="store_true", help="Run Module B graph and features.")
    parser.add_argument("--ml", action="store_true", help="Run Module C ML detection.")
    parser.add_argument("--visualize", action="store_true", help="Run Module D dashboard rendering.")
    parser.add_argument("--all", action="store_true", help="Run all modules in sequence.")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_POST_LIMIT, help="Max posts per channel.")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated Telegram channel usernames.")
    parser.add_argument("--full-refresh", action="store_true", help="Rescan selected channels and deduplicate into cache.")
    parser.add_argument("--retry-inactive-channels", action="store_true", help="Retry channels previously marked inactive/invalid.")
    parser.add_argument("--comments-per-post", type=int, default=None, help="Optional max comments per post.")
    parser.add_argument("--comment-sample", choices=("latest", "earliest"), default="latest", help="Which comments to keep when capped.")
    parser.add_argument("--channel-delay", type=float, default=config.DEFAULT_CHANNEL_DELAY_SECONDS, help="Delay between channels in seconds.")
    parser.add_argument("--post-delay", type=float, default=config.DEFAULT_POST_DELAY_SECONDS, help="Delay between comment-thread requests in seconds.")
    parser.add_argument("--category-mode", choices=("tag", "filter"), default="tag", help="Tag all comments or filter export to selected categories.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category names.")
    return parser


def should_run_all(args: argparse.Namespace) -> bool:
    """Return whether all modules should run for the current CLI options."""
    return args.all or not any((args.collect, args.features, args.ml, args.visualize))


def print_module_complete(module: str, output_file: Path) -> None:
    """Print the required module completion line."""
    print(f"✅ Module {module} complete — {output_file}")


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Run selected pipeline modules and return output paths plus statistics."""
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    config.ensure_directories()
    outputs: dict[str, object] = {}
    run_all = should_run_all(args)

    if run_all or args.collect:
        raw_comments = asyncio.run(
            collector.collect_all(
                limit=args.limit,
                channels=args.channels,
                full_refresh=args.full_refresh,
                retry_inactive_channels=args.retry_inactive_channels,
                comments_per_post=args.comments_per_post,
                comment_sample=args.comment_sample,
                channel_delay=args.channel_delay,
                post_delay=args.post_delay,
                category_mode=args.category_mode,
                selected_categories=args.categories,
            )
        )
        outputs["raw_comments"] = raw_comments
        print_module_complete("A", raw_comments)

    if run_all or args.features:
        features_path, graph_path = features.build_features()
        outputs["features"] = features_path
        outputs["graph"] = graph_path
        print_module_complete("B", features_path)

    if run_all or args.ml:
        ml_results, stats = ml_engine.detect_bot_farms()
        outputs["ml_results"] = ml_results
        outputs["stats"] = stats
        print_module_complete("C", ml_results)

    if run_all or args.visualize:
        dashboard = visualizer.render_dashboard()
        outputs["dashboard"] = dashboard
        print_module_complete("D", dashboard)

    if run_all:
        print_run_summary(outputs)
    return outputs


def print_run_summary(outputs: dict[str, object]) -> None:
    """Print a full run summary for --all executions."""
    stats = outputs.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    print("\nFull run summary")
    print(f"- Raw comments: {outputs.get('raw_comments', config.RAW_COMMENTS_PATH)}")
    print(f"- Comments cache: {config.COMMENTS_CACHE_PATH}")
    print(f"- Collection state: {config.COLLECTION_STATE_PATH}")
    print(f"- Features: {outputs.get('features', config.FEATURES_PATH)}")
    print(f"- Graph: {outputs.get('graph', config.GRAPH_PATH)}")
    print(f"- ML results: {outputs.get('ml_results', config.ML_RESULTS_PATH)}")
    print(f"- Dashboard: {outputs.get('dashboard', config.DASHBOARD_PATH)}")
    print(f"- Static report: {config.REPORT_PATH}")
    print(f"- Bot candidates table: {config.BOT_CANDIDATES_PATH}")
    print(f"- Cluster summary: {config.CLUSTER_SUMMARY_PATH}")
    print(f"- User profiles: {config.USER_PROFILES_PATH}")
    print(f"- Total users: {int(stats.get('total_users', 0))}")
    print(f"- Bot candidates: {int(stats.get('bot_count', 0))} ({float(stats.get('bot_percentage', 0.0)):.1f}%)")
    print(f"- Bot farm clusters: {int(stats.get('cluster_count', 0))}")


def main() -> None:
    """Run the command-line pipeline."""
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
