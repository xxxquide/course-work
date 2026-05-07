"""Module A: collect Telegram channel comments into a cached CSV dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError, MsgIdInvalidError, PeerIdInvalidError, RPCError
from telethon.tl.custom.message import Message

import categories as category_rules
import config


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

COMMENT_COLUMNS = [
    "comment_key",
    "comment_id",
    "user_id",
    "message_text",
    "timestamp",
    "post_id",
    "reply_to_msg_id",
    "channel",
    "post_timestamp",
    "collected_at",
    "category_tags",
    "category_match_terms",
]
SKIPPED_CHANNEL_STATUSES = {"inactive_no_comments", "invalid_username"}


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def parse_channels(channels: str | None) -> list[str]:
    """Parse channel CLI input or return configured defaults."""
    source: Iterable[str] = channels.split(",") if channels else config.TELEGRAM_CHANNELS
    parsed = list(
        dict.fromkeys(channel.strip().lstrip("@") for channel in source if channel.strip())
    )
    if not parsed:
        raise ValueError("At least one Telegram channel username is required.")
    return parsed


def bounded_limit(limit: int) -> int:
    """Clamp the requested post limit to the supported collection range."""
    if limit < config.MIN_POST_LIMIT:
        LOGGER.warning(
            "Requested limit %s is below %s; using %s.",
            limit,
            config.MIN_POST_LIMIT,
            config.MIN_POST_LIMIT,
        )
        return config.MIN_POST_LIMIT
    if limit > config.MAX_POST_LIMIT:
        LOGGER.warning(
            "Requested limit %s is above %s; using %s.",
            limit,
            config.MAX_POST_LIMIT,
            config.MAX_POST_LIMIT,
        )
        return config.MAX_POST_LIMIT
    return limit


def initial_channel_state(channel: str) -> dict[str, object]:
    """Return default metadata for a channel, including known bad channels."""
    if channel in config.INVALID_CHANNELS:
        status = "invalid_username"
        last_error = "Seeded from previous resolve failure."
    elif channel in config.INACTIVE_CHANNELS:
        status = "inactive_no_comments"
        last_error = "Seeded from previous run: no reachable comment threads."
    elif channel in config.LOW_VOLUME_CHANNELS:
        status = "low_volume"
        last_error = ""
    else:
        status = "active"
        last_error = ""
    return {
        "status": status,
        "last_seen_post_id": None,
        "last_success_at": None,
        "last_error": last_error,
        "posts_seen": 0,
        "posts_with_comments": 0,
        "comments_collected": 0,
        "flood_waits": 0,
        "skipped_posts": 0,
    }


def load_collection_state(path: Path = config.COLLECTION_STATE_PATH) -> dict[str, dict[str, object]]:
    """Load collection state, seeding known inactive/invalid channels when absent."""
    if path.exists():
        raw_state = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw_state = {}
    state = dict(raw_state)
    for channel in config.DEFAULT_CHANNELS:
        state.setdefault(channel, initial_channel_state(channel))
    return state


def save_collection_state(
    state: dict[str, dict[str, object]],
    path: Path = config.COLLECTION_STATE_PATH,
) -> None:
    """Persist collection state as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def infer_state_from_cache(
    state: dict[str, dict[str, object]],
    cache: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Populate missing last-seen counters from an existing comments cache."""
    if cache.empty or "channel" not in cache.columns or "post_id" not in cache.columns:
        return state
    for channel, channel_comments in cache.groupby("channel"):
        channel = str(channel)
        channel_state = state.setdefault(channel, initial_channel_state(channel))
        numeric_posts = pd.to_numeric(channel_comments["post_id"], errors="coerce").dropna()
        if not numeric_posts.empty and safe_int(channel_state.get("last_seen_post_id")) is None:
            channel_state["last_seen_post_id"] = int(numeric_posts.max())
        if int(channel_state.get("comments_collected", 0)) == 0:
            channel_state["comments_collected"] = int(len(channel_comments))
        if int(channel_state.get("posts_with_comments", 0)) == 0:
            channel_state["posts_with_comments"] = int(channel_comments["post_id"].nunique())
        if channel_state.get("status") in {None, "", "inactive_no_comments"} and len(channel_comments) > 0:
            channel_state["status"] = "low_volume" if len(channel_comments) < 100 else "active"
    return state


def load_comment_cache(path: Path = config.COMMENTS_CACHE_PATH) -> pd.DataFrame:
    """Load the persistent comment cache or bootstrap it from raw comments."""
    source = path if path.exists() else config.RAW_COMMENTS_PATH
    if not source.exists():
        return pd.DataFrame(columns=COMMENT_COLUMNS)
    frame = pd.read_csv(source, dtype=str).fillna("")
    return normalize_comment_frame(frame)


def normalize_comment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure comment cache columns exist and stable dedupe keys are populated."""
    output = frame.copy()
    for column in COMMENT_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output["comment_id"] = output["comment_id"].fillna("").astype(str)
    output["post_id"] = output["post_id"].fillna("").astype(str)
    output["reply_to_msg_id"] = output["reply_to_msg_id"].fillna("").astype(str)
    output["user_id"] = output["user_id"].fillna("").astype(str)
    output["timestamp"] = output["timestamp"].fillna("").astype(str)
    output["message_text"] = output["message_text"].fillna("").astype(str)
    output["channel"] = output["channel"].fillna("").astype(str)
    output["comment_key"] = output.apply(comment_key_from_row, axis=1)
    tagged = category_rules.tag_comments(output, mode="tag")
    return tagged.loc[:, COMMENT_COLUMNS]


def comment_key_from_row(row: pd.Series) -> str:
    """Build a stable key for deduplicating old and new comment records."""
    channel = str(row.get("channel", ""))
    post_id = str(row.get("post_id", ""))
    comment_id = str(row.get("comment_id", ""))
    if comment_id and comment_id.lower() != "nan":
        return f"{channel}:{post_id}:{comment_id}"
    return "|".join(
        (
            channel,
            post_id,
            str(row.get("reply_to_msg_id", "")),
            str(row.get("user_id", "")),
            str(row.get("timestamp", "")),
            str(row.get("message_text", "")),
        )
    )


def merge_and_save_cache(
    cache: pd.DataFrame,
    records: list[dict[str, object]],
    category_mode: str,
    selected_categories: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge new records into the cache and export raw/filter views."""
    new_frame = normalize_comment_frame(pd.DataFrame(records, columns=COMMENT_COLUMNS))
    merged = pd.concat([cache, new_frame], ignore_index=True)
    merged = normalize_comment_frame(merged)
    merged = merged.drop_duplicates("comment_key", keep="last").sort_values(
        ["channel", "post_id", "timestamp", "comment_id"]
    )
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(config.COMMENTS_CACHE_PATH, index=False)
    export_frame = category_rules.tag_comments(merged, selected_categories, category_mode)
    export_frame.to_csv(config.RAW_COMMENTS_PATH, index=False)
    if category_mode == "filter":
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        export_frame.to_csv(config.FILTERED_COMMENTS_PATH, index=False)
    return merged, export_frame


async def retry_telegram_call(
    operation: Callable[[], Awaitable[T]],
    label: str,
    skip_exceptions: tuple[type[BaseException], ...] = (),
    channel_state: dict[str, object] | None = None,
) -> T | None:
    """Run a Telegram API operation with retry and rate-limit handling."""
    delay = config.RATE_LIMIT_BASE_DELAY_SECONDS
    for attempt in range(1, config.TELEGRAM_REQUEST_RETRIES + 1):
        try:
            return await operation()
        except FloodWaitError as exc:
            if channel_state is not None:
                channel_state["flood_waits"] = int(channel_state.get("flood_waits", 0)) + 1
            LOGGER.warning("FloodWaitError during %s: waiting %s seconds.", label, exc.seconds)
            await asyncio.sleep(exc.seconds)
            if not ask_continue_after_flood_wait(label, exc.seconds):
                LOGGER.warning("User chose to stop after Telegram flood wait during %s.", label)
                return None
        except skip_exceptions as exc:
            LOGGER.debug("Skipping %s due to expected %s.", label, exc.__class__.__name__)
            return None
        except (OSError, ConnectionError, TimeoutError, RPCError) as exc:
            if attempt == config.TELEGRAM_REQUEST_RETRIES:
                LOGGER.error("Telegram operation failed after %s attempts: %s", attempt, label)
                if channel_state is not None:
                    channel_state["last_error"] = str(exc)
                return None
            LOGGER.warning(
                "Telegram operation failed (%s/%s) for %s: %s. Retrying in %s seconds.",
                attempt,
                config.TELEGRAM_REQUEST_RETRIES,
                label,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2
    return None


def ask_continue_after_flood_wait(label: str, waited_seconds: int) -> bool:
    """Ask whether collection should continue after a Telegram flood wait."""
    prompt = (
        f"Telegram requested a {waited_seconds}s wait during {label}. "
        "Continue collection? [y/N]: "
    )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        LOGGER.warning("No interactive input available after flood wait; continuing safely.")
        return True


def should_skip_channel(
    channel: str,
    state: dict[str, dict[str, object]],
    retry_inactive_channels: bool,
) -> bool:
    """Return whether channel should be skipped based on previous state."""
    channel_state = state.setdefault(channel, initial_channel_state(channel))
    status = str(channel_state.get("status", "active"))
    if status in SKIPPED_CHANNEL_STATUSES and not retry_inactive_channels:
        print(f"[A] Skipping @{channel} ({status}; use --retry-inactive-channels to retry)")
        return True
    return False


async def collect_comments_for_channel(
    client: TelegramClient,
    channel: str,
    limit: int,
    state: dict[str, dict[str, object]],
    full_refresh: bool = False,
    retry_inactive_channels: bool = False,
    comments_per_post: int | None = None,
    comment_sample: str = "latest",
    post_delay: float = config.DEFAULT_POST_DELAY_SECONDS,
) -> list[dict[str, object]]:
    """Collect recent post comments for one public Telegram channel."""
    records: list[dict[str, object]] = []
    channel_state = state.setdefault(channel, initial_channel_state(channel))
    if should_skip_channel(channel, state, retry_inactive_channels):
        return records

    async def get_entity() -> object:
        """Resolve the configured channel username to a Telethon entity."""
        return await client.get_entity(channel)

    channel_entity = await retry_telegram_call(
        get_entity,
        f"resolve @{channel}",
        channel_state=channel_state,
    )
    if channel_entity is None:
        channel_state["status"] = "invalid_username"
        channel_state["last_error"] = channel_state.get("last_error") or "Could not resolve username."
        return records

    last_seen = safe_int(channel_state.get("last_seen_post_id"))
    post_count = 0
    posts_with_comments = 0
    skipped_posts = 0
    newest_seen_post_id = last_seen
    old_posts_processed = 0
    try:
        async for post in client.iter_messages(channel_entity, limit=limit):
            if not isinstance(post, Message) or post.id is None:
                continue
            post_count += 1
            post_id = int(post.id)
            newest_seen_post_id = max(newest_seen_post_id or post_id, post_id)
            if not full_refresh and last_seen is not None and post_id <= last_seen:
                if old_posts_processed >= config.INCREMENTAL_LOOKBACK_POSTS:
                    skipped_posts += 1
                    break
                old_posts_processed += 1
            if not post_has_comment_thread(post):
                continue
            posts_with_comments += 1
            comments = await collect_comments_for_post(
                client,
                channel_entity,
                channel,
                post,
                comments_per_post=comments_per_post,
                comment_sample=comment_sample,
                channel_state=channel_state,
            )
            records.extend(comments)
            if post_delay > 0:
                await asyncio.sleep(post_delay)
    except FloodWaitError as exc:
        channel_state["flood_waits"] = int(channel_state.get("flood_waits", 0)) + 1
        LOGGER.warning("FloodWaitError while listing posts for %s: waiting %s seconds.", channel, exc.seconds)
        await asyncio.sleep(exc.seconds)
        ask_continue_after_flood_wait(f"listing posts for @{channel}", exc.seconds)
    except (OSError, ConnectionError, TimeoutError, RPCError) as exc:
        channel_state["last_error"] = str(exc)
        LOGGER.error("Could not list posts for %s after %s posts: %s", channel, post_count, exc)

    update_channel_state(
        channel_state,
        post_count=post_count,
        posts_with_comments=posts_with_comments,
        comments_collected=len(records),
        skipped_posts=skipped_posts,
        newest_seen_post_id=newest_seen_post_id,
    )
    if post_count > 0 and posts_with_comments == 0:
        LOGGER.warning(
            "@%s exposed %s posts but none had a reachable comment thread. "
            "The channel likely has comments disabled or no linked discussion group.",
            channel,
            post_count,
        )
    print(f"[A] Collected {len(records)} comments from {channel}")
    return records


def update_channel_state(
    channel_state: dict[str, object],
    post_count: int,
    posts_with_comments: int,
    comments_collected: int,
    skipped_posts: int,
    newest_seen_post_id: int | None,
) -> None:
    """Update persistent channel metadata after a collection attempt."""
    channel_state["posts_seen"] = int(channel_state.get("posts_seen", 0)) + post_count
    channel_state["posts_with_comments"] = int(channel_state.get("posts_with_comments", 0)) + posts_with_comments
    channel_state["comments_collected"] = int(channel_state.get("comments_collected", 0)) + comments_collected
    channel_state["skipped_posts"] = int(channel_state.get("skipped_posts", 0)) + skipped_posts
    if newest_seen_post_id is not None:
        channel_state["last_seen_post_id"] = max(
            safe_int(channel_state.get("last_seen_post_id")) or newest_seen_post_id,
            newest_seen_post_id,
        )
    if post_count > 0 and posts_with_comments == 0:
        channel_state["status"] = "inactive_no_comments"
        channel_state["last_error"] = "No reachable comment threads in the inspected posts."
    elif comments_collected > 0:
        channel_state["status"] = "low_volume" if comments_collected < 100 else "active"
        channel_state["last_error"] = ""
        channel_state["last_success_at"] = utc_now_iso()


def safe_int(value: object) -> int | None:
    """Convert a value to int or return None."""
    if value in {None, "", "nan"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def post_has_comment_thread(post: Message) -> bool:
    """Return whether a channel post advertises a reachable comment thread."""
    replies = getattr(post, "replies", None)
    if replies is None:
        return False
    if not getattr(replies, "comments", False):
        return False
    return int(getattr(replies, "replies", 0) or 0) > 0


async def collect_comments_for_post(
    client: TelegramClient,
    channel_entity: object,
    channel: str,
    post: Message,
    comments_per_post: int | None = None,
    comment_sample: str = "latest",
    channel_state: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Collect comments attached to a single Telegram channel post."""
    records: list[dict[str, object]] = []
    post_timestamp = to_utc_iso(post.date)

    if not post_has_comment_thread(post):
        return records

    async def list_comments() -> list[Message]:
        """Fetch available comments for one post."""
        comments: list[Message] = []
        reverse = comment_sample == "earliest"
        async for comment in client.iter_messages(channel_entity, reply_to=post.id, reverse=reverse):
            if isinstance(comment, Message):
                comments.append(comment)
            if comments_per_post is not None and len(comments) >= comments_per_post:
                break
        return comments

    comments = await retry_telegram_call(
        list_comments,
        f"comments for @{channel}/{post.id}",
        skip_exceptions=(PeerIdInvalidError, MsgIdInvalidError),
        channel_state=channel_state,
    )
    if comments is None:
        return records

    for comment in comments:
        if comment.sender_id is None or comment.id is None:
            continue
        record = {
            "comment_id": str(comment.id),
            "user_id": str(comment.sender_id),
            "message_text": comment.message or "",
            "timestamp": to_utc_iso(comment.date),
            "post_id": str(post.id),
            "reply_to_msg_id": str(comment.reply_to_msg_id or post.id),
            "channel": channel,
            "post_timestamp": post_timestamp,
            "collected_at": utc_now_iso(),
        }
        record["comment_key"] = comment_key_from_row(pd.Series(record))
        tags, terms = category_rules.tag_text(record["message_text"])
        record["category_tags"] = ",".join(tags)
        record["category_match_terms"] = ",".join(terms)
        records.append(record)
    return records


def to_utc_iso(value: object) -> str:
    """Convert Telethon datetime values to UTC ISO-8601 strings."""
    if hasattr(value, "astimezone"):
        return value.astimezone(timezone.utc).isoformat()
    raise ValueError(f"Expected datetime-like value, received {type(value)!r}.")


async def collect_all(
    limit: int = config.DEFAULT_POST_LIMIT,
    channels: str | None = None,
    full_refresh: bool = False,
    retry_inactive_channels: bool = False,
    comments_per_post: int | None = None,
    comment_sample: str = "latest",
    channel_delay: float = config.DEFAULT_CHANNEL_DELAY_SECONDS,
    post_delay: float = config.DEFAULT_POST_DELAY_SECONDS,
    category_mode: str = "tag",
    selected_categories: str | None = None,
) -> Path:
    """Collect comments for all requested channels, cache them, and export raw CSV."""
    config.configure_logging()
    config.ensure_directories()
    config.validate_telegram_credentials()
    selected_channels = parse_channels(channels)
    post_limit = bounded_limit(limit)
    state = load_collection_state()
    cache = load_comment_cache()
    state = infer_state_from_cache(state, cache)

    client = TelegramClient(
        str(config.BASE_DIR / "telegram_session"),
        config.get_api_id(),
        str(config.TELEGRAM_API_HASH),
    )

    all_records: list[dict[str, object]] = []
    async with client:
        if not await client.is_user_authorized():
            await client.start(phone=str(config.TELEGRAM_PHONE))
        for index, channel in enumerate(selected_channels):
            all_records.extend(
                await collect_comments_for_channel(
                    client,
                    channel,
                    post_limit,
                    state,
                    full_refresh=full_refresh,
                    retry_inactive_channels=retry_inactive_channels,
                    comments_per_post=comments_per_post,
                    comment_sample=comment_sample,
                    post_delay=post_delay,
                )
            )
            save_collection_state(state)
            if channel_delay > 0 and index < len(selected_channels) - 1:
                await asyncio.sleep(channel_delay)

    merged, export_frame = merge_and_save_cache(cache, all_records, category_mode, selected_categories)
    save_collection_state(state)
    print(f"[A] Cache rows: {len(merged)}; raw export rows: {len(export_frame)}")
    return config.RAW_COMMENTS_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the Module A command-line parser."""
    parser = argparse.ArgumentParser(description="Collect Telegram comments for bot-farm analysis.")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_POST_LIMIT, help="Max posts per channel.")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel usernames.")
    parser.add_argument("--full-refresh", action="store_true", help="Rescan selected channels and deduplicate into cache.")
    parser.add_argument("--retry-inactive-channels", action="store_true", help="Retry channels previously marked inactive/invalid.")
    parser.add_argument("--comments-per-post", type=int, default=None, help="Optional max comments per post.")
    parser.add_argument("--comment-sample", choices=("latest", "earliest"), default="latest", help="Which comments to keep when capped.")
    parser.add_argument("--channel-delay", type=float, default=config.DEFAULT_CHANNEL_DELAY_SECONDS, help="Delay between channels in seconds.")
    parser.add_argument("--post-delay", type=float, default=config.DEFAULT_POST_DELAY_SECONDS, help="Delay between comment-thread requests in seconds.")
    parser.add_argument("--category-mode", choices=("tag", "filter"), default="tag", help="Tag all comments or filter export to selected categories.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category names for filtering/tagging.")
    return parser


def main() -> None:
    """Run Telegram collection from the command line."""
    args = build_parser().parse_args()
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    output_path = asyncio.run(
        collect_all(
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
    print(f"Module A output saved → {output_path}")


if __name__ == "__main__":
    main()
