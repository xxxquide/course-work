"""Category tagging for Telegram comments used as explainable evidence signals."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

import config


CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "18_plus": (
        r"\b18\+",
        r"\bnsfw\b",
        r"\bпорно\b",
        r"\bеротик\w*",
        r"\bінтим\w*",
        r"\bинтим\w*",
        r"\bсекс\w*",
    ),
    "military": (
        r"\bзсу\b",
        r"\bвсу\b",
        r"\bвійськ\w*",
        r"\bвоенн\w*",
        r"\bмобілізац\w*",
        r"\bмобилизац\w*",
        r"\bтцк\b",
        r"\bфронт\w*",
        r"\bдрон\w*",
        r"\bракета\w*",
        r"\bппо\b",
        r"\bбпла\b",
    ),
    "ipso": (
        r"\bіпсо\b",
        r"\bипсо\b",
        r"\bдезінформац\w*",
        r"\bдезинформац\w*",
        r"\bфейк\w*",
        r"\bпропаганд\w*",
        r"\bботоферм\w*",
        r"\bпсихологічн\w+\s+операц\w*",
    ),
    "violence_threats": (
        r"\bвбити\b",
        r"\bубить\b",
        r"\bрозстріл\w*",
        r"\bрасстрел\w*",
        r"\bпідірв\w*",
        r"\bвзорв\w*",
        r"\bтерорист\w*",
        r"\bтеракт\w*",
        r"\bсмерт\w*",
    ),
    "political_agitation": (
        r"\bвибор\w*",
        r"\bвыбор\w*",
        r"\bголосуй\w*",
        r"\bпарті[яї]\w*",
        r"\bпартия\w*",
        r"\bпрезидент\w*",
        r"\bдепутат\w*",
        r"\bзрада\b",
        r"\bперемога\b",
    ),
    "spam_scam": (
        r"\bзаробіт\w*",
        r"\bзаработ\w*",
        r"\bкрипт\w*",
        r"\bказино\b",
        r"\bставк\w*",
        r"\bбонус\w*",
        r"\bрозіграш\w*",
        r"\bрозыгрыш\w*",
        r"\bперейдіть\b",
        r"\bпереходи\b",
        r"https?://",
        r"t\.me/",
    ),
}

CATEGORY_LABELS_UK = {
    "18_plus": "18+",
    "military": "Військова тематика",
    "ipso": "ІПСО / дезінформація",
    "violence_threats": "Насильство / погрози",
    "political_agitation": "Політична агітація",
    "spam_scam": "Спам / шахрайство",
}

COMPILED_PATTERNS = {
    category: tuple(re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in patterns)
    for category, patterns in CATEGORY_PATTERNS.items()
}


def normalize_categories(categories: str | Iterable[str] | None) -> list[str]:
    """Return valid category names from CLI/config input."""
    if categories is None:
        return list(CATEGORY_PATTERNS)
    if isinstance(categories, str):
        requested = [part.strip() for part in categories.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in categories if str(part).strip()]
    invalid = sorted(set(requested) - set(CATEGORY_PATTERNS))
    if invalid:
        raise ValueError(f"Unknown categories: {invalid}. Available: {sorted(CATEGORY_PATTERNS)}")
    return list(dict.fromkeys(requested))


def tag_text(text: object, enabled_categories: Iterable[str] | None = None) -> tuple[list[str], list[str]]:
    """Return matched category names and exact terms/pattern hits for one text."""
    value = "" if pd.isna(text) else str(text)
    tags: list[str] = []
    terms: list[str] = []
    for category in normalize_categories(enabled_categories):
        category_terms: list[str] = []
        for pattern in COMPILED_PATTERNS[category]:
            category_terms.extend(match.group(0) for match in pattern.finditer(value))
        if category_terms:
            tags.append(category)
            terms.extend(f"{category}:{term}" for term in sorted(set(category_terms), key=str.casefold))
    return tags, terms


def tag_comments(
    frame: pd.DataFrame,
    enabled_categories: Iterable[str] | None = None,
    mode: str = "tag",
) -> pd.DataFrame:
    """Add category tag columns and optionally filter to matched comments."""
    if mode not in {"tag", "filter"}:
        raise ValueError("category mode must be either 'tag' or 'filter'.")
    enabled = normalize_categories(enabled_categories)
    output = frame.copy()
    if "message_text" not in output.columns:
        output["message_text"] = ""
    tagged = output["message_text"].map(lambda text: tag_text(text, enabled))
    output["category_tags"] = tagged.map(lambda item: ",".join(item[0]))
    output["category_match_terms"] = tagged.map(lambda item: ",".join(item[1]))
    if mode == "filter":
        output = output[output["category_tags"].astype(str).str.len() > 0].copy()
    return output


def category_columns() -> tuple[str, ...]:
    """Return per-user ratio feature column names for configured categories."""
    return tuple(f"category_{category}_ratio" for category in CATEGORY_PATTERNS)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI for tagging an existing raw comments file."""
    parser = argparse.ArgumentParser(description="Tag Telegram comments with content categories.")
    parser.add_argument("--mode", choices=("tag", "filter"), default="tag")
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--input", type=Path, default=config.RAW_COMMENTS_PATH)
    parser.add_argument("--output", type=Path, default=config.FILTERED_COMMENTS_PATH)
    return parser


def main() -> None:
    """Tag comments from the command line."""
    args = build_parser().parse_args()
    frame = pd.read_csv(args.input, dtype=str)
    tagged = tag_comments(frame, args.categories, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(args.output, index=False)
    print(f"Tagged comments saved -> {args.output}")


if __name__ == "__main__":
    main()
