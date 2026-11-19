"""
The collector preserves the Stage I schema and can be run repeatedly. Existing
URLs and normalized titles are used as stable duplicate keys, while new IDs are
deterministic UUID5 values derived from the canonical URL.
"""
#Script para extraer noticias de los feeds RSS de La Jornada, Reforma, El Financiero y Google News.

from __future__ import annotations

import argparse
import csv
import html
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import feedparser


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "news_raw_corpus.tsv"
ERROR_LOG = ROOT / "data" / "raw" / "collection_errors.tsv"
SCHEMA = ["id", "title", "description", "source", "category", "date", "url", "retrieved_at"]
ID_NAMESPACE = uuid.UUID("c0be2d57-1f24-4ca7-a65f-fb0e2dfb6b4b")

# Category is assigned by the configured feed, not inferred from article text.
FEEDS = [
    ("La Jornada", "Cultura", "https://www.jornada.com.mx/rss/cultura.xml?v=1"),
    ("La Jornada", "Ciencia", "https://www.jornada.com.mx/rss/ciencias.xml?v=1"),
    ("La Jornada", "Negocios", "https://www.jornada.com.mx/rss/economia.xml?v=1"),
    ("Reforma", "Ciencia", "https://www.reforma.com/rss/ciencia.xml"),
    ("Reforma", "Cultura", "https://www.reforma.com/rss/cultura.xml"),
    ("Reforma", "Negocios", "https://www.reforma.com/rss/negocios.xml"),
    ("El Financiero", "Ciencia", "https://www.elfinanciero.com.mx/rss/ciencia.xml"),
    ("El Financiero", "Cultura", "https://www.elfinanciero.com.mx/rss/cultura.xml"),
    ("El Financiero", "Negocios", "https://www.elfinanciero.com.mx/rss/economia.xml"),
    ("Google News", "Ciencia", "https://news.google.com/rss/search?q=ciencia%20tecnologia&hl=es-419&gl=MX&ceid=MX:es-419"),
    ("Google News", "Cultura", "https://news.google.com/rss/search?q=cultura%20arte&hl=es-419&gl=MX&ceid=MX:es-419"),
    ("Google News", "Negocios", "https://news.google.com/rss/search?q=economia%20negocios&hl=es-419&gl=MX&ceid=MX:es-419"),
]


def canonical_url(value: str) -> str:
    value = html.unescape(str(value or "")).strip()
    value = re.sub(r"[?#].*$", "", value)
    return value.rstrip("/")


def clean_html(value: str) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_title(value: str) -> str:
    return re.sub(r"\W+", " ", clean_html(value).lower(), flags=re.UNICODE).strip()


def category_name(value: str) -> str:
    # Stage II uses three labels: Economía is intentionally part of Negocios.
    if value in {"Economía", "Economia", "Negocios"}:
        return "Negocios"
    if value in {"Ciencias", "Ciencia"}:
        return "Ciencia"
    if value == "Cultura":
        return "Cultura"
    raise ValueError(f"Unsupported category: {value!r}")


def entry_date(entry) -> str:
    for key in ("published", "updated", "created"):
        if entry.get(key):
            return str(entry[key])
    return ""


def entry_description(entry) -> str:
    return clean_html(entry.get("summary") or entry.get("description") or "")


def stable_id(url: str, title: str) -> str:
    key = canonical_url(url) or normalize_title(title)
    return str(uuid.uuid5(ID_NAMESPACE, key))


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row.setdefault("description", "")
        row["category"] = category_name(row.get("category", ""))
        row["url"] = canonical_url(row.get("url", ""))
        row["id"] = row.get("id") or stable_id(row["url"], row.get("title", ""))
    return rows


def collect(
    path: Path,
    target_total: int,
    target_per_category: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_existing(path)
    seen_urls = {row["url"] for row in rows if row["url"]}
    seen_titles = {normalize_title(row["title"]) for row in rows}
    errors = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    for source, category, url in FEEDS:
        if len(rows) >= target_total:
            break
        normalized_category = category_name(category)
        if counts.get(normalized_category, 0) >= target_per_category:
            continue
        try:
            parsed = feedparser.parse(url)
            if getattr(parsed, "bozo", False) and not parsed.entries:
                raise RuntimeError(str(getattr(parsed, "bozo_exception", "invalid RSS")))
            for entry in parsed.entries:
                title = clean_html(entry.get("title", ""))
                link = canonical_url(entry.get("link", ""))
                description = entry_description(entry)
                title_key = normalize_title(title)
                if not title or not link or link in seen_urls or title_key in seen_titles:
                    continue
                rows.append({
                    "id": stable_id(link, title),
                    "title": title,
                    "description": description,
                    "source": source,
                    "category": category_name(category),
                    "date": entry_date(entry),
                    "url": link,
                    "retrieved_at": retrieved_at,
                })
                seen_urls.add(link)
                seen_titles.add(title_key)
                counts[normalized_category] = counts.get(normalized_category, 0) + 1
                if len(rows) >= target_total:
                    break
                if counts[normalized_category] >= target_per_category:
                    break
        except Exception as exc:
            errors.append({"source": source, "category": category, "url": url, "error": str(exc)})
    return rows, errors


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-total", type=int, default=600)
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=200,
        help="Maximum number collected per final category; use a balanced target.",
    )
    args = parser.parse_args()
    if args.target_total < 1 or args.target_per_category < 1:
        raise ValueError("Targets must be positive")
    rows, errors = collect(args.output, args.target_total, args.target_per_category)
    write_rows(args.output, rows)
    if errors:
        write_rows(ERROR_LOG, errors)
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"Corpus: {len(rows)} noticias")
    print("Categorías:", counts)
    if len(rows) < args.target_total:
        print(f"Advertencia: RSS disponible produjo {len(rows)} de {args.target_total}.")


if __name__ == "__main__":
    main()
