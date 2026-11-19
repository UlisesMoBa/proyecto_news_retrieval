"""Prepare the extensible raw corpus for Stage I and Stage II."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "news_raw_corpus.tsv"
PROCESSED = ROOT / "data" / "processed"
NORMALIZED = PROCESSED / "news_normalized_corpus.tsv"
LABELED = PROCESSED / "labeled_documents.tsv"
SPLIT = PROCESSED / "labeled_documents_split.tsv"


def clean(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def category(value: object) -> str:
    text = str(value).strip()
    if text in {"Economía", "Economia", "Negocios"}:
        return "Negocios"
    if text in {"Ciencia", "Ciencias"}:
        return "Ciencia"
    if text == "Cultura":
        return "Cultura"
    raise ValueError(f"Categoría no permitida: {text}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=200,
        help="Maximum number of processed documents per category.",
    )
    args = parser.parse_args()
    if args.max_per_category < 2:
        raise ValueError("--max-per-category must be at least 2")
    if not RAW.exists():
        raise FileNotFoundError(f"No existe el corpus crudo: {RAW}")
    df = pd.read_csv(RAW, sep="\t", dtype=str, keep_default_na=False)
    required = {"id", "title", "description", "source", "category", "date", "url", "retrieved_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el corpus crudo: {sorted(missing)}")
    df["category"] = df["category"].map(category)
    df["title"] = df["title"].map(clean)
    df["description"] = df["description"].map(clean)
    df["url_key"] = df["url"].str.lower().str.replace(r"[?#].*$", "", regex=True).str.rstrip("/")
    df["title_key"] = df["title"].str.replace(r"\W+", " ", regex=True).str.strip()
    df = df[(df["title"] != "") & (df["description"] != "")]
    df = df.drop_duplicates(subset=["url_key"], keep="first")
    df = df.drop_duplicates(subset=["title_key"], keep="first")
    # Keep a reproducible balanced subset for classification and unsupervised
    # learning while retaining every raw record for future expansion.
    df = (
        df.sort_values(["category", "id"])
        .groupby("category", group_keys=False)
        .head(args.max_per_category)
        .reset_index(drop=True)
    )
    df["text"] = (df["title"] + " " + df["description"]).str.strip()
    df["normalized_text"] = df["text"].map(clean)
    df = df.drop(columns=["url_key", "title_key"])
    df = df.sort_values("id").reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORMALIZED, sep="\t", index=False, encoding="utf-8")
    df[["id", "text", "category", "source", "date", "url", "retrieved_at"]].rename(
        columns={"id": "document_id"}
    ).to_csv(LABELED, sep="\t", index=False, encoding="utf-8")

    train, remainder = train_test_split(
        df, test_size=0.30, random_state=0, shuffle=True, stratify=df["category"]
    )
    dev, test = train_test_split(
        remainder, test_size=0.50, random_state=0, shuffle=True, stratify=remainder["category"]
    )
    train = train.assign(split="train")
    dev = dev.assign(split="dev")
    test = test.assign(split="test")
    split = pd.concat([train, dev, test], ignore_index=True).sort_values("id")
    split[["id", "text", "category", "split", "source", "date", "url", "retrieved_at"]].rename(
        columns={"id": "document_id"}
    ).to_csv(SPLIT, sep="\t", index=False, encoding="utf-8")
    print(f"Noticias procesadas: {len(df)}")
    print(df["category"].value_counts().sort_index().to_string())
    print(split["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
