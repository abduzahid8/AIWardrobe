#!/usr/bin/env python3
"""Normalize accepted GANT men's product photos from website-downloader.

The downloader output is already filtered for a plain bright background and
no visible person. This step uses the official product URL as the canonical
name/category source because some GANT image alt text describes a campaign
photo even when the selected flat product image is correct.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path.home() / "startups" / "website-downloader" / "shop_catalog_uz_gant.json"
DEFAULT_OUTPUT = ROOT / "data" / "shop_catalog_uz_gant.json"

NON_CLOTHING = re.compile(
    r"\b(socks?|bags?|accessor(?:y|ies)|wallets?|belts?|underwear|caps?|"
    r"scarves?|ties?|gloves?|boxers?|sunglasses|watches?)\b",
    re.IGNORECASE,
)
BOTTOMS = re.compile(
    r"\b(jeans?|pants?|trousers?|chinos?|shorts?|swim[- ]?(shorts?|trunks?))\b",
    re.IGNORECASE,
)
SHOES = re.compile(
    r"\b(shoes?|boots?|loafers?|sneakers?|trainers?|sandals?|slippers?)\b",
    re.IGNORECASE,
)


def title_from_slug(slug: str) -> str:
    return " ".join(word[:1].upper() + word[1:] for word in re.sub(r"[-_]+", " ", slug).split())


def curate(raw_items):
    curated = []
    for raw in raw_items:
        source_url = str(raw.get("source_url") or "")
        match = re.search(r"/p/([^/]+)/", source_url)
        if not match or "gant.com" not in source_url:
            continue
        slug = match.group(1)
        if NON_CLOTHING.search(slug):
            continue
        image_url = str(raw.get("image_url") or "")
        if "-flat-" not in image_url:
            continue

        if SHOES.search(slug):
            category, garment_type = "shoes", "shoes"
        elif BOTTOMS.search(slug):
            category, garment_type = "bottoms", "lower_body"
        else:
            category, garment_type = "tops", "upper_body"

        curated.append({
            "id": str(raw["id"]),
            "brand": "GANT",
            "name": title_from_slug(slug),
            "price": float(raw["price"]),
            "currency": "UZS",
            "category": category,
            "garment_type": garment_type,
            "description": "GANT · men's collection · official flat product photo on a plain background · priced in UZS",
            "image_url": image_url,
            "source": str(raw.get("source") or "gant-white-bg"),
            "source_url": source_url,
            "is_active": True,
            "sort_order": len(curated),
        })
    return curated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_items = json.loads(args.input.read_text(encoding="utf-8"))
    items = curate(raw_items)
    if not items:
        raise SystemExit("No valid GANT men's items found; output was not changed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Curated {len(items)} GANT men's items -> {args.output}")
    print("Categories:", dict(Counter(item["category"] for item in items)))


if __name__ == "__main__":
    main()
