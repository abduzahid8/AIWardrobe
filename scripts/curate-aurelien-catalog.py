#!/usr/bin/env python3
"""Curate the Aurelien men's catalog to confirmed flat, model-free apparel."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

DOWNLOADER = Path.home() / "startups" / "website-downloader"
sys.path.insert(0, str(DOWNLOADER))
from no_model import has_person_bgr  # noqa: E402
from aurelien_mens import safe_name  # noqa: E402
import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/white_bg_products/aurelien_mens/all_products.json")
OUTPUT = ROOT / "data" / "shop_catalog_aurelien.json"
LOWER_BODY = re.compile(r"\b(?:trousers?|pants?|jeans?|bermudas?|shorts?|swimshorts?)\b", re.I)


def main():
    products = json.loads(SOURCE.read_text(encoding="utf-8"))
    kept = []
    rejected = Counter()
    for product in products:
        tags = {str(tag).lower() for tag in product.get("tags", [])}
        if "__men" not in tags or "__clothing" not in tags or not product.get("kept"):
            continue
        image_path = Path("/tmp/white_bg_products/aurelien_mens/images") / (safe_name(product.get("handle") or str(product.get("id"))) + ".jpg")
        image = cv2.imread(str(image_path))
        if image is None:
            rejected["missing image"] += 1
            continue
        if has_person_bgr(image):
            rejected["person detected"] += 1
            continue

        name = str(product.get("name") or "").strip()
        if not name or not product.get("featured_image") or not product.get("url"):
            rejected["missing product metadata"] += 1
            continue
        price = float(product["price"])
        category, garment_type = ("bottoms", "lower_body") if LOWER_BODY.search(name + " " + str(product.get("product_type"))) else ("tops", "upper_body")
        kept.append({
            "id": f"aurelien-{product['id']}",
            "brand": "Aurélien",
            "name": name,
            "price": price,
            "currency": "EUR",
            "category": category,
            "garment_type": garment_type,
            "description": "Aurélien · men's apparel · official flat product photo on a plain background",
            "image_url": product["featured_image"],
            "source": "aurelien-white-bg-no-model",
            "source_url": product["url"],
            "is_active": True,
            "sort_order": 0,
        })
    kept.sort(key=lambda item: (item["category"], item["name"], item["id"]))
    for i, item in enumerate(kept):
        item["sort_order"] = i
    if not kept:
        raise SystemExit("No Aurelien items passed the background, gender, clothing, and person checks.")
    OUTPUT.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Curated {len(kept)} Aurélien men's apparel items -> {OUTPUT}")
    print("Categories:", dict(Counter(item["category"] for item in kept)))
    print("Rejected during person/metadata check:", dict(rejected))


if __name__ == "__main__":
    main()
