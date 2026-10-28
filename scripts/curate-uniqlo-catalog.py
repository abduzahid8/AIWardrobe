#!/usr/bin/env python3
"""Keep only men's UNIQLO clothing photos with plain backgrounds and no model."""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

DOWNLOADER = Path.home() / "startups" / "website-downloader"
sys.path.insert(0, str(DOWNLOADER))
from no_model import has_person_bgr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "uniqlo-catalog-entries.json"
PRODUCTS = ROOT / "data" / "uniqlo-products.json"
OUTPUT = ROOT / "data" / "shop_catalog_uniqlo.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
           "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}
WHITE_BG_THRESHOLD = 215
TOP_CATEGORIES = {"tops", "outerwear", "shirts & polos", "polos",
                  "sweatshirts & hoodies", "sweaters", "t-shirts"}
BOTTOM_CATEGORIES = {"bottoms", "shorts", "jeans"}


def plain_background(image):
    image = image.convert("RGB")
    w, h = image.size
    if w < 80 or h < 80:
        return False
    points = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3),
              (10, 10), (w - 11, 10), (10, h - 11), (w - 11, h - 11)]
    edges = []
    for x in range(0, w, max(1, w // 16)):
        edges.extend((image.getpixel((x, 0)), image.getpixel((x, h - 1))))
    for y in range(0, h, max(1, h // 16)):
        edges.extend((image.getpixel((0, y)), image.getpixel((w - 1, y))))
    samples = [image.getpixel(p) for p in points] + edges
    return sum(sum(px) / 3 for px in samples) / len(samples) >= WHITE_BG_THRESHOLD


def check(row):
    try:
        response = requests.get(row["item"]["imageUrl"], headers=HEADERS, timeout=15)
        if response.status_code != 200 or len(response.content) < 7000:
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
        if not plain_background(image):
            return None
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if has_person_bgr(bgr):
            return None
        item = row["item"]
        category = row["category"]
        return {
            "id": item["id"],
            "brand": "UNIQLO",
            "name": item["name"],
            "price": float(item["price"]),
            "currency": "USD",
            "category": "bottoms" if category == "bottoms" else "tops",
            "garment_type": item.get("garmentType") or ("lower_body" if category == "bottoms" else "upper_body"),
            "description": "UNIQLO · men's apparel · product-only photo on a plain background",
            "image_url": item["imageUrl"],
            "source": "uniqlo-us-white-bg-no-model",
            "source_url": row["source_url"],
            "is_active": True,
            "sort_order": 0,
        }
    except Exception:
        return None


def main():
    products = json.loads(PRODUCTS.read_text(encoding="utf-8"))
    by_id = {str(row.get("productId")): row for row in products}
    entries = json.loads(INPUT.read_text(encoding="utf-8"))
    candidates = []
    for item in entries:
        match = re.search(r"/imagesgoods/(\d+)/", str(item.get("imageUrl") or ""))
        source = by_id.get(match.group(1)) if match else None
        if not source or not source.get("productUrl"):
            continue
        category = str(source.get("category") or "").lower()
        if category not in TOP_CATEGORIES | BOTTOM_CATEGORIES:
            continue
        candidates.append({"item": item,
                           "category": "bottoms" if category in BOTTOM_CATEGORIES else "tops",
                           "source_url": source["productUrl"]})

    with ThreadPoolExecutor(max_workers=6) as pool:
        accepted = [item for item in pool.map(check, candidates) if item]
    accepted.sort(key=lambda item: (item["category"], item["name"].lower(), item["id"]))
    for index, item in enumerate(accepted):
        item["sort_order"] = index
    if not accepted:
        raise SystemExit("No UNIQLO apparel passed the plain-background/no-person checks.")
    OUTPUT.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {len(candidates)} apparel images; kept {len(accepted)} UNIQLO items -> {OUTPUT}")
    print("Categories:", dict(Counter(item["category"] for item in accepted)))


if __name__ == "__main__":
    main()
