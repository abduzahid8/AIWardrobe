#!/usr/bin/env python3
"""Curate Armani Exchange, Giorgio Armani, and EA7 flat apparel photos."""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

DOWNLOADER = Path.home() / "startups" / "website-downloader"
sys.path.insert(0, str(DOWNLOADER))
from no_model import has_person_bgr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/armani_all_mens/all_products.json")
OUTPUTS = {
    "GIORGIO ARMANI": ("giorgio_armani", "Giorgio Armani"),
    "ARMANI EXCHANGE": ("armani_exchange", "Armani Exchange"),
    "EA7": ("ea7", "EA7 Emporio Armani"),
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
           "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}
TOPS = {"T-shirts", "Knitwear", "Outerwear", "Shirts", "Jackets",
        "Sweatshirts", "Suits", "Leather", "Co-ords"}
BOTTOMS = {"Trousers", "Jeans", "Swimwear"}
NON_APPAREL = re.compile(r"\b(?:beach\s+)?towels?\b", re.I)
MAX_PER_MACRO_CATEGORY = 18
MAX_ITEMS_PER_BRAND = 65
WHITE_BG_THRESHOLD = 215


def category_for(item):
    macro = str(item.get("macroCat") or "").strip()
    if macro in TOPS:
        return "tops", "upper_body"
    if macro in BOTTOMS:
        return "bottoms", "lower_body"
    return None


def front_image(item):
    for url in item.get("imgs") or []:
        if re.search(r"_F_(?:FW|SS)\d{4}", url, re.I):
            return url
    return None


def style_key(item, image_url):
    parts = Path(urlparse(image_url).path).stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2])
    return str(item.get("model") or item.get("id") or image_url)


def is_plain_background(image):
    image = image.convert("RGB")
    w, h = image.size
    if w < 80 or h < 80:
        return False
    points = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3),
              (10, 10), (w - 11, 10), (10, h - 11), (w - 11, h - 11)]
    edge = []
    for x in range(0, w, max(1, w // 16)):
        edge.extend((image.getpixel((x, 0)), image.getpixel((x, h - 1))))
    for y in range(0, h, max(1, h // 16)):
        edge.extend((image.getpixel((0, y)), image.getpixel((w - 1, y))))
    samples = [image.getpixel(p) for p in points] + edge
    return sum(sum(px) / 3 for px in samples) / len(samples) >= WHITE_BG_THRESHOLD


def check_image(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200 or len(response.content) < 8000:
            return False
        image = Image.open(BytesIO(response.content)).convert("RGB")
        if not is_plain_background(image):
            return False
        bgr = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return bgr is not None and not has_person_bgr(bgr)
    except Exception:
        return False


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Run armani_all_mens.py --metadata-only first; missing {SOURCE}")
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidate_buckets = defaultdict(list)
    seen = set()
    for item in rows:
        brand = str(item.get("brand") or "").upper().strip()
        if brand not in OUTPUTS:
            continue
        cat = category_for(item)
        if not cat or not item.get("hasStock") or not item.get("price"):
            continue
        image_url = front_image(item)
        name = str(item.get("name") or "").strip()
        if not name or NON_APPAREL.search(name) or not image_url:
            continue
        key = (brand, style_key(item, image_url))
        if key in seen:
            continue
        seen.add(key)
        macro = str(item.get("macroCat") or "")
        candidate_buckets[(brand, cat[0], macro)].append((item, cat, image_url))

    for bucket in candidate_buckets.values():
        bucket.sort(key=lambda row: (str(row[0].get("name") or "").lower(),
                                     str(row[0].get("id") or "")))

    selected = defaultdict(list)
    for brand in OUTPUTS:
        buckets = sorted(key for key in candidate_buckets if key[0] == brand)
        counts = Counter()
        while len(selected[brand]) < MAX_ITEMS_PER_BRAND:
            advanced = False
            for bucket_key in buckets:
                bucket = candidate_buckets[bucket_key]
                if bucket and counts[bucket_key[2]] < MAX_PER_MACRO_CATEGORY:
                    selected[brand].append(bucket.pop(0))
                    counts[bucket_key[2]] += 1
                    advanced = True
                    if len(selected[brand]) >= MAX_ITEMS_PER_BRAND:
                        break
            if not advanced:
                break

    for source_brand, (key, label) in OUTPUTS.items():
        accepted = []
        rejected = Counter()
        for item, (category, garment_type), image_url in selected[source_brand]:
            if not check_image(image_url):
                rejected["background/person/image check"] += 1
                continue
            product_id = re.sub(r"[^a-zA-Z0-9_-]", "-", str(item.get("id") or "")).lower()
            accepted.append({
                "id": f"{key}-{product_id}",
                "brand": label,
                "name": str(item["name"]).strip(),
                "price": float(item["price"]),
                "currency": "USD",
                "category": category,
                "garment_type": garment_type,
                "description": f"{label} · men's apparel · front-view product photo on a plain background",
                "image_url": image_url,
                "source": f"{key.replace('_', '-')}-us-white-bg-no-model",
                "source_url": "https://www.armani.com/en-us/",
                "is_active": True,
                "sort_order": 0,
            })
            if len(accepted) >= MAX_ITEMS_PER_BRAND:
                break
            time.sleep(0.05)
        if not accepted:
            print(f"{label}: no apparel images passed the plain-background/no-person checks; rejected {sum(rejected.values())}")
            continue
        accepted.sort(key=lambda row: (row["category"], row["name"].lower(), row["id"]))
        for index, item in enumerate(accepted):
            item["sort_order"] = index
        output = ROOT / "data" / f"shop_catalog_{key}.json"
        output.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = Counter(item["category"] for item in accepted)
        print(f"{label}: {len(accepted)} kept ({dict(counts)}); {sum(rejected.values())} rejected -> {output}")


if __name__ == "__main__":
    main()
