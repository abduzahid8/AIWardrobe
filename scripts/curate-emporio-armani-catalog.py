#!/usr/bin/env python3
"""Build a varied Emporio Armani men's apparel feed from isolated front shots."""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

DOWNLOADER = Path.home() / "startups" / "website-downloader"
sys.path.insert(0, str(DOWNLOADER))
from no_model import has_person_bgr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/armani_mens/all_products.json")
OUTPUT = ROOT / "data" / "shop_catalog_emporio_armani.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
           "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}
MAX_ITEMS = 140
WHITE_BG_THRESHOLD = 215
MAX_PER_CATEGORY = 20

TOPS = {"T-shirts", "Knitwear", "Outerwear", "Shirts", "Jackets",
        "Sweatshirts", "Suits", "Leather", "Co-ords"}
BOTTOMS = {"Trousers", "Jeans", "Swimwear"}


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
    filename = Path(urlparse(image_url).path).stem
    parts = filename.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2])
    return str(item.get("model") or item.get("id") or filename)


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
        response = requests.get(url, headers=HEADERS, timeout=20)
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
        raise SystemExit(f"Run the downloader first; missing {SOURCE}")
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    categories = defaultdict(list)
    seen_styles = set()
    for item in rows:
        cat = category_for(item)
        if not cat or not item.get("hasStock") or not item.get("price"):
            continue
        name = str(item.get("name") or "").strip()
        image_url = front_image(item)
        if not name or not image_url:
            continue
        key = (cat[0], style_key(item, image_url))
        if key in seen_styles:
            continue
        seen_styles.add(key)
        categories[(cat[0], str(item.get("macroCat")))].append((item, cat, image_url))

    for bucket in categories.values():
        bucket.sort(key=lambda row: (str(row[0].get("name") or "").lower(),
                                     str(row[0].get("id") or "")))

    # Round-robin clothing categories so the feed is varied instead of one large tee run.
    selected = []
    names = sorted(categories)
    counts = Counter()
    while len(selected) < MAX_ITEMS:
        advanced = False
        for name in names:
            bucket = categories[name]
            if bucket and counts[name] < MAX_PER_CATEGORY:
                selected.append(bucket.pop(0))
                counts[name] += 1
                advanced = True
                if len(selected) >= MAX_ITEMS:
                    break
        if not advanced:
            break

    accepted = []
    rejected = Counter()
    for item, (category, garment_type), image_url in selected:
        if not check_image(image_url):
            rejected["background/person/image check"] += 1
            continue
        price = float(item["price"])
        accepted.append({
            "id": "emporio-armani-" + re.sub(r"[^a-zA-Z0-9_-]", "-", str(item["id"])).lower(),
            "brand": "Emporio Armani",
            "name": str(item["name"]).strip(),
            "price": price,
            "currency": "USD",
            "category": category,
            "garment_type": garment_type,
            "description": "Emporio Armani · men's apparel · front-view product photo on a plain background",
            "image_url": image_url,
            "source": "emporio-armani-us-white-bg-no-model",
            "source_url": "https://www.armani.com/en-us/",
            "is_active": True,
            "sort_order": 0,
        })
        if len(accepted) >= MAX_ITEMS:
            break
        if len(accepted) % 25 == 0:
            print(f"  accepted {len(accepted)}")
        time.sleep(0.1)

    if not accepted:
        raise SystemExit("No Emporio Armani apparel passed the plain-background and no-person checks.")
    accepted.sort(key=lambda row: (row["category"], row["name"].lower(), row["id"]))
    for index, item in enumerate(accepted):
        item["sort_order"] = index
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Curated {len(accepted)} Emporio Armani items -> {OUTPUT}")
    print("Categories:", dict(Counter(item["category"] for item in accepted)))
    print("Rejected:", dict(rejected))


if __name__ == "__main__":
    main()
