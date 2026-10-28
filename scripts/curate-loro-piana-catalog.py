#!/usr/bin/env python3
"""Curate a varied Loro Piana men's apparel feed from verified FR flat shots."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DOWNLOADER = Path.home() / "startups" / "website-downloader"
sys.path.insert(0, str(DOWNLOADER))
from no_model import has_person_bgr  # noqa: E402
import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/white_bg_products/loropiana_mens/all_products.json")
OUTPUT = ROOT / "data" / "shop_catalog_loro_piana.json"
EXCLUDED_PATHS = re.compile(r"accessor|shoe|small-leather|leather-goods|hat|scarf|sock|tie|belt|bag|wallet|sunglass", re.I)
BOTTOMS = re.compile(r"trouser|pant|bermuda|jean|short", re.I)
ACCESSORY_NAME = re.compile(r"\b(socks?|scarves?|ties?|belts?|wallets?|bags?|caps?|hats?|gloves?|sunglasses|pouches?)\b", re.I)
MAX_UPPER = 125
MAX_BOTTOMS = 25
MAX_PRICE_EUR = 3500


def main():
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    styles = {}
    for item in rows:
        url = str(item.get("url") or "")
        categories = " ".join(str(x) for x in (item.get("categories") or []))
        name = str(item.get("name") or "").strip()
        if not item.get("kept") or "/en/man/" not in url or not item.get("fr_image"):
            continue
        if EXCLUDED_PATHS.search(url) or EXCLUDED_PATHS.search(categories) or ACCESSORY_NAME.search(name):
            continue
        price = float(item.get("price") or 0)
        if not price or price > MAX_PRICE_EUR:
            continue
        category, garment_type = ("bottoms", "lower_body") if BOTTOMS.search(url + " " + name) else ("tops", "upper_body")
        style_id = str(item.get("id") or "").split("_")[0]
        if not style_id:
            continue
        # Keep one color per style to favor garment variety over repeated color rows.
        candidate = {**item, "_category": category, "_garment_type": garment_type}
        prior = styles.get(style_id)
        if prior is None or (str(candidate.get("color") or "").lower() in {"white", "ivory", "beige", "grey", "gray", "navy", "black", "brown"} and str(prior.get("color") or "").lower() not in {"white", "ivory", "beige", "grey", "gray", "navy", "black", "brown"}):
            styles[style_id] = candidate

    groups = defaultdict(list)
    for item in styles.values():
        url_path = str(item["url"]).split("/en/man/", 1)[-1]
        group = url_path.split("/", 1)[0]
        groups[(item["_category"], group)].append(item)
    for group in groups:
        groups[group].sort(key=lambda item: (str(item.get("name") or "").lower(), str(item.get("id"))))

    def balanced(category, limit):
        group_names = sorted(name for (kind, name) in groups if kind == category)
        result = []
        while len(result) < limit:
            advanced = False
            for group in group_names:
                bucket = groups[(category, group)]
                if bucket:
                    result.append(bucket.pop(0))
                    advanced = True
                    if len(result) == limit:
                        break
            if not advanced:
                break
        return result

    selected = balanced("tops", MAX_UPPER) + balanced("bottoms", MAX_BOTTOMS)
    accepted = []
    rejected = Counter()
    for product in selected:
        image_path = Path("/tmp/white_bg_products/loropiana_mens/images") / f"{product['id']}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            rejected["missing image"] += 1
            continue
        if has_person_bgr(image):
            rejected["person detected"] += 1
            continue
        display_name = str(product.get("name") or "").title()
        # A product model named "Umbrella Pine" is a sweatshirt, but the
        # shared shop-catalog trigger classifies any unrelated "umbrella"
        # mention as an accessory. Keep the garment name clear for the app.
        if display_name.lower().startswith("umbrella pine hooded sweatshirt"):
            display_name = display_name.replace("Umbrella ", "", 1)
        accepted.append({
            "id": "loro-piana-" + str(product["id"]).lower(),
            "brand": "Loro Piana",
            "name": display_name,
            "price": float(product["price"]),
            "currency": "EUR",
            "category": product["_category"],
            "garment_type": product["_garment_type"],
            "description": "Loro Piana · men's apparel · official front-view product photo on a plain background",
            "image_url": product["fr_image"],
            "source": "loro-piana-white-bg-no-model",
            "source_url": product["url"],
            "is_active": True,
            "sort_order": 0,
        })
    accepted.sort(key=lambda item: (item["category"], item["name"], item["id"]))
    for i, item in enumerate(accepted):
        item["sort_order"] = i
    if not accepted:
        raise SystemExit("No Loro Piana apparel passed the background and person checks.")
    OUTPUT.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Curated {len(accepted)} Loro Piana men's apparel items -> {OUTPUT}")
    print("Categories:", dict(Counter(item["category"] for item in accepted)))
    print("Rejected during person check:", dict(rejected))


if __name__ == "__main__":
    main()
