#!/usr/bin/env python3
"""Vendor white-bg product shots from the website-downloader project into AIWardrobe.

Source (website-downloader output):
    /tmp/white_bg_products/massimo_mens/  (all_products.json + images/*.jpg)
    /tmp/white_bg_products/zara_mens/     (all_products.json + images/*.jpg)

Destination (this repo — STAGING ONLY, never bundled into the app):
    assets/shop/massimo/*.jpg
    assets/shop/zara/*.jpg
    assets/shop/manifest.json              (metadata for tooling / future uploads)

IMPORTANT: shop images must NOT be bundled into the app binary (they were
1.4GB via require() + assetBundlePatterns). After vendoring, upload to
Supabase with:
    node scripts/upload-shop-images-to-supabase.js
then delete assets/shop/ locally. The app reads the catalog from the
Supabase `shop_catalog` table (see hooks/useShopCatalog.ts).

Only products that passed the white-bg filter AND have a local image file on
disk are vendored. Re-running is safe (idempotent): existing files are skipped
unless the source is newer.

Usage:
    python3 scripts/vendor-shop-images.py
"""

import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = "/tmp/white_bg_products"

BRANDS = {
    "massimo": {
        "brand": "Massimo Dutti",
        "src_json": os.path.join(SRC_ROOT, "massimo_mens", "all_products.json"),
        "src_img_dir": os.path.join(SRC_ROOT, "massimo_mens", "images"),
        "dst_img_dir": os.path.join(REPO, "assets", "shop", "massimo"),
        "id_prefix": "md-",
        "currency": "EUR",
    },
    "zara": {
        "brand": "ZARA",
        "src_json": os.path.join(SRC_ROOT, "zara_mens", "all_products.json"),
        "src_img_dir": os.path.join(SRC_ROOT, "zara_mens", "images"),
        "dst_img_dir": os.path.join(REPO, "assets", "shop", "zara"),
        "id_prefix": "zara-",
        "currency": "USD",
    },
}

SHOES_KW = [
    "shoe", "sneaker", "boot", "loafer", "espadrille", "sandal", "derby",
    "oxford", "moccasin", "moc toe", "driving", "slipper", "flip-flop",
]
ACCESSORY_KW = [
    "bag", "belt", "tie", "scarf", "hat", "cap", "beanie", "wallet",
    "backpack", "tote", "briefcase", "sunglass", "glasses", "perfume",
    "cologne", "towel", "watch", "jewel", "cufflink", "pocket square",
    "socks", "underwear", "bra", "bikini", "kerchief",
]
LOWER_KW = [
    "trouser", "jean", "pant", "short", "bermuda", "trunk", "swim",
    "jogger", "chino", "cargo", "culotte", "skirt",
]


def classify_garment(name: str, hint: str) -> str:
    text = f"{hint or ''} {name or ''}".lower()
    for kw in SHOES_KW:
        if kw in text:
            return "shoes"
    for kw in ACCESSORY_KW:
        if kw in text:
            return "accessory"
    for kw in LOWER_KW:
        if kw in text:
            return "lower_body"
    return "upper_body"


def parse_price(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"\d+[.,]\d{2}", value.replace(",", "."))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
        m = re.search(r"\d+", value)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
    return 0.0


def ts_escape(value: str) -> str:
    return (
        (value or "")
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def ts_truncate(value: str, limit: int) -> str:
    value = value or ""
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def load_products(cfg: dict) -> list:
    with open(cfg["src_json"], encoding="utf-8") as f:
        return json.load(f)


def normalize(key: str, raw: dict, cfg: dict) -> Optional[dict]:
    pid = str(raw.get("id", "")).strip()
    if not pid:
        return None
    src_file = os.path.join(cfg["src_img_dir"], f"{pid}.jpg")
    if not os.path.isfile(src_file):
        return None
    # Massimo entries predate the `kept` flag (all passed); Zara uses it.
    if raw.get("kept") is False:
        return None

    name = ts_truncate(str(raw.get("name", "")).strip() or pid, 120)
    if key == "massimo":
        hint = str(raw.get("category", "") or "")
        price = parse_price(raw.get("price_eur"))
    else:
        hint = str(raw.get("search_term", "") or "")
        price = parse_price(raw.get("price"))
    hint = ts_truncate(hint, 80)
    description = ts_truncate(
        f"{cfg['brand']} · {hint} · white-bg product shot".strip(" ·"), 220
    )
    return {
        "id": f"{cfg['id_prefix']}{pid}",
        "file": f"{pid}.jpg",
        "brand": cfg["brand"],
        "name": ts_escape(name),
        "price": price,
        "currency": cfg["currency"],
        "garmentType": classify_garment(name, hint),
        "description": ts_escape(description),
        "sourceUrl": ts_escape(str(raw.get("product_url", "") or "")),
    }


def copy_images(items: list, cfg: dict) -> int:
    os.makedirs(cfg["dst_img_dir"], exist_ok=True)
    copied = 0
    for item in items:
        pid_file = item["file"]
        src = os.path.join(cfg["src_img_dir"], pid_file)
        dst = os.path.join(cfg["dst_img_dir"], pid_file)
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> int:
    manifest: list = []
    summary: dict = {}
    for key, cfg in BRANDS.items():
        if not os.path.isfile(cfg["src_json"]):
            print(f"[{key}] SKIP — source JSON not found: {cfg['src_json']}")
            continue
        raw_products = load_products(cfg)
        items = []
        for raw in raw_products:
            item = normalize(key, raw, cfg)
            if item:
                items.append(item)
        copied = copy_images(items, cfg)
        # NOTE: no TS module generation — require() images would bundle all
        # 1.4GB into the app binary. Upload via
        # scripts/upload-shop-images-to-supabase.js instead.
        manifest.extend(
            [
                {**item, "image": f"assets/shop/{key}/{item['file']}"}
                for item in items
            ]
        )
        summary[key] = (len(items), copied)
        print(f"[{key}] {len(items)} items vendored ({copied} copied)")

    os.makedirs(os.path.join(REPO, "assets", "shop"), exist_ok=True)
    manifest_path = os.path.join(REPO, "assets", "shop", "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"manifest: {manifest_path} ({len(manifest)} entries)")
    print(f"summary: {summary}")
    print("NEXT: node scripts/upload-shop-images-to-supabase.js")
    return 0


if __name__ == "__main__":
    sys.exit(main())
