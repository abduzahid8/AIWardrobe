#!/usr/bin/env python3
"""Offline, owner-authorized Instagram image import for AIWardrobe Style References.

Never fetch Instagram or accept account login credentials. Reads an export ZIP
or unpacked export directory and produces a private local image corpus plus a
review manifest. Requires per-asset curation BEFORE usage as positive fashion
training/reference examples. No uploads, DB writes, or production changes.

Example:
  python3 scripts/style_references/import_instagram_archive.py \
      --source /secure/sainttnoirclub-export.zip \
      --output /secure/aiwardrobe-reference-staging \
      --account sainttnoirclub --owner-authorized
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_MEDIA_BYTES = 35 * 1024 * 1024
MAX_IMAGES = 40_000
MAX_TOTAL_UNCOMPRESSED = 8 * 1024 * 1024 * 1024
ACCOUNT_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")


def safe_entry_name(raw: str) -> str | None:
    name = raw.replace("\\", "/")
    p = PurePosixPath(name)
    if (not name or name.startswith("/") or ".." in p.parts or
            any(part in ("", ".") for part in p.parts) or
            any(":" in part for part in p.parts)):
        return None
    return p.as_posix()


def is_post_media(name: str) -> bool:
    parts = [p.lower() for p in PurePosixPath(name).parts]
    if "stories" in parts or "messages" in parts or "direct" in parts:
        return False
    # Modern Meta export layout: media/posts; older: posts/media; archive root
    return "posts" in parts or "posts_1" in parts or "feed" in parts


def magic_ok(data: bytes, suffix: str) -> bool:
    if suffix in (".jpg", ".jpeg"):
        return data.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if suffix in (".heic", ".heif"):
        return len(data) > 12 and data[4:8] == b"ftyp" and data[8:12] in (
            b"heic", b"heix", b"hevc", b"mif1", b"msf1", b"heim")
    return False


def captions_from_json(raw_files: dict[str, bytes]) -> dict[str, dict]:
    """Metadata paths in official exports vary; exact path matches only."""
    entries = {}
    for filename, blob in raw_files.items():
        if not filename.lower().endswith(".json"):
            continue
        if "posts" not in filename.lower() or len(blob) > 25_000_000:
            continue
        try:
            parsed = json.loads(blob.decode("utf-8-sig"))
        except (ValueError, UnicodeError):
            continue
        arrays = []
        if isinstance(parsed, list):
            arrays = [parsed]
        elif isinstance(parsed, dict):
            arrays = [v for v in parsed.values() if isinstance(v, list)]
        for arr in arrays:
            for post in arr:
                if not isinstance(post, dict):
                    continue
                media = post.get("media") or []
                if not isinstance(media, list):
                    continue
                for m in media:
                    if not isinstance(m, dict) or not isinstance(m.get("uri"), str):
                        continue
                    key = safe_entry_name(m["uri"])
                    if key:
                        entries[key] = {
                            "caption": str(m.get("title") or post.get("title") or "")[:800],
                            "creation_timestamp": m.get("creation_timestamp"),
                        }
    return entries


def load_archive(source: Path):
    """Produce (filename, bytes) without ever extracting arbitrary ZIP paths."""
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if not file.is_file() or file.is_symlink():
                continue
            path = safe_entry_name(file.relative_to(source).as_posix())
            if not path:
                continue
            if file.stat().st_size > MAX_MEDIA_BYTES and not path.endswith(".json"):
                continue
            yield path, file.read_bytes()
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            total = 0
            if len(z.infolist()) > 100_000:
                raise ValueError("Archive has too many members")
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = safe_entry_name(info.filename)
                if not name:
                    raise ValueError("Unsafe archive path rejected")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlink rejected")
                total += info.file_size
                if total > MAX_TOTAL_UNCOMPRESSED:
                    raise ValueError("Archive too large when decompressed")
                if info.file_size > MAX_MEDIA_BYTES and not name.endswith(".json"):
                    continue
                if info.file_size > 25_000_000 and name.endswith(".json"):
                    continue
                yield name, z.read(info)
    else:
        raise ValueError("Input must be an Instagram export ZIP or directory")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="Official Instagram export ZIP or unpacked directory")
    ap.add_argument("--output", type=Path, required=True, help="Private local directory, outside Git checkout")
    ap.add_argument("--account", required=True, help="Exact expected Instagram handle")
    ap.add_argument("--owner-authorized", action="store_true", help="Affirm creator has authorized storage and ML/reference use")
    args = ap.parse_args(argv)
    if not args.owner_authorized:
        ap.error("Explicit --owner-authorized acknowledgement required")
    handle = args.account.lstrip("@").lower()
    if not ACCOUNT_RE.fullmatch(handle):
        ap.error("Invalid Instagram account handle")
    src = args.source.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not src.exists():
        ap.error("Archive not found")
    if out == src or out.is_relative_to(src) or src.is_relative_to(out):
        ap.error("Source and output paths must not overlap")
    if any(part == ".git" for part in out.parts):
        ap.error("Do not save source images into Git internals")
    if out.exists() and any(out.iterdir()):
        ap.error("Output directory must be empty for reproducible import")
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)

    blobs = {}
    images = []
    for name, data in load_archive(src):
        ext = Path(name).suffix.lower()
        if ext in ALLOWED_EXTENSIONS and is_post_media(name):
            if len(images) >= MAX_IMAGES:
                raise ValueError("Image limit exceeded")
            if len(data) <= MAX_MEDIA_BYTES and magic_ok(data, ext):
                images.append((name, data, ext))
        elif name.lower().endswith(".json") and "posts" in name.lower():
            blobs[name] = data
    captions = captions_from_json(blobs)
    records = []
    sha_seen = set()
    for source_name, data, ext in images:
        digest = hashlib.sha256(data).hexdigest()
        if digest in sha_seen:
            continue
        sha_seen.add(digest)
        stored = f"{digest}{ext}"
        target = out / "images" / stored
        target.write_bytes(data)
        meta = captions.get(source_name)
        # No post URL invented from filename or export identifier.
        records.append({
            "source_account": handle,
            "source_path": source_name,
            "source_post_url": None,
            "sha256": digest,
            "local_path": f"images/{stored}",
            "byte_size": len(data),
            "caption": meta["caption"] if meta else "",
            "creation_timestamp": meta["creation_timestamp"] if meta else None,
            "rights_basis": "creator_permission_reported_by_project_owner",
            "rights_scope": ["local_storage", "internal_reference_analysis", "ml_training_if_rights_confirmed"],
            "review_status": "pending_creator_ownership_and_outfit_review",
            "contains_full_outfit": None,
            "style_tags": [],
            "occasion_tags": [],
            "quality_score": None,
        })
    manifest = {
        "schema_version": 1,
        "source": "instagram_creator_export",
        "account": handle,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images_imported": len(records),
        "deduplicated_count": len(images) - len(records),
        "approved_for_training": 0,
        "rights_attestation": "project_owner_reports_account_owner_permission",
        "production_connected": False,
        "assets": records,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_account": handle, "images_imported": len(records),
        "deduplicated": len(images) - len(records),
        "approved_for_training": 0,
        "manifest": str(out / "manifest.json"),
        "production_connected": False,
    }))
    if not records:
        print("No post images found. Check the export layout and ZIP contents.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
