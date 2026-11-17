#!/usr/bin/env python3
"""Verify the owner's existing 'image-teaste/' folder against the canonical ZIP.

Works WITHOUT the ZIP on this Mac. The 46 files are compared to the independently
verified user-uploaded canonical archive using a SHA256 *aggregate of filename
and per-file SHA256*, and the original total image bytes. No pictures in Git,
no database writes, no model calls, and no authorization to train.

Examples (from AIWardrobe repo root):
 python3 scripts/style_references/verify_image_teaste_folder.py --folder ./image-teaste
 python3 scripts/style_references/verify_image_teaste_folder.py \
   --folder ./image-teaste --output-private-bundle /private/s6-reference46.zip

The optional bundle can be passed to upload_private_reference_bundle.py
in DRY-RUN mode. It contains the SAME original image bytes, with a newly
written ZIP header, so its overall ZIP checksum need NOT equal the original.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

DATASET = "image-teaste-2026-11-10"
EXPECTED_AGGREGATE = "8b7b47845d678d15952dce6b1b16a42aeb4d94955e4ba08b8d0ede828167f04f"
EXPECTED_TOTAL_BYTES = 12_972_713
EXPECTED_COUNT = 46
MAX_IMAGE_BYTES = 5 * 1024 * 1024
FILENAME = re.compile(r"^photo_([1-9]|[1-3][0-9]|4[0-6])_2026-11-10_09-23-10\.jpg$")


def validate_folder(folder: Path, *, expected_aggregate=EXPECTED_AGGREGATE,
                    expected_total_bytes=EXPECTED_TOTAL_BYTES):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("photo_folder_not_found")
    # A screenshot may show the image-teaste folder directly under the repo.
    if not list(folder.glob("photo_*")) and (folder / "image-teaste").is_dir():
        folder = folder / "image-teaste"
    if folder.is_symlink():
        raise ValueError("symlink_folder_not_permitted")
    images = {}
    matched_names = set()
    for child in folder.iterdir():
        if child.name.startswith(".") or child.name == "__MACOSX":
            continue
        if child.is_symlink():
            raise ValueError("symlink_photo_not_permitted")
        if not child.is_file():
            continue
        if child.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic", ".webp"}:
            continue
        match = FILENAME.fullmatch(child.name)
        if not match:
            raise ValueError(f"unexpected_image_file:{child.name}")
        index = int(match.group(1))
        if index in images:
            raise ValueError("duplicate_number")
        size = child.stat().st_size
        if not 4096 <= size <= MAX_IMAGE_BYTES:
            raise ValueError("invalid_image_size")
        data = child.read_bytes()
        if not data.startswith(b"\xff\xd8\xff"):
            raise ValueError("invalid_jpeg_signature")
        images[index] = (child.name, hashlib.sha256(data).hexdigest(), data)
        matched_names.add(child.name)
    if len(images) != EXPECTED_COUNT or set(images) != set(range(1, EXPECTED_COUNT+1)):
        raise ValueError(f"expected_46_original_photos_got_{len(images)}")
    if len({row[1] for row in images.values()}) != EXPECTED_COUNT:
        raise ValueError("duplicate_image_bytes")
    total = sum(len(row[2]) for row in images.values())
    if total != expected_total_bytes:
        raise ValueError(f"image_total_size_mismatch:{total}")
    digest_source = "".join(
        f"{name}\t{sha}\n" for name, sha, _ in sorted(images.values(), key=lambda item:item[0])
    ).encode("utf-8")
    digest = hashlib.sha256(digest_source).hexdigest()
    if digest != expected_aggregate:
        raise ValueError("canonical_image_sha256_mismatch")
    return images, digest, total


def write_private_bundle(images, destination: Path):
    destination = Path(destination).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    if destination.is_relative_to(project_root):
        raise ValueError("private_bundle_must_stay_outside_public_repository")
    if destination.exists():
        raise ValueError("private_bundle_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sorted(images):
        name, digest, _ = images[n]
        rows.append({
            "id": f"rt_{n:03d}", "filename": name, "sha256": digest,
            "local_media_path": f"images/{digest}.jpg",
            "training_approved": False,
            "rights_confirmed": False,
        })
    manifest = {
        "schema_version": 2,
        "dataset_id": DATASET,
        "media_count": EXPECTED_COUNT,
        "published_or_uploaded": False,
        "rights_confirmed": False,
        "training_approved": False,
        "source": "original_image_teaste_folder_verified_against_canonical_sha256",
        "images": rows,
    }
    try:
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2)+"\n")
            for n in sorted(images):
                _, digest, data = images[n]
                z.writestr(f"images/{digest}.jpg", data)
        destination.chmod(0o600)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--output-private-bundle", type=Path)
    args = p.parse_args(argv)
    images, aggregate, total = validate_folder(args.folder)
    report = {
        "status": "verified_canonical_46_photo_folder",
        "count": len(images),
        "total_bytes": total,
        "per_photo_sha256_aggregate": aggregate,
        "uploaded": False,
        "model_trained": False,
    }
    if args.output_private_bundle:
        generated = write_private_bundle(images, args.output_private_bundle)
        report["private_bundle"] = str(generated)
        report["note"] = "Repacked ZIP SHA differs; individual image bytes match the canonical source."
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
