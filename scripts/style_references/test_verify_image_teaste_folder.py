"""46-folder importer tests; synthetic JPEG bytes only, no remote services."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
def load(name, filename):
    s = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod

verify = load("s6_verify_image_folder", "verify_image_teaste_folder.py")
upload = load("s6_private_upload", "upload_private_reference_bundle.py")


class VerifiedFolderTests(unittest.TestCase):
    def create_folder(self, root, omit=None, corrupt=None):
        folder = root / "image-teaste"
        folder.mkdir()
        rows = []
        for n in range(1,47):
            if n == omit:
                continue
            data = b"\xff\xd8\xff" + bytes([n])*6000
            if corrupt == n:
                data += b"new_unapproved_bytes"
            name = f"photo_{n}_2026-11-10_09-23-10.jpg"
            (folder/name).write_bytes(data)
            rows.append((name,hashlib.sha256(data).hexdigest(),len(data)))
        digest = hashlib.sha256("".join(f"{a}\t{b}\n" for a,b,_ in
            sorted(rows)).encode()).hexdigest()
        return folder, digest, sum(x[2] for x in rows)

    def test_46_images_verified_with_source_folder_not_zip(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            folder, sha, total = self.create_folder(root)
            images, actual, count = verify.validate_folder(
                folder, expected_aggregate=sha, expected_total_bytes=total)
            self.assertEqual(len(images),46)
            self.assertEqual(actual,sha)
            self.assertEqual(count,total)
            # Mac user may supply repo root, with an image-teaste subfolder.
            self.assertEqual(len(verify.validate_folder(root,
                expected_aggregate=sha, expected_total_bytes=total)[0]),46)

    def test_untrusted_image_modification_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            folder, sha, total = self.create_folder(Path(t))
            (folder/"photo_3_2026-11-10_09-23-10.jpg").write_bytes(
                b"\xff\xd8\xff" + b"something"*1000)
            with self.assertRaisesRegex(ValueError,"image_total_size_mismatch|canonical_image_sha256_mismatch"):
                verify.validate_folder(folder,
                    expected_aggregate=sha, expected_total_bytes=total)

    def test_missing_photo_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            folder,sha,total=self.create_folder(Path(t),omit=1)
            with self.assertRaisesRegex(ValueError,"expected_46"):
                verify.validate_folder(folder,
                    expected_aggregate=sha,expected_total_bytes=total)

    def test_reference_folder_builds_bundle_existing_uploader_accepts(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            folder,sha,total=self.create_folder(root)
            images,_,_=verify.validate_folder(folder,
                expected_aggregate=sha,expected_total_bytes=total)
            archive=verify.write_private_bundle(images,root/"private.zip")
            rows=upload.validated_bundle(archive)
            self.assertEqual(len(rows),46)
            self.assertEqual(len(set(r[3] for r in rows)),46)
            self.assertEqual(sum(len(r[2]) for r in rows),total)
            self.assertFalse(json.loads(__import__('zipfile').ZipFile(
                archive).read('manifest.json'))['training_approved'])

    def test_default_canonical_fingerprint_denies_synthetic_images(self):
        with tempfile.TemporaryDirectory() as t:
            folder,_,_=self.create_folder(Path(t))
            with self.assertRaisesRegex(ValueError,"image_total_size_mismatch|canonical_image_sha256_mismatch"):
                verify.validate_folder(folder)

if __name__=="__main__":
    unittest.main()
