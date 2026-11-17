"""Offline unit/contract tests. Test data is synthetic, not creator imagery."""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_instagram_archive.py")
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 60


def execute(archive, dest, extra=()):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(archive), "--output", str(dest),
         "--account", "sainttnoirclub", *extra],
        text=True, capture_output=True, check=False, timeout=20,
    )


class ImportTests(unittest.TestCase):
    def test_real_creator_export_shape_and_sha_dedup_without_auto_training(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base / "export.zip"
            with zipfile.ZipFile(source, "w") as zipout:
                zipout.writestr("media/posts/2026/photo1.jpg", JPG)
                zipout.writestr("media/posts/2026/photo1-duplicate.jpg", JPG)
                zipout.writestr("media/posts/2026/video.mp4", b"FAKE-VIDEO")
                zipout.writestr("media/stories/2026/story.jpg", JPG + b"1")
                zipout.writestr("messages/dm_photo.jpg", JPG + b"2")
                zipout.writestr("your_instagram_activity/posts/posts_1.json", json.dumps([
                    {"media": [{"uri": "media/posts/2026/photo1.jpg", "title": "Summer Old Money",
                                "creation_timestamp": 1794200000}]}
                ]))
            result = execute(source, base / "out", ("--owner-authorized",))
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((base / "out" / "manifest.json").read_text())
            self.assertEqual(data["account"], "sainttnoirclub")
            self.assertEqual(data["images_imported"], 1)
            self.assertEqual(data["deduplicated_count"], 1)
            self.assertEqual(data["approved_for_training"], 0)
            self.assertEqual(data["assets"][0]["caption"], "Summer Old Money")
            self.assertEqual(data["assets"][0]["review_status"],
                             "pending_creator_ownership_and_outfit_review")
            self.assertTrue((base / "out" / data["assets"][0]["local_path"]).exists())

    def test_no_permission_no_import(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"e.zip"
            with zipfile.ZipFile(src,"w") as out:out.writestr("media/posts/a.jpg",JPG)
            result=execute(src,Path(d)/"out")
            self.assertNotEqual(result.returncode,0)
            self.assertIn("--owner-authorized",result.stderr)

    def test_reject_path_traversal_in_archive(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"e.zip"
            with zipfile.ZipFile(src,"w") as out:
                out.writestr("../outside.jpg",JPG)
            result=execute(src,Path(d)/"out",("--owner-authorized",))
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((Path(d)/"outside.jpg").exists())

    def test_reject_spoofed_jpg_and_ignore_dms_stories(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"e.zip"
            with zipfile.ZipFile(src,"w") as out:
                out.writestr("media/posts/some.jpg",b"not-an-image")
                out.writestr("messages/message.jpg",JPG)
                out.writestr("media/stories/story.jpg",JPG)
            result=execute(src,Path(d)/"out",("--owner-authorized",))
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads((Path(d)/"out"/"manifest.json").read_text())
            self.assertEqual(data["images_imported"],0)

    def test_output_must_be_empty_and_not_inside_source(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"e.zip"
            with zipfile.ZipFile(src,"w") as out:out.writestr("media/posts/a.jpg",JPG)
            dst=Path(d)/"out"
            dst.mkdir()
            (dst/"existing.txt").write_text("do-not-overwrite")
            result=execute(src,dst,("--owner-authorized",))
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((dst/"existing.txt").read_text(),"do-not-overwrite")


if __name__=="__main__":
    unittest.main()
