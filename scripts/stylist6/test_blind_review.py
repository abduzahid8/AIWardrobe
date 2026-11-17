"""Regression tests for private blind deck review validation (synthetic data only)."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("blind_review.py")
spec = importlib.util.spec_from_file_location("aiw_blind_review", MODULE)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

SLOTS = ("top", "outerwear", "bottom", "shoes")


class BlindReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def save_json(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def submission(self, reviewer, preference, quality=4):
        values = {"Q01.preference": preference, "reviewer": reviewer}
        for label in "AB":
            values[f"Q01.variety{label}"] = "4"
            for look in range(1, 4):
                prefix = f"Q01.{label}.{look}"
                for metric, maximum in review.METRICS.items():
                    values[f"{prefix}.{metric}"] = str(min(maximum, quality))
                values[f"{prefix}.critical"] = "No"
                values[f"{prefix}.note"] = ""
        return {"package": "fixed-package", "reviewer": reviewer, "values": values}

    def score_files(self, votes):
        key = {"package": "fixed-package", "cases": {"Q01": {"A": "baseline", "B": "v6Retrieval"}}}
        key_file = self.save_json("blind-key.json", key)
        paths = [self.save_json(f"vote{i}.json", self.submission(f"r{i+1}", vote))
                 for i, vote in enumerate(votes)]
        return key_file, paths

    def test_majority_v6_and_ties_remain_distinct(self):
        key, paths = self.score_files(("B", "B", "A"))
        result = review.score(key, paths)
        self.assertEqual((result["wins"], result["losses"], result["ties"], result["splits"]), (1, 0, 0, 0))
        self.assertEqual(result["cases"]["Q01"]["votes"], {"v6Retrieval": 2, "baseline": 1})
        self.assertEqual(result["reviewers"], 3)
        self.assertEqual(result["disagreementCases"], 1)
        key, paths = self.score_files(("B", "A", "Tie"))
        result = review.score(key, paths)
        self.assertEqual((result["wins"], result["losses"], result["splits"]), (0, 0, 1))

    def test_rejects_duplicate_reviewers_missing_scores_and_invalid_keys(self):
        key, paths = self.score_files(("B", "B", "A"))
        bad = json.loads(paths[1].read_text())
        bad["reviewer"] = "r1"
        paths[1].write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "distinct reviewers"):
            review.score(key, paths)
        bad["reviewer"] = "r2"
        bad["values"]["reviewer"] = "r2"
        bad["values"]["Q01.A.1.quality"] = "8"
        paths[1].write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Missing or invalid score"):
            review.score(key, paths)
        key.write_text(json.dumps({"package": "fixed-package", "cases": {}}))
        with self.assertRaisesRegex(ValueError, "Invalid or empty blind review key"):
            review.score(key, paths)

    def fixture(self, *, unsafe_photo=False, unsafe_case=False):
        folder = self.root / "organizer"
        folder.mkdir()
        (folder / "references").mkdir()
        (folder / "photos").mkdir()
        data = b"synthetic-photo-not-real-nvidia-input"
        digest = hashlib.sha256(data).hexdigest()
        safe_photo = f"photos/{digest}.jpg"
        (folder / safe_photo).write_bytes(data)
        (folder / "references/rt_001.jpg").write_bytes(data)
        (folder / "references/hashes.json").write_text(json.dumps({"rt_001": digest}))
        rows = []
        ids = []
        for slot in SLOTS:
            sku = f"sku-{slot}"
            ids.append(sku)
            rows.append({"row": {"id": sku, "name": f"{slot} synthetic", "price": 10,
                                 "currency": "UZS", "is_active": True},
                         "photo": "../../outside.jpg" if unsafe_photo else safe_photo,
                         "photoSha256": digest})
        inventory_bytes = json.dumps(rows).encode()
        scenario_id = '<svg onload=alert(1)>' if unsafe_case else "Q01"
        scenarios = [{"id": scenario_id, "reference": "rt_001", "occasion": "Work",
                      "brief": "Synthetic test", "budget": {"amount": 100, "currency": "UZS"}}]
        scenarios_bytes = json.dumps(scenarios).encode()
        (folder / "inventory.json").write_bytes(inventory_bytes)
        (folder / "scenarios.json").write_bytes(scenarios_bytes)
        replay = {"inventorySha256": hashlib.sha256(inventory_bytes).hexdigest(),
                  "scenariosSha256": hashlib.sha256(scenarios_bytes).hexdigest(),
                  "admitted": [{"id": sku, "macroCategory": slot}
                               for sku, slot in zip(ids, SLOTS)],
                  "results": [{"id": scenario_id, "baseline": [ids] * 3,
                               "v6Retrieval": [ids] * 3}]}
        (folder / "replay.json").write_text(json.dumps(replay))
        grant = {"status": "active", "validFrom": "2020-01-01T00:00:00Z", "expiresAt": None,
                 "sha256": digest, "rightsHolder": "Synthetic fixture author", "reviewedBy": "Fixture test",
                 "evidence": {"reference": "synthetic fixture", "sha256": digest},
                 "privateReview": {"redistribute": True, "audience": "assigned_independent_reviewers"}}
        (folder / "image-rights.json").write_text(json.dumps({"version": 1, "assets": [
            {**grant, "kind": "reference", "id": "rt_001"},
            *[{**grant, "kind": "product", "id": sku} for sku in ids],
        ]}))
        return folder

    def test_builds_local_review_pack_and_disallows_unsafe_paths(self):
        folder = self.fixture()
        output = self.root / "review"
        review.build(folder, output)
        self.assertTrue((output / "index.html").exists())
        self.assertTrue(output.with_suffix(".zip").exists())
        self.assertEqual(len(list((output / "assets").glob("*.jpg"))), 1)
        key = json.loads((folder / "retail-blind-key.json").read_text())
        self.assertEqual(set(key["cases"]["Q01"]), {"A", "B"})
        seed = (folder / "retail-review-allocation.txt").read_text().strip()
        self.assertNotIn(seed, (output / "index.html").read_text())
        again = self.root / "again"
        review.build(folder, again)
        self.assertEqual((output / "index.html").read_bytes(), (again / "index.html").read_bytes())
        with self.assertRaisesRegex(ValueError, "Hosted comparison requires judged"):
            review.build(folder, self.root / "not-hosted", v6_arm="v6Hosted")

    def test_twelve_cases_counterbalance_hidden_deck_position(self):
        folder = self.fixture()
        cases = [{"id": f"Q{i:02}", "reference": "rt_001", "occasion": "Work",
                  "brief": "Synthetic test", "budget": {"amount": 100, "currency": "UZS"}}
                 for i in range(1, 13)]
        scenario_bytes = json.dumps(cases).encode()
        (folder / "scenarios.json").write_bytes(scenario_bytes)
        replay = json.loads((folder / "replay.json").read_text())
        replay["scenariosSha256"] = hashlib.sha256(scenario_bytes).hexdigest()
        first = replay["results"][0]
        replay["results"] = [{**first, "id": case["id"]} for case in cases]
        (folder / "replay.json").write_text(json.dumps(replay))
        review.build(folder, self.root / "balanced-review")
        assignments = json.loads((folder / "retail-blind-key.json").read_text())["cases"]
        self.assertEqual(len(assignments), 12)
        self.assertEqual(sum(labels["A"] == "baseline" for labels in assignments.values()), 6)
        self.assertEqual(sum(labels["B"] == "baseline" for labels in assignments.values()), 6)

    def test_owner_preview_does_not_distribute_and_default_requires_review_rights(self):
        folder = self.fixture()
        (folder / "image-rights.json").unlink()
        output = self.root / "review"
        with self.assertRaisesRegex(ValueError, "distribution requires"):
            review.build(folder, output)
        self.assertFalse(output.exists())
        review.build(folder, output, owner_local=True)
        self.assertFalse(output.with_suffix(".zip").exists())
        self.assertNotIn('type="submit"', (output / "index.html").read_text())
        with self.assertRaisesRegex(ValueError, "Owner preview"):
            review.score(folder / "owner-preview-key.json", [])

    def test_display_permission_does_not_authorize_reviewer_distribution(self):
        folder = self.fixture()
        path = folder / "image-rights.json"
        rights = json.loads(path.read_text())
        rights["assets"][0]["display"] = True
        rights["assets"][0].pop("privateReview")
        path.write_text(json.dumps(rights))
        output = self.root / "display-only"
        with self.assertRaisesRegex(ValueError, "lacks assigned-reviewer distribution grant"):
            review.build(folder, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".zip").exists())

    def test_rejects_unexpected_photo_paths(self):
        folder = self.fixture(unsafe_photo=True)
        with self.assertRaisesRegex(ValueError, "Unexpected product photo path"):
            review.build(folder, self.root / "bad-review")

    def test_rejects_unsafe_scenario_identifier(self):
        folder = self.fixture(unsafe_case=True)
        with self.assertRaisesRegex(ValueError, "Unexpected review scenario ID"):
            review.build(folder, self.root / "unsafe-review")


if __name__ == "__main__":
    unittest.main()
