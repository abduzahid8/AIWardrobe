"""Build a private static deck review and score three independent submissions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from math import comb
from pathlib import Path
import random
import re
import secrets
import shutil
from statistics import mean, quantiles
from zipfile import ZipFile, ZIP_DEFLATED

ARMS = ("baseline", "v6Retrieval")
METRICS = {"quality": 5, "coherence": 5, "fidelity": 5, "occasion": 5, "purchase": 3}
LABELS = {"quality": "Overall attractiveness", "coherence": "Garment compatibility",
          "fidelity": "Reference fidelity", "occasion": "Occasion suitability", "purchase": "Purchase plausibility"}
SCHEMA = "deck-review-v2"
SLOTS = ("top", "outerwear", "bottom", "shoes")


def read(file):
    def unique_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate JSON field")
        return result

    return json.loads(file.read_text(), object_pairs_hook=unique_keys)


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def choose(name, label, values):
    options = "".join(f'<option value="{value}">{value}</option>' for value in values)
    return f'<label>{label} <select name="{name}" required><option value="">Choose</option>{options}</select></label>'


def review_granted(manifest, kind, asset_id, expected, source_url):
    rows = [row for row in manifest.get("assets", [])
            if (row.get("kind"), row.get("id"), row.get("sha256")) == (kind, asset_id, expected)]
    if len(rows) != 1:
        return False
    row = rows[0]
    scope = row.get("privateReview") or {}
    now = datetime.now(timezone.utc)
    try:
        starts = datetime.fromisoformat(row["validFrom"].replace("Z", "+00:00"))
        expiry = row["expiresAt"]
        ends = datetime.fromisoformat(expiry.replace("Z", "+00:00")) if expiry is not None else None
        return (row.get("status") == "active" and starts <= now and (ends is None or ends > now)
                and bool(row.get("reviewedBy")) and bool(row.get("rightsHolder"))
                and bool(row["evidence"]["reference"])
                and bool(re.fullmatch(r"[0-9a-f]{64}", row["evidence"]["sha256"]))
                and (source_url is None or row.get("sourceUrl") == source_url)
                and scope.get("redistribute") is True and scope.get("audience") == "assigned_independent_reviewers")
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def build(folder, output, *, owner_local=False, v5_arm="baseline", v6_arm="v6Retrieval"):
    assets = output / "assets"
    media = {}
    rights = None
    if not owner_local:
        manifest_file = folder / "image-rights.json"
        if not manifest_file.is_file():
            raise ValueError("Reviewer distribution requires image-rights.json")
        rights = read(manifest_file)
        if rights.get("version") != 1 or not isinstance(rights.get("assets"), list):
            raise ValueError("Invalid image rights manifest")
    replay_path = folder / "replay.json"
    replay = read(replay_path)
    if not isinstance(replay.get("results"), list) or not replay["results"] or len({row["id"] for row in replay["results"]}) != len(replay["results"]):
        raise ValueError("Expected nonempty, unique review scenarios")
    arms = (v5_arm, v6_arm)
    if v5_arm not in ("baseline", "v5Seed") or v6_arm not in ("v6Retrieval", "v6Hosted"):
        raise ValueError("Unsupported comparison arms")
    if v6_arm == "v6Hosted" and any(row.get("hostedStatus") != "JUDGED" or not row.get(v6_arm) for row in replay["results"]):
        raise ValueError("Hosted comparison requires judged Stylist 6 decks for every case")
    seed_file = folder / ("owner-preview-allocation.txt" if owner_local else "retail-review-allocation.txt")
    seed = seed_file.read_text().strip() if seed_file.exists() else secrets.token_hex(32)
    if not re.fullmatch(r"[0-9a-f]{64}", seed):
        raise ValueError("Invalid private allocation seed")
    package = sha256(("|".join((SCHEMA, digest(replay_path), *arms, seed))).encode()).hexdigest()
    if (digest(folder / "inventory.json") != replay["inventorySha256"]
            or digest(folder / "scenarios.json") != replay["scenariosSha256"]):
        raise ValueError("Fixture digest mismatch")
    inventory = {record["row"]["id"]: record for record in read(folder / "inventory.json")}
    scenarios = {case["id"]: case for case in read(folder / "scenarios.json")}
    slots = {item["id"]: item["macroCategory"] for item in replay["admitted"]}
    hashes = read(folder / "references/hashes.json")
    rng = random.Random(int(seed, 16))
    # Counterbalance deck sides so each arm appears first equally often
    # across the twelve paired cases, while keeping allocation blinded.
    count = len(replay["results"])
    flips = [False] * ((count + 1) // 2) + [True] * (count // 2)
    rng.shuffle(flips)
    key = {"package": package, "cases": {}, "arms": arms,
           "scope": {row["id"]: {"baseline": "deterministic" if v5_arm == "v5Seed" else row.get("v5Mode", "unspecified"),
                                 "v6": "hosted_visual" if v6_arm == "v6Hosted" else "retrieval_only"}
                     for row in replay["results"]}, "ownerLocal": owner_local}
    fields = []

    def copy_photo(source, expected, kind, asset_id, source_url=None):
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid photo digest")
        source = source.resolve(strict=True)
        if not source.is_relative_to(folder.resolve(strict=True)) or not source.is_file():
            raise ValueError("Review photo must be inside organizer")
        if not owner_local and not review_granted(rights, kind, asset_id, expected, source_url):
            raise ValueError("Image lacks assigned-reviewer distribution grant: " + asset_id)
        if digest(source) != expected:
            raise ValueError("Fixture photo digest mismatch")
        filename = expected + ".jpg"
        media[filename] = source
        return "assets/" + filename

    for case in replay["results"]:
        name = case["id"]
        if not isinstance(name, str) or not re.fullmatch(r"Q[0-9]{2}", name):
            raise ValueError("Unexpected review scenario ID")
        scenario = scenarios[name]
        ref = scenario["reference"]
        if not isinstance(ref, str) or not re.fullmatch(r"rt_[0-9]{3}", ref):
            raise ValueError("Unexpected reference ID")
        ref_asset = copy_photo(folder / "references" / (ref + ".jpg"), hashes[ref], "reference", ref)
        budget = scenario["budget"]
        cap = escape(f'{budget["amount"]:,} {budget["currency"]}') if budget else "No price cap"
        parts = [f'<details><summary>{name} · {escape(scenario["occasion"])}</summary>',
                 f'<p>{escape(scenario["brief"])}. Budget per look {cap}.</p>',
                 f'<img class="reference" src="{ref_asset}" alt="Fashion reference for {name}" loading="lazy">']
        order = list(arms)
        if flips[len(key["cases"])]:
            order.reverse()
        key["cases"][name] = dict(zip("AB", order))
        for label, arm in key["cases"][name].items():
            if len(case[arm]) != 3:
                raise ValueError("Expected three complete looks per arm")
            parts.append(f'<section><h2>Deck {label}</h2><div class="deck">')
            for index, ids in enumerate(case[arm], 1):
                if len(ids) != 4 or {slots[sku] for sku in ids} != set(SLOTS):
                    raise ValueError("Incomplete outfit")
                prefix = f"{name}.{label}.{index}"
                parts.append(f'<fieldset><legend>Deck {label}, look {index}</legend><div class="items">')
                totals = Counter()
                for sku in sorted(ids, key=lambda sku: SLOTS.index(slots[sku])):
                    record = inventory[sku]
                    row = record["row"]
                    totals[row["currency"]] += row["price"]
                    if record["photo"] != f'photos/{record["photoSha256"]}.jpg':
                        raise ValueError("Unexpected product photo path")
                    src = copy_photo(folder / record["photo"], record["photoSha256"], "product", sku, row.get("image_url"))
                    caption = escape(f'{slots[sku]} · {row["name"]}')
                    parts.append(f'<figure><img src="{src}" alt="{caption}" loading="lazy"><figcaption>{caption}'
                                 f'<br>{row["price"]:,.2f} {escape(row["currency"])}</figcaption></figure>')
                if budget and (set(totals) != {budget["currency"]} or totals[budget["currency"]] > budget["amount"]):
                    raise ValueError("Frozen outfit budget violated")
                parts.append('</div>')
                for metric, maximum in METRICS.items():
                    parts.append(choose(f'{prefix}.{metric}', LABELS[metric], range(1, maximum + 1)))
                parts.append(choose(f'{prefix}.critical', 'Critical issue', ("No", "Yes")))
                parts.append(f'<label>Critical issue detail <input name="{prefix}.note" maxlength="400"></label></fieldset>')
            parts.append('</div></section>')
        parts.extend(choose(f'{name}.variety{label}', f'Deck {label} variety', range(1, 6)) for label in "AB")
        parts.append(choose(f'{name}.preference', 'Which deck would you rather receive?', ("A", "B", "Tie")) + '</details>')
        fields.extend(parts)
    css = """
body{font:1rem/1.5 system-ui;max-width:86rem;margin:auto;padding:1rem;color:#1b1c1b;background:#fafbf8}
details,fieldset{border:1px solid #65705e;margin:1rem 0;padding:.8rem;border-radius:.35rem}
summary,legend{font-weight:700}summary{cursor:pointer}.deck,.items{display:grid;gap:.5rem}
.deck{grid-template-columns:repeat(auto-fit,minmax(min(100%,19rem),1fr))}.items{grid-template-columns:1fr 1fr}
figure{margin:0}img{width:100%;height:9rem;object-fit:contain;background:#eeefec}
.reference{width:min(24rem,100%);height:auto}label{display:block;margin:.5rem 0}
select,input,button{font:inherit;padding:.35rem;color:#161916;background:white}
:focus-visible{outline:3px solid #185b46;outline-offset:2px}
"""
    js = r"""
const form=document.querySelector('form'), status=document.querySelector('#status');
const key='aiwardrobe-'+form.dataset.package;
function fields(){return Object.fromEntries(new FormData(form));}
function save(){try{localStorage.setItem(key, JSON.stringify(fields()));}catch{}}
form.addEventListener('input',save);
try{for(const [name,value] of Object.entries(JSON.parse(localStorage.getItem(key)||'{}'))){
  const field=form.elements.namedItem(name);if(field)field.value=value;
}}catch{}
form.addEventListener('submit',event=>{
  event.preventDefault();
  if(form.dataset.ownerLocal==='true') return;
  const values=fields();
  for(const field of form.elements){
    if(field.required && (!field.value || !field.checkValidity())){
      const detail=field.closest('details');if(detail)detail.open=true;
      field.focus();field.reportValidity();status.textContent='Complete every rating before exporting.';return;
    }
    if(field.name.endsWith('.note') && values[field.name.replace(/\.note$/,'.critical')]==='Yes' && !field.value.trim()){
      field.closest('details').open=true;field.focus();status.textContent='Explain each critical issue.';return;
    }
  }
  const reviewer=values.reviewer.trim();
  const blob=new Blob([JSON.stringify({package:form.dataset.package,reviewer,values},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob), a=document.createElement('a');
  a.href=url;a.download='review-'+reviewer+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  status.textContent='Review exported. Submit the downloaded JSON privately.';
});
"""
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    assets.mkdir(mode=0o700)
    for filename, source in media.items():
        shutil.copyfile(source, assets / filename)
    notice = '<p><strong>Owner-only preview. Images have no reviewer distribution clearance.</strong></p>' if owner_local else ''
    (output / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Outfit deck review</title>
<style>{css}</style><main><h1>Independent outfit review</h1>{notice}
<p>Compare two anonymous decks per scenario. Rate attractiveness, compatibility, reference fidelity and occasion suitability from 1 (poor) to 5 (excellent).
Rate purchase plausibility from 1 (unlikely) to 3 (plausible). Price is from the catalog snapshot.
Use critical for a serious product, outfit or constraint issue, and explain it.
Rate each deck’s useful variety from 1 to 5. Pick the deck you would want, or mark a tie.</p>
<p>Work independently in your own browser profile. Use the private reviewer code assigned by the organizer. Export once all cases are complete.
The page runs locally. Keep images and submissions within the authorized review group.</p>
<form novalidate data-package="{package}" data-owner-local="{str(owner_local).lower()}">
<label>Reviewer code <input name="reviewer" pattern="[A-Za-z0-9_]{{2,32}}" maxlength="32" required></label>
{''.join(fields)}{'' if owner_local else '<button type="submit">Export ratings</button>'}<p id="status" role="status"></p></form></main>
<script>{js}</script></html>""")
    if not seed_file.exists():
        seed_file.touch(mode=0o600, exist_ok=False)
        seed_file.write_text(seed + "\n")
    key_name = "owner-preview-key.json" if owner_local else "retail-blind-key.json"
    (folder / key_name).write_text(json.dumps(key, indent=2) + "\n")
    if not owner_local:
        with ZipFile(output.with_suffix(".zip"), "x", ZIP_DEFLATED) as archive:
            for file in sorted(output.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(output))


def rate(values, field, high):
    value = values.get(field)
    if value not in [str(n) for n in range(1, high + 1)]:
        raise ValueError("Missing or invalid score: " + field)
    return int(value)


def score(key_file, files):
    key = read(key_file)
    arms = key.get("arms", ARMS)
    if (len(arms) != 2 or arms[0] not in ("baseline", "v5Seed") or arms[1] not in ("v6Retrieval", "v6Hosted")
            or not key.get("cases") or any(set(labels) != {"A", "B"} or set(labels.values()) != set(arms)
                                          for labels in key["cases"].values())):
        raise ValueError("Invalid or empty blind review key")
    if key.get("ownerLocal"):
        raise ValueError("Owner preview cannot receive independent ratings")
    submissions = [read(file) for file in files]
    if (len(submissions) != 3 or any(x.get("package") != key["package"] for x in submissions)
            or len({x.get("reviewer") for x in submissions}) != 3
            or not all(isinstance(x.get("reviewer"), str) and re.fullmatch(r"[A-Za-z0-9_]{2,32}", x["reviewer"])
                       for x in submissions)):
        raise ValueError("Expected three distinct reviewers for this package")
    cases, flags = {}, []
    for name, labels in key["cases"].items():
        votes, differences = Counter(), []
        for review in submissions:
            values = review["values"]
            if not isinstance(values, dict) or values.get("reviewer") != review["reviewer"]:
                raise ValueError("Reviewer code does not match form")
            preference = values.get(f'{name}.preference')
            if preference not in ("A", "B", "Tie"):
                raise ValueError("Missing deck preference")
            votes[labels.get(preference, "tie")] += 1
            scores = {}
            for label, arm in labels.items():
                deck = {metric: [] for metric in METRICS}
                for index in range(1, 4):
                    prefix = f"{name}.{label}.{index}"
                    for metric, limit in METRICS.items():
                        deck[metric].append(rate(values, f"{prefix}.{metric}", limit))
                    critical, note = values.get(prefix + ".critical"), values.get(prefix + ".note", "")
                    if (critical not in ("Yes", "No") or not isinstance(note, str) or len(note) > 400
                            or critical == "Yes" and not note.strip()):
                        raise ValueError("Missing critical issue decision or detail")
                    if critical == "Yes":
                        flags.append({"case": name, "arm": arm, "look": index, "reviewer": review["reviewer"], "note": note})
                scores[arm] = {metric: mean(scores) for metric, scores in deck.items()}
                scores[arm]["variety"] = rate(values, f"{name}.variety{label}", 5)
            differences.append({m: scores[arms[1]][m] - scores[arms[0]][m] for m in (*METRICS, "variety")})
        paired = {metric: mean(diff[metric] for diff in differences) for metric in differences[0]}
        cases[name] = {"votes": dict(votes), "majority": next((v for v in votes if votes[v] >= 2), "split"),
                       "pairedDifference": paired, "reviewersDisagreed": len(votes) > 1}
    counts = Counter(row["majority"] for row in cases.values())
    wins, losses = counts[arms[1]], counts[arms[0]]
    decisive = wins + losses
    pvalue = min(1, 2 * sum(comb(decisive, i) for i in range(min(wins, losses) + 1)) / 2**decisive) if decisive else None
    uncertainty = {}
    rng = random.Random(48201)
    for metric in (*METRICS, "variety"):
        diffs = [row["pairedDifference"][metric] for row in cases.values()]
        draws = [mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000)]
        limits = quantiles(draws, n=40, method="inclusive")
        uncertainty[metric] = {"mean": mean(diffs), "scenarioBootstrap95": [limits[0], limits[-1]]}
    return {"wins": wins, "losses": losses, "ties": counts["tie"], "splits": counts["split"],
            "exactSignTestP": pvalue, "decisiveCases": decisive, "reviewers": len(submissions),
            "disagreementCases": sum(row["reviewersDisagreed"] for row in cases.values()),
            "scope": key.get("scope"), "uncertainty": uncertainty, "cases": cases, "criticalFlags": flags}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("organizer", type=Path)
    build_command.add_argument("output", type=Path)
    build_command.add_argument("--owner-local", action="store_true")
    build_command.add_argument("--v5-arm", choices=("baseline", "v5Seed"), default="baseline")
    build_command.add_argument("--v6-arm", choices=("v6Retrieval", "v6Hosted"), default="v6Retrieval")
    score_command = commands.add_parser("score")
    score_command.add_argument("key", type=Path)
    score_command.add_argument("submissions", type=Path, nargs=3)
    args = parser.parse_args()
    if args.command == "build":
        build(args.organizer, args.output, owner_local=args.owner_local, v5_arm=args.v5_arm, v6_arm=args.v6_arm)
    else:
        print(json.dumps(score(args.key, args.submissions), indent=2))
