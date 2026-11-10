"""Build a private static deck review and score three independent submissions."""
import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
from math import comb
from pathlib import Path
import random
import shutil
from statistics import mean
from zipfile import ZipFile, ZIP_DEFLATED

ARMS = ("baseline", "v6Retrieval")
METRICS = {"quality": 5, "coherence": 5, "fidelity": 5, "purchase": 3}
SLOTS = ("top", "outerwear", "bottom", "shoes")


def read(file):
    return json.loads(file.read_text())


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def choose(name, label, values):
    options = "".join(f'<option value="{value}">{value}</option>' for value in values)
    return f'<label>{label} <select name="{name}" required><option value="">Choose</option>{options}</select></label>'


def build(folder, output):
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    replay_path = folder / "replay.json"
    replay = read(replay_path)
    package = digest(replay_path)
    if (digest(folder / "inventory.json") != replay["inventorySha256"]
            or digest(folder / "scenarios.json") != replay["scenariosSha256"]):
        raise ValueError("Fixture digest mismatch")
    inventory = {record["row"]["id"]: record for record in read(folder / "inventory.json")}
    scenarios = {case["id"]: case for case in read(folder / "scenarios.json")}
    slots = {item["id"]: item["macroCategory"] for item in replay["admitted"]}
    hashes = read(folder / "references/hashes.json")
    rng = random.Random(int(package[:16], 16))
    key = {"package": package, "cases": {}}
    fields = []

    def copy_photo(source, expected):
        if digest(source) != expected:
            raise ValueError("Fixture photo digest mismatch")
        filename = expected + ".jpg"
        if not (assets / filename).exists():
            shutil.copyfile(source, assets / filename)
        return "assets/" + filename

    for case in replay["results"]:
        name = case["id"]
        scenario = scenarios[name]
        ref = scenario["reference"]
        ref_asset = copy_photo(folder / "references" / (ref + ".jpg"), hashes[ref])
        budget = scenario["budget"]
        cap = f'{budget["amount"]:,} {budget["currency"]}' if budget else "No price cap"
        parts = [f'<details><summary>{name} · {escape(scenario["occasion"])}</summary>',
                 f'<p>{escape(scenario["brief"])}. Budget per look {cap}.</p>',
                 f'<img class="reference" src="{ref_asset}" alt="Fashion reference for {name}" loading="lazy">']
        order = list(ARMS)
        rng.shuffle(order)
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
                    src = copy_photo(folder / record["photo"], record["photoSha256"])
                    caption = escape(f'{slots[sku]} · {row["name"]}')
                    parts.append(f'<figure><img src="{src}" alt="{caption}" loading="lazy"><figcaption>{caption}'
                                 f'<br>{row["price"]:,.2f} {escape(row["currency"])}</figcaption></figure>')
                if budget and (set(totals) != {budget["currency"]} or totals[budget["currency"]] > budget["amount"]):
                    raise ValueError("Frozen outfit budget violated")
                parts.append('</div>')
                for metric, maximum in METRICS.items():
                    parts.append(choose(f'{prefix}.{metric}', metric.capitalize(), range(1, maximum + 1)))
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
    (output / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Outfit deck review</title>
<style>{css}</style><main><h1>Independent outfit review</h1>
<p>Compare two anonymous decks per scenario. Rate quality, coherence and fidelity from 1 (poor) to 5 (excellent).
Rate purchase plausibility from 1 (unlikely) to 3 (plausible). Price is from the catalog snapshot.
Use critical for a serious product, outfit or constraint issue, and explain it.
Rate each deck’s useful variety from 1 to 5. Pick the deck you would want, or mark a tie.</p>
<p>Work independently. Use the private reviewer code assigned by the organizer. Export once all cases are complete.
The page runs locally. Keep images and submissions within the authorized review group.</p>
<form novalidate data-package="{package}">
<label>Reviewer code <input name="reviewer" pattern="[A-Za-z0-9_]{{2,32}}" maxlength="32" required></label>
{''.join(fields)}<button type="submit">Export ratings</button><p id="status" role="status"></p></form></main>
<script>{js}</script></html>""")
    (folder / "retail-blind-key.json").write_text(json.dumps(key, indent=2) + "\n")
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
    submissions = [read(file) for file in files]
    if (any(x.get("package") != key["package"] for x in submissions)
            or len({x.get("reviewer") for x in submissions}) != 3
            or not all(x.get("reviewer") for x in submissions)):
        raise ValueError("Expected three distinct reviewers for this package")
    cases, flags = {}, []
    for name, labels in key["cases"].items():
        votes, differences = Counter(), []
        for review in submissions:
            values = review["values"]
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
                    if critical not in ("Yes", "No") or critical == "Yes" and not note.strip():
                        raise ValueError("Missing critical issue decision or detail")
                    if critical == "Yes":
                        flags.append({"case": name, "arm": arm, "look": index, "reviewer": review["reviewer"], "note": note})
                scores[arm] = {metric: mean(scores) for metric, scores in deck.items()}
                scores[arm]["variety"] = rate(values, f"{name}.variety{label}", 5)
            differences.append({m: scores[ARMS[1]][m] - scores[ARMS[0]][m] for m in (*METRICS, "variety")})
        paired = {metric: mean(diff[metric] for diff in differences) for metric in differences[0]}
        cases[name] = {"votes": dict(votes), "majority": next((v for v in votes if votes[v] >= 2), "split"),
                       "pairedDifference": paired}
    counts = Counter(row["majority"] for row in cases.values())
    wins, losses = counts[ARMS[1]], counts[ARMS[0]]
    decisive = wins + losses
    pvalue = min(1, 2 * sum(comb(decisive, i) for i in range(min(wins, losses) + 1)) / 2**decisive) if decisive else None
    uncertainty = {}
    rng = random.Random(48201)
    for metric in (*METRICS, "variety"):
        diffs = [row["pairedDifference"][metric] for row in cases.values()]
        draws = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
        uncertainty[metric] = {"mean": mean(diffs), "scenarioBootstrap95": [draws[124], draws[4874]]}
    return {"wins": wins, "losses": losses, "ties": counts["tie"], "splits": counts["split"],
            "exactSignTestP": pvalue, "uncertainty": uncertainty, "cases": cases, "criticalFlags": flags}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("organizer", type=Path)
    build_command.add_argument("output", type=Path)
    score_command = commands.add_parser("score")
    score_command.add_argument("key", type=Path)
    score_command.add_argument("submissions", type=Path, nargs=3)
    args = parser.parse_args()
    if args.command == "build":
        build(args.organizer, args.output)
    else:
        print(json.dumps(score(args.key, args.submissions), indent=2))
