"""Render a local developer gallery from an offline fashion replay."""
import argparse
from html import escape
import json
from pathlib import Path
from urllib.parse import urlsplit

ARMS = {
    "stylist5": "Stylist 5 · deterministic production kernel",
    "stylist6Previous": "Stylist 6 · previous retrieval",
    "stylist6Fashion": "Stylist 6 · outfit-ranked",
}


def safe_link(value):
    try:
        url = urlsplit(value or "")
        return value if url.scheme == "https" and url.hostname and not (url.username or url.password) else ""
    except ValueError:
        return ""


def gallery(data):
    catalog = data["catalog"]
    cases = []
    for row in data["results"]:
        case = row["scenario"]
        title = escape(f'{case["id"]} · {case["style"].replace("_", " ")} · {case["occasion"]}')
        weather = case["weather"]["temp"]
        budget = case.get("budget")
        cap = f'{budget["amount"]:,} {budget["currency"]}' if budget else "open cap"
        refs = row.get("reference")
        reference = escape(refs["name"]) if refs else "No matched lookbook brief"
        html = [f'<details id="{escape(case["id"], quote=True)}"><summary>{title}</summary><p>{weather}°C · {escape(cap)} · {reference}</p>']
        if case.get("preferences"):
            pref = ", ".join(f"{k} {', '.join(v) if isinstance(v, list) else v}" for k, v in case["preferences"].items())
            html.append(f"<p>Preferences · {escape(pref)}</p>")
        for arm, label in ARMS.items():
            looks = row["arms"][arm]
            html.append(f"<h2>{label}</h2><div class='deck'>")
            for i, look in enumerate(looks, 1):
                description = escape(look.get("description") or "")
                html.append(f'<article><h3>Look {i} · {look["spend"]:,.2f} UZS</h3>'
                            f'<p>Diagnostic coherence {look["coherence"]:.3f}</p><p>{description}</p><div class="items">')
                for sku in look["ids"]:
                    p = catalog[sku]
                    source = escape(safe_link(p["imageUrl"]), quote=True)
                    url = escape(safe_link(p["shopUrl"]), quote=True)
                    caption = escape(f'{p["slot"]} · {p["name"]}')
                    name = f'<a href="{url}" rel="noreferrer">{caption}</a>' if url else caption
                    html.append(f'<figure><img data-src="{source}" alt="{caption}" loading="lazy" referrerpolicy="no-referrer">'
                                f'<figcaption>{name}<br><code>{escape(sku)}</code>'
                                f'<br>{p["price"]:,.2f} {escape(p["currency"])}</figcaption></figure>')
                if look["reasons"]:
                    html.append(f'<p>{escape(", ".join(look["reasons"]))}</p>')
                html.append("</div></article>")
            if not looks:
                html.append("<p>No feasible look in this replay.</p>")
            html.append("</div>")
        cases.append("".join(html) + "</details>")
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Stylist outfit comparison</title>
<style>
body{background:#fafaf7;color:#191916;font:16px/1.5 system-ui;max-width:90rem;margin:auto;padding:1rem}
details{border:1px solid #7b7b70;padding:.75rem;margin:1rem 0}summary{cursor:pointer;font-weight:700}
.deck,.items{display:grid;gap:.7rem}.deck{grid-template-columns:repeat(auto-fit,minmax(min(100%,21rem),1fr))}
.items{grid-template-columns:repeat(2,minmax(0,1fr))}article{background:white;padding:.75rem;min-width:0}
figure{margin:0;overflow-wrap:anywhere}img{width:100%;height:10rem;object-fit:contain;background:#eeeeea}
img:not([src]){visibility:hidden;height:0}h3{margin:0}:focus-visible{outline:3px solid #145b49}
</style>
<main><h1>Stylist outfit comparison</h1>
<p>Private developer replay. Prices are from the frozen catalog. The scores are automated diagnostics.
Stock, shipping and shopper preference remain unverified.</p>
<label><input id="photos" type="checkbox"> Load catalog photos when authorized for this private display</label>
<p>Opening a case with photos enabled loads its exact image URLs from merchant hosts.</p>
""" + "".join(cases) + """</main>
<script>
const toggle = document.querySelector('#photos');
function syncImages() {
  for (const image of document.querySelectorAll('img[data-src]')) {
    if (toggle.checked && image.closest('details').open && image.dataset.src) {
      if (!image.hasAttribute('src')) image.src = image.dataset.src;
    } else if (!toggle.checked) image.removeAttribute('src');
  }
}
toggle.addEventListener('change', syncImages);
const linked = document.getElementById(location.hash.slice(1));
if (linked?.tagName === 'DETAILS') linked.open = true;
for (const section of document.querySelectorAll('details')) section.addEventListener('toggle', syncImages);
</script></html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as out:
        out.write(gallery(json.loads(args.replay.read_text())))
