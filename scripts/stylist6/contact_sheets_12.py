"""Anonymous side-by-side visual grids for Stylist 6 12-scenario pilot.

Reads the PRIVATE pilot output (never committed) and renders, per scenario:
  1. blind grid  — 3 anonymized columns (arms shuffled with a fixed seed;
     the key is stored separately in the private out dir), 3 looks each,
     4 real product photos per look with name/price/currency/SKU.
  2. reference panel — the lookbook references arm C matched, next to the
     same candidate photos (grounding view, NOT blind).

Zero-look arms render an explicit NO-COMPLIANT-LOOK cell: abstention is
evidence, not a rendering bug. No AI calls, no uploads, no secrets.
"""
import json
import sys
import random
import hashlib
import pathlib
import urllib.request
import concurrent.futures
from PIL import Image, ImageOps, ImageDraw

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/s6-pilot-20261110')
SEED = 20261110
REFDIR = pathlib.Path.home() / 'Downloads' / 'Telegram Desktop'

data = json.load(open(OUT / 'pilot-private.json'))
index = json.load(open('benchmarks/stylist6/references/image_teaste_46.json'))
ref_file = {im['id']: im['filename'] for im in index['images']}

cache = OUT / 'review-images'
cache.mkdir(exist_ok=True)

products = {}
for run in data['runs']:
    for arm in run['arms'].values():
        for look in arm.get('looks', []) or []:
            for item in look['items']:
                products[item['id']] = item


def fetch(item):
    key = hashlib.sha256(item['imageUrl'].encode()).hexdigest() + '.jpg'
    path = cache / key
    if path.exists():
        return item['id'], path
    try:
        req = urllib.request.Request(item['imageUrl'], headers={'User-Agent': 'AIWardrobe-Stylist6-Pilot/1.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read(3 * 1024 * 1024)
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(raw)
        img = Image.open(tmp)
        img.thumbnail((480, 600))
        img.convert('RGB').save(path, quality=88)
        tmp.unlink(missing_ok=True)
        return item['id'], path
    except Exception:
        return item['id'], None


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    paths = dict(pool.map(fetch, products.values()))

unavailable = sum(p is None for p in paths.values())
print(json.dumps({'products': len(products), 'unavailable': unavailable}))

COL_W, LOOK_H, ITEM_W, ITEM_H = 600, 900, 285, 350


def draw_look(draw, sheet, x0, y0, look, label):
    draw.text((x0, y0), label[:40], fill='black')
    y = y0 + 22
    items = look['items']
    if not items:
        draw.text((x0 + 60, y + 120), 'NO COMPLIANT LOOK', fill='red')
        return
    for k, item in enumerate(items):
        x = x0 + (k % 2) * (ITEM_W + 10)
        yy = y + (k // 2) * (ITEM_H + 62)
        p = paths.get(item['id'])
        if p:
            try:
                img = ImageOps.contain(Image.open(p).convert('RGB'), (ITEM_W, ITEM_H - 70))
                sheet.paste(img, (x + (ITEM_W - img.width) // 2, yy))
            except Exception:
                draw.text((x + 25, yy + 100), 'IMAGE UNAVAILABLE', fill='red')
        else:
            draw.text((x + 25, yy + 100), 'IMAGE UNAVAILABLE', fill='red')
        caption = item['name'][:34] + f"\n{item['price']:,.0f} {item['currency']}\n{item['id'][:40]}"
        draw.text((x, yy + ITEM_H - 58), caption, fill='black')


key = {}
for n, run in enumerate(data['runs']):
    arms = [a for a in ('v5-baseline', 'creative-first', 'reference-first') if a in run['arms']]
    rng = random.Random(f'{SEED}-{run["id"]}')
    order = arms[:]
    rng.shuffle(order)
    labels = [chr(ord('A') + i) for i in range(len(order))]
    key[run['id']] = {label: arm for label, arm in zip(labels, order)}

    sheet = Image.new('RGB', (COL_W * 3 + 30, 60 + LOOK_H * 3 + 40), '#f1efe9')
    draw = ImageDraw.Draw(sheet)
    budget = next((s.get('budget') for s in data['scenarios'] if s['id'] == run['id']), None)
    title = run['id'] + ' blind outfits (3 systems x 3 looks, real SKUs)'
    draw.text((15, 12), title, fill='black')
    for ci, (label, arm) in enumerate(zip(labels, order)):
        x0 = 15 + ci * COL_W
        draw.text((x0, 34), f'System {label}', fill='black')
        for li, look in enumerate((run['arms'][arm].get('looks') or [])[:3]):
            draw_look(draw, sheet, x0, 60 + li * LOOK_H, look, f'Look {li + 1}')
        if not (run['arms'][arm].get('looks') or []):
            draw.text((x0 + 60, 200), 'NO COMPLIANT LOOK', fill='red')
    sheet.save(OUT / f'blind-{run["id"]}.jpg', quality=88)

    # Reference grounding panel (not blind): references used by arm C + its looks.
    clooks = (run['arms'].get('reference-first', {}).get('looks') or [])[:3]
    ref_ids = [look.get('referenceId') for look in clooks if look.get('referenceId')]
    panel = Image.new('RGB', (COL_W * 3 + 30, 60 + 420 + LOOK_H * 3 + 40), '#e9e7e2')
    pdraw = ImageDraw.Draw(panel)
    pdraw.text((15, 12), run['id'] + ' reference grounding (NOT blind)', fill='black')
    for ri, rid in enumerate(ref_ids[:3]):
        src = REFDIR / ref_file.get(rid, '')
        x0 = 15 + ri * COL_W
        pdraw.text((x0, 34), f'{rid}', fill='black')
        if src.exists():
            try:
                img = ImageOps.contain(Image.open(src).convert('RGB'), (COL_W - 20, 400))
                panel.paste(img, (x0, 60))
            except Exception:
                pdraw.text((x0 + 60, 200), 'REF UNAVAILABLE', fill='red')
        else:
            pdraw.text((x0 + 60, 200), 'REF MISSING', fill='red')
    for li, look in enumerate(clooks):
        draw_look(pdraw, panel, 15, 60 + 420 + li * LOOK_H, look,
                  f"C look {li + 1} ({look.get('referenceId') or look.get('concept') or ''})")
    panel.save(OUT / f'grounded-{run["id"]}.jpg', quality=88)

json.dump(key, open(OUT / 'blind-key.json', 'w'), indent=2)
print(json.dumps({'sheets': len(data['runs']) * 2, 'key': str(OUT / 'blind-key.json')}))
