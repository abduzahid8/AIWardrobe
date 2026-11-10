/**
 * image-teaste check against the ACTUAL catalog (offline snapshot universe).
 *
 * For each of the 46 verified reference blueprints, counts how many adapted
 * catalog products can fill each garment role the reference demands. This is
 * the reference-to-shelf coverage map: it names exactly which looks are
 * shoppable, which are thin, and which have hard gaps — per slot, with the
 * wearing-role projections (knit base layer, live pullover) applied.
 *
 * Inputs: verified image_teaste_46.json blueprints + adapted snapshot items.
 * No network, no keys, no uploads. Thresholds are reported, not hidden:
 *   shoppable: every required slot has >=12 candidates
 *   thin:      every required slot has >=3 but some slot <12
 *   gap:       any required slot has <3 candidates
 *
 * Usage:
 *   node scripts/stylist6/reference_catalog_check.mjs [--report <path>]
 * Prints JSON summary; --report writes the full per-reference table.
 */

import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '../..');
const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => {
    if (!a.startsWith('--')) return [];
    const eq = a.indexOf('=');
    if (eq > 0) return [[a.slice(2, eq), a.slice(eq + 1)]];
    const next = all[i + 1];
    return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
  }),
);

const index = JSON.parse(fs.readFileSync(
  path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json'), 'utf8'));
const { adaptCatalog } = await import('../../backend/stylist6/catalog_adapter.mjs');

const rows = [];
for (const file of fs.readdirSync(path.join(root, 'data'))) {
  if (!file.startsWith('shop_catalog_') || !file.endsWith('.json')) continue;
  rows.push(...JSON.parse(fs.readFileSync(path.join(root, 'data', file), 'utf8')));
}
const adapted = adaptCatalog(rows);
const bySlotSubtype = new Map();
for (const item of adapted.items) {
  const k = `${item.macroCategory}/${item.subtype}`;
  if (!bySlotSubtype.has(k)) bySlotSubtype.set(k, []);
  bySlotSubtype.get(k).push(item.id);
}

// Exact per-reference briefs (no relevance ranking): every one of the 46
// references is checked, not just the top-ranked subset.
const { briefForReferenceId } = await import('../../backend/stylist6/lookbook_briefs.mjs');
const table = [];
for (const ref of index.images) {
  let brief = null;
  try {
    brief = briefForReferenceId(index, ref.id);
  } catch {
    table.push({ referenceId: ref.id, status: 'no_brief', slots: {} });
    continue;
  }
  const slots = {};
  let worst = Infinity;
  for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) {
    if (brief.layerPolicy === 'absent' && slot === 'outerwear') {
      slots[slot] = { demanded: [], candidates: 0, skipped: true };
      continue;
    }
    const demanded = brief.roles[slot] || [];
    let count = 0;
    const perSubtype = {};
    for (const subtype of demanded) {
      const ids = bySlotSubtype.get(`${slot}/${subtype}`) || [];
      perSubtype[subtype] = ids.length;
      count += ids.length;
    }
    slots[slot] = { demanded, candidates: count, perSubtype };
    if (count < worst) worst = count;
  }
  const status = worst >= 12 ? 'shoppable' : worst >= 3 ? 'thin' : 'gap';
  table.push({
    referenceId: ref.id, title: ref.title, status,
    layerPolicy: brief.layerPolicy, slots,
    knitBaseLayer: /knit|turtleneck|mockneck|roll-neck|crewneck|sweater/i.test(
      `${ref.garments?.top || ''} ${ref.garments?.outerwear || ''}`),
  });
}

const summary = {
  tool: 'reference-catalog-check-v1',
  references: table.length,
  shoppable: table.filter((r) => r.status === 'shoppable').length,
  thin: table.filter((r) => r.status === 'thin').length,
  gap: table.filter((r) => r.status === 'gap').length,
  noBrief: table.filter((r) => r.status === 'no_brief').length,
  knitBaseLayerLooks: table.filter((r) => r.knitBaseLayer).length,
  snapshotAdmitted: adapted.items.length,
  byProjection: adapted.byProjection,
};
console.log(JSON.stringify(summary, null, 1));
if (args.report) {
  fs.writeFileSync(args.report, JSON.stringify({ summary, table }, null, 1) + '\n');
  console.log('wrote ' + args.report);
}
const gaps = table.filter((r) => r.status === 'gap' || r.status === 'no_brief');
if (gaps.length) {
  console.log('gaps:');
  for (const g of gaps.slice(0, 20)) {
    const worst = Object.entries(g.slots || {}).map(([s, v]) => `${s}:${v.candidates ?? '-'}`).join(' ');
    console.log(` ${g.referenceId} ${g.title || ''} [${worst}]`);
  }
}
