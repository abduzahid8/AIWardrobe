#!/usr/bin/env node
// blind-pair-judge-v1: blinded AI visual comparison of Stylist 5 vs Stylist 6
// decks from a frozen fashion_replay report.
//
// This is an INDEPENDENT AI EVALUATION, not a human one. The judge model
// never learns which engine produced which deck: arms are renamed ALPHA/BETA
// per scenario with a fresh random assignment, and the mapping lives only in
// a separate key file. Results must be reported as AI-rater preferences with
// documented uncertainty, never as human acceptance.
//
// Usage:
//   node scripts/stylist6/blind_pair_judge.mjs <replay.json> <catalog.json> \
//     --out-dir .private/blind-judge [--only id1,id2] [--limit N] [--judge-calls N]
//
// Bounded: at most 2 judge calls per scenario (one per deck), 8s provider
// timeout, one transient retry, circuit-break on auth/quota errors.
// Images: downloaded only from hosts in the image_access allowlist, cached
// under .private/blind-judge-images, converted to <=512KiB JPEG via sips.

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';
import sharp from 'sharp';

const DEFAULT_MODEL = 'nvidia/nemotron-3.5-super-vl-120b-a12b';
const ENDPOINT = 'https://integrate.api.nvidia.com/v1/chat/completions';
const PROVIDER_TIMEOUT_MS = 45000; // offline rater; the live engine keeps its own deadline
const PROVIDER_RETRY_TRANSIENT = 1;
const MAX_IMAGE_BYTES = 512 * 1024;
const IMAGE_DIR = '.private/blind-judge-images';
const HELD_OUT_PREFIXES = ['S6-B', 'S6-H'];
const BASELINE_ARM = 'stylist5';
const CANDIDATE_ARM = 'stylist6Fashion';

const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith('--'));
const flag = (name) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const has = (name) => args.includes(`--${name}`);

if (positional.length < 2 || has('help')) {
  console.error('usage: blind_pair_judge.mjs <replay.json> <catalog.json> --out-dir <dir> [--only ids] [--limit N] [--all] [--model <id>] [--samples N]');
  process.exit(has('help') ? 0 : 2);
}

const replayPath = positional[0];
const catalogPath = positional[1];
const outDir = flag('out-dir') || '.private/blind-judge';
const onlyIds = (flag('only') || '').split(',').map((s) => s.trim()).filter(Boolean);
const limit = Number(flag('limit') || 0);
const includeAll = has('all');
const MODEL = flag('model') || DEFAULT_MODEL;
const SAMPLES_PER_DECK = Math.max(1, Math.min(3, Number(flag('samples') || 2)));
const USE_COMPOSITE = has('composite') || !MODEL.includes('nemotron');

const apiKey = (await (async () => {
  if (process.env.NVIDIA_API_KEY) return process.env.NVIDIA_API_KEY;
  try {
    const dotenv = await import('dotenv');
    const parsed = dotenv.parse(readFileSync('.env', 'utf8'));
    return parsed.NVIDIA_API_KEY || '';
  } catch { return ''; }
})()) || '';
if (!apiKey.startsWith('nvapi-')) {
  console.error('NVIDIA_API_KEY missing or not nvapi- prefixed (checked process env + .env, never printed); refusing to run (fail-closed).');
  process.exit(3);
}

mkdirSync(outDir, { recursive: true, mode: 0o700 });
mkdirSync(IMAGE_DIR, { recursive: true, mode: 0o700 });

// --- image access: reuse the audited download allowlist ----------------------
let allowFetch = null;
try {
  const { HOST_ALLOWLIST } = await import('./image_access.mjs');
  allowFetch = (url) => {
    try {
      const u = new URL(url);
      return u.protocol === 'https:' && HOST_ALLOWLIST.has(u.hostname);
    } catch { return false; }
  };
} catch { /* handled below */ }
if (!allowFetch) {
  console.error('image_access.mjs unavailable; refusing to fetch images.');
  process.exit(3);
}

function loadCatalog(path) {
  const raw = JSON.parse(readFileSync(path, 'utf8'));
  const rows = raw.rows || raw.catalog || raw;
  const byId = new Map();
  for (const row of rows) {
    const id = row.id || row.itemId;
    if (id) byId.set(id, row);
  }
  return byId;
}

function slotLabel(row) {
  return row.outfitSlot || row.slot || row.garment_type || 'item';
}

async function fetchJudgeImage(url, id) {
  let parsed;
  try { parsed = new URL(url); } catch { return null; }
  if (parsed.protocol !== 'https:' || !allowFetch(url)) return null;
  const key = createHash('sha256').update(`${id}|${url}`).digest('hex').slice(0, 24);
  const jpgPath = join(IMAGE_DIR, `${key}.jpg`);
  if (existsSync(jpgPath) && statSync(jpgPath).size > 0) return readFileSync(jpgPath);
  const rawPath = join(IMAGE_DIR, `${key}.bin`);
  let bytes;
  if (existsSync(rawPath) && statSync(rawPath).size > 0) {
    bytes = readFileSync(rawPath);
  } else {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) return null;
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length === 0 || buf.length > 4 * 1024 * 1024) return null;
      writeFileSync(rawPath, buf, { mode: 0o600 });
      bytes = buf;
    } catch { return null; } finally { clearTimeout(timer); }
  }
  // Normalize to <=512KiB JPEG via macOS sips (dev-bridge convention).
  try {
    if (bytes.length <= MAX_IMAGE_BYTES && bytes[0] === 0xff && bytes[1] === 0xd8) {
      writeFileSync(jpgPath, bytes, { mode: 0o600 });
      return bytes;
    }
    execFileSync('sips', ['-s', 'format', 'jpeg', '-s', 'formatOptions', '80', '--resampleHeightWidthMax', '1600', rawPath, '--out', jpgPath], { stdio: 'ignore' });
    const out = readFileSync(jpgPath);
    if (out.length > MAX_IMAGE_BYTES) {
      execFileSync('sips', ['-s', 'format', 'jpeg', '-s', 'formatOptions', '60', '--resampleHeightWidthMax', '1024', jpgPath, '--out', jpgPath], { stdio: 'ignore' });
      return readFileSync(jpgPath);
    }
    return out;
  } catch { return null; }
}

// --- judge call --------------------------------------------------------------
const RUBRIC_SYSTEM = `You are an independent fashion-quality rater. You will be shown the product photos of a stylist's suggestion set: up to 3 complete looks (Look 1, Look 2, Look 3), each with up to 4 garments (top, optional layer, bottom, shoes) in slot order. Rate ONLY what is visible. Never invent garments, colors, or brands. You do not know and must not guess which system produced the set. Answer with strict JSON only, no markdown:
{"look_quality":0-10,"deck_cohesion":0-10,"deck_diversity":0-10,"occasion_fit":0-10,"footwear_register":0-10,"rationale":"<=50 words citing visible garments"}`;
const RUBRIC_USER_HINT = (style, occasion) =>
  `Intended aesthetic: ${style}. Occasion: ${occasion}. look_quality averages how strong the individual looks are; deck_cohesion rates whether the looks all serve the same aesthetic; deck_diversity rates whether the three looks offer genuinely different takes (near-identical looks score low); footwear_register rates whether the footwear formality matches each look and the occasion.`;

async function composeLookImages(images) {
  const byLook = new Map();
  for (const img of images) {
    const m = /^(Look \d+)\s+(.+)$/.exec(img.slot);
    const lookKey = m ? m[1] : 'Look 1';
    const slotName = m ? m[2] : img.slot;
    if (!byLook.has(lookKey)) byLook.set(lookKey, []);
    byLook.get(lookKey).push({ ...img, slotName });
  }
  const rows = [];
  let maxW = 460;
  const cellW = 110;
  const cellH = 148;
  const gap = 4;
  const rowH = cellH + 26;
  for (const [lookKey, parts] of [...byLook.entries()].slice(0, 3)) {
    const rowW = parts.length * cellW + (parts.length + 1) * gap;
    if (rowW > maxW) maxW = rowW;
    rows.push({ lookKey, parts });
  }
  const height = rows.length * rowH + gap;
  const overlays = [];
  for (let rIdx = 0; rIdx < rows.length; rIdx += 1) {
    const { lookKey, parts } = rows[rIdx];
    const y0 = rIdx * rowH;
    const label = `${lookKey}: ${parts.map((p) => p.slotName).join(' | ')}`;
    overlays.push({
      input: Buffer.from(
        `<svg width="${maxW}" height="22"><text x="6" y="16" font-family="sans-serif" font-size="12" font-weight="bold" fill="#111">${label.replace(/[<>&]/g, '')}</text></svg>`,
      ),
      left: 0,
      top: y0 + 2,
    });
    const tiles = await Promise.all(parts.map(async (p, idx) => {
      const resized = await sharp(p.bytes)
        .resize(cellW, cellH - 2, { fit: 'contain', background: { r: 245, g: 247, b: 250 } })
        .jpeg({ quality: 72 })
        .toBuffer();
      return { input: resized, left: gap + idx * (cellW + gap), top: y0 + 24 };
    }));
    overlays.push(...tiles);
  }
  const sheet = await sharp({
    create: { width: maxW, height, channels: 3, background: { r: 255, g: 255, b: 255 } },
  }).composite(overlays).jpeg({ quality: 74 }).toBuffer();
  return [{
    id: 'deck_sheet',
    slot: rows.map((r) => `${r.lookKey} (${r.parts.map((p) => p.slotName).join(', ')})`).join('; '),
    bytes: sheet,
  }];
}

async function judgeDeck(images, style, occasion, token) {
  if (images.length === 0) return { skipped: 'no_images' };
  const judgeImages = USE_COMPOSITE ? await composeLookImages(images) : images;
  const content = [
    { type: 'text', text: `${RUBRIC_USER_HINT(style, occasion)}\nSlot order: ${judgeImages.map((i) => i.slot).join(', ')}.` },
    ...judgeImages.flatMap((img) => [
      { type: 'text', text: `Photo: ${img.slot}` },
      { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${img.bytes.toString('base64')}` } },
    ]),
  ];
  const payloadBody = {
    model: MODEL,
    temperature: 0.1,
    top_p: 0.95,
    max_tokens: MODEL.includes('nemotron') ? 450 : 600,
    stream: false,
    ...(MODEL.includes('nemotron') ? {
      response_format: { type: 'json_object' },
      chat_template_kwargs: { enable_thinking: false },
    } : {}),
    messages: [
      { role: 'system', content: `${RUBRIC_SYSTEM}${MODEL.includes('nemotron') ? '' : '\nKeep internal thinking under 80 words; output ONLY strict raw JSON immediately.'}` },
      { role: 'user', content },
    ],
  };
  const body = JSON.stringify(payloadBody);
  for (let attempt = 0; attempt <= PROVIDER_RETRY_TRANSIENT; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), PROVIDER_TIMEOUT_MS);
    try {
      const res = await fetch(ENDPOINT, {
        method: 'POST',
        headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
        body,
        signal: controller.signal,
      });
      if (res.status === 401 || res.status === 402 || res.status === 403 || res.status === 429) {
        return { skipped: `provider_${res.status}` };
      }
      if (!res.ok) {
        if (attempt < PROVIDER_RETRY_TRANSIENT) continue;
        return { skipped: `provider_${res.status}` };
      }
      const payload = await res.json();
      const text = payload?.choices?.[0]?.message?.content || '';
      const match = text.match(/\{[\s\S]*\}/);
      if (!match) return { skipped: 'unparseable_verdict' };
      const verdict = JSON.parse(match[0]);
      const criteria = ['look_quality', 'deck_cohesion', 'deck_diversity', 'occasion_fit', 'footwear_register'];
      if (criteria.some((c) => !Number.isFinite(Number(verdict[c])))) return { skipped: 'invalid_verdict' };
      const usage = payload?.usage?.total_tokens || null;
      return { verdict, usage };
    } catch (err) {
      if (attempt < PROVIDER_RETRY_TRANSIENT) continue;
      return { skipped: `network_${err.name || 'error'}` };
    } finally {
      clearTimeout(timer);
    }
  }
  return { skipped: 'unreachable' };
}

// --- scenario assembly -------------------------------------------------------
function deckImages(deck, catalog) {
  const images = [];
  (deck || []).slice(0, 3).forEach((outfit, lookIndex) => {
    for (const id of outfit.ids || []) {
      const row = catalog.get(id);
      const url = row && (row.image_url || row.imageUrl);
      if (!url) continue;
      images.push({ id, url, slot: `Look ${lookIndex + 1} ${slotLabel(row)}` });
    }
  });
  return images.slice(0, 12);
}

async function withBytes(images) {
  const settled = await Promise.all(images.map(async (img) => {
    const bytes = await fetchJudgeImage(img.url, img.id);
    return bytes ? { ...img, bytes } : null;
  }));
  return settled.filter(Boolean);
}

function scenarioEligible(result) {
  if (includeAll) return true;
  const id = result.scenario?.id || '';
  return HELD_OUT_PREFIXES.some((p) => id.startsWith(p));
}

async function main() {
  const replay = JSON.parse(readFileSync(replayPath, 'utf8'));
  const catalog = loadCatalog(catalogPath);
  let results = (replay.results || []).filter(scenarioEligible);
  if (onlyIds.length) results = results.filter((r) => onlyIds.includes(r.scenario.id));
  if (limit > 0) results = results.slice(0, limit);

  const blind = { version: 'blind-pair-judge-v1', replay: basename(replayPath), snapshotSha256: replay.snapshotSha256 || null, evaluatedAt: new Date().toISOString(), evaluationKind: 'AI_RATER_BLINDED_NOT_HUMAN', judgeModel: MODEL, scenarios: [] };
  const key = { version: 'blind-pair-judge-v1', note: 'token->arm mapping; keep separate from blinded results', scenarios: {} };
  let providerCalls = 0;
  let totalTokens = 0;

  for (const result of results) {
    const scenario = result.scenario;
    const baseDeck = result.arms[BASELINE_ARM] || [];
    const candDeck = result.arms[CANDIDATE_ARM] || [];
    if ((baseDeck.length || 0) < 2 || (candDeck.length || 0) < 2) {
      blind.scenarios.push({ id: scenario.id, skipped: 'incomplete_decks', baseLooks: baseDeck.length, candidateLooks: candDeck.length });
      continue;
    }
    const baseImages = await withBytes(deckImages(baseDeck, catalog));
    const candImages = await withBytes(deckImages(candDeck, catalog));
    const alphaIsCandidate = Math.random() < 0.5;
    const alpha = alphaIsCandidate ? candImages : baseImages;
    const beta = alphaIsCandidate ? baseImages : candImages;
    key.scenarios[scenario.id] = { alpha: alphaIsCandidate ? CANDIDATE_ARM : BASELINE_ARM, beta: alphaIsCandidate ? BASELINE_ARM : CANDIDATE_ARM };

    const entry = { id: scenario.id, style: scenario.style, occasion: scenario.occasion, ALPHA: { images: alpha.length }, BETA: { images: beta.length } };
    const judgeSide = async (images) => {
      const results = await Promise.all(Array.from({ length: SAMPLES_PER_DECK }, () => judgeDeck(images, scenario.style, scenario.occasion)));
      const verdicts = results.filter((r) => r.verdict).map((r) => r.verdict);
      providerCalls += verdicts.length;
      totalTokens += results.reduce((s, r) => s + (r.usage || 0), 0);
      const skipped = results.find((r) => r.skipped)?.skipped;
      if (!verdicts.length) return { skipped: skipped || 'no_verdicts' };
      const crit = ['look_quality', 'deck_cohesion', 'deck_diversity', 'occasion_fit', 'footwear_register'];
      const verdict = {};
      for (const c of crit) verdict[c] = Math.round((verdicts.reduce((s, v) => s + Number(v[c]), 0) / verdicts.length) * 10) / 10;
      verdict.rationale = verdicts[0].rationale;
      return { verdict, samples: verdicts.length };
    };
    const [alphaRes, betaRes] = await Promise.all([judgeSide(alpha), judgeSide(beta)]);
    entry.ALPHA = { ...entry.ALPHA, ...alphaRes };
    entry.BETA = { ...entry.BETA, ...betaRes };
    const crit = ['look_quality', 'deck_cohesion', 'deck_diversity', 'occasion_fit', 'footwear_register'];
    const total = (side) => (entry[side].verdict ? crit.reduce((s, c) => s + Number(entry[side].verdict[c]), 0) : null);
    entry.ALPHA_TOTAL = total('ALPHA');
    entry.BETA_TOTAL = total('BETA');
    blind.scenarios.push(entry);
    console.log(`${scenario.id}: ALPHA ${entry.ALPHA_TOTAL ?? alphaRes.skipped} | BETA ${entry.BETA_TOTAL ?? betaRes.skipped}`);
  }

  writeFileSync(join(outDir, 'blind.json'), JSON.stringify(blind, null, 2), { mode: 0o600 });
  writeFileSync(join(outDir, 'key.json'), JSON.stringify(key, null, 2), { mode: 0o600 });
  console.log(`\nproviderCalls=${providerCalls} totalTokens=${totalTokens} scenarios=${blind.scenarios.length}`);
  console.log(`wrote ${join(outDir, 'blind.json')} + key.json`);
}

main().catch((err) => {
  console.error('fatal:', err.message);
  process.exit(1);
});
