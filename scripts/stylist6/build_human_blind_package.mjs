#!/usr/bin/env node
/**
 * Builds an auditable, offline blind human acceptance package (>=30 scenarios)
 * comparing Stylist 6 Fashion vs Stylist 5 from a frozen fashion_replay report,
 * and scores independent human reviewer exports into release_candidate_gate JSON.
 *
 * Rules:
 * - Never fabricates human evaluations or claims acceptance from model scores.
 * - Randomizes A/B side assignment per scenario and stores the mapping ONLY
 *   in a separate blind_key.json (mode 0o600), never embedded in index.html.
 * - Downloads/copies product photos only from HOST_ALLOWLIST hosts, normalized
 *   to bounded JPEG assets inside <outDir>/assets/ with SHA-256 verification.
 *
 * Usage:
 *   Build: node scripts/stylist6/build_human_blind_package.mjs build <replay.json> <outDir> [--min 30] [--limit 49]
 *   Score: node scripts/stylist6/build_human_blind_package.mjs score <blind_key.json> <sub1.json> <sub2.json> <sub3.json>
 */
import { createHash, randomInt } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { HOST_ALLOWLIST } from './image_access.mjs';

const BASELINE_ARM = 'stylist5';
const CANDIDATE_ARM = 'stylist6Fashion';
const SLOT_ORDER = ['top', 'outerwear', 'bottom', 'shoes'];
const MAX_IMAGE_BYTES = 512 * 1024;
const DEFAULT_CACHE_DIR = '.private/blind-judge-images';

const escapeHtml = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

export function isAllowlistedImageUrl(rawUrl) {
  try {
    const u = new URL(String(rawUrl || ''));
    return u.protocol === 'https:' && !u.username && !u.password && !u.port
      && HOST_ALLOWLIST.has(u.hostname.toLowerCase());
  } catch {
    return false;
  }
}

export function selectComparableThreeLookScenarios(replay, { limit = 0 } = {}) {
  const results = Array.isArray(replay?.results) ? replay.results : [];
  const eligible = results.filter((row) =>
    Array.isArray(row?.arms?.[BASELINE_ARM]) && row.arms[BASELINE_ARM].length === 3
    && Array.isArray(row?.arms?.[CANDIDATE_ARM]) && row.arms[CANDIDATE_ARM].length === 3,
  );
  // Prioritize held-out scenarios first, then deterministic order for the rest.
  const heldOut = eligible.filter((row) => String(row?.scenario?.id || '').startsWith('S6-H'));
  const others = eligible.filter((row) => !String(row?.scenario?.id || '').startsWith('S6-H'));
  const ordered = [...heldOut, ...others];
  return limit > 0 ? ordered.slice(0, limit) : ordered;
}

async function resolveProductPhoto(sku, product, { cacheDir, fetchImpl }) {
  const imageUrl = product?.imageUrl;
  if (!isAllowlistedImageUrl(imageUrl)) {
    throw new Error(`unallowlisted_image_host:${sku}`);
  }
  const cacheKey = sha256(`${sku}|${imageUrl}`).slice(0, 24);
  const cachedJpg = path.join(cacheDir, `${cacheKey}.jpg`);
  if (existsSync(cachedJpg) && statSync(cachedJpg).size > 0) {
    const bytes = readFileSync(cachedJpg);
    return { bytes, sha256: sha256(bytes) };
  }
  const response = await fetchImpl(imageUrl, {
    redirect: 'error',
    signal: AbortSignal.timeout(12_000),
    headers: { 'User-Agent': 'AIWardrobe-BlindReview-Builder/1.0' },
  });
  if (!response?.ok) {
    throw new Error(`photo_fetch_failed:${sku}:${response?.status || 'err'}`);
  }
  const raw = Buffer.from(await response.arrayBuffer());
  if (raw.length === 0 || raw.length > 4 * 1024 * 1024) {
    throw new Error(`photo_size_invalid:${sku}`);
  }
  const normalized = await sharp(raw)
    .rotate()
    .flatten({ background: { r: 245, g: 247, b: 250 } })
    .resize(720, 720, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 80, mozjpeg: true })
    .toBuffer();
  if (normalized.length > MAX_IMAGE_BYTES) {
    throw new Error(`photo_normalized_oversize:${sku}`);
  }
  mkdirSync(cacheDir, { recursive: true, mode: 0o700 });
  writeFileSync(cachedJpg, normalized, { mode: 0o600 });
  return { bytes: normalized, sha256: sha256(normalized) };
}

function renderSelect(name, label, options) {
  const opts = options.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  return `<label>${escapeHtml(label)} <select name="${escapeHtml(name)}" required><option value="">Choose</option>${opts}</select></label>`;
}

export async function buildHumanBlindPackage({
  replayPath,
  outDir,
  cacheDir = DEFAULT_CACHE_DIR,
  minScenarios = 30,
  limit = 0,
  fetchImpl = fetch,
  rng = (n) => randomInt(n),
  gitSha = null,
}) {
  const replayBytes = readFileSync(replayPath);
  const replay = JSON.parse(replayBytes.toString('utf8'));
  const replaySha256 = sha256(replayBytes);
  const catalog = replay.catalog || {};
  const candidates = selectComparableThreeLookScenarios(replay, { limit });
  if (candidates.length < minScenarios) {
    throw new Error(`insufficient_comparable_3look_scenarios:${candidates.length}<${minScenarios}`);
  }

  const assetsDir = path.join(outDir, 'assets');
  mkdirSync(outDir, { recursive: true, mode: 0o700 });
  mkdirSync(assetsDir, { recursive: true, mode: 0o700 });

  const mediaBySha = new Map();
  const skuAssets = new Map();
  const resolveSkuAsset = async (sku) => {
    if (skuAssets.has(sku)) return skuAssets.get(sku);
    const product = catalog[sku];
    if (!product) throw new Error(`missing_catalog_sku:${sku}`);
    const photo = await resolveProductPhoto(sku, product, { cacheDir, fetchImpl });
    const filename = `${photo.sha256}.jpg`;
    if (!mediaBySha.has(photo.sha256)) {
      mediaBySha.set(photo.sha256, filename);
      writeFileSync(path.join(assetsDir, filename), photo.bytes, { mode: 0o600 });
    }
    const entry = { filename, sha256: photo.sha256, product };
    skuAssets.set(sku, entry);
    return entry;
  };

  // Counterbalance A/B flips evenly across scenarios while keeping order random.
  const flips = Array.from({ length: candidates.length }, (_, i) => i % 2 === 1);
  for (let i = flips.length - 1; i > 0; i -= 1) {
    const j = rng(i + 1);
    [flips[i], flips[j]] = [flips[j], flips[i]];
  }

  const keyCases = {};
  const manifestScenarios = [];
  const scenarioHtmlBlocks = [];

  for (let idx = 0; idx < candidates.length; idx += 1) {
    const row = candidates[idx];
    const sc = row.scenario;
    const scenarioId = String(sc.id);
    const flip = flips[idx];
    const assignment = flip
      ? { A: CANDIDATE_ARM, B: BASELINE_ARM }
      : { A: BASELINE_ARM, B: CANDIDATE_ARM };
    keyCases[scenarioId] = assignment;

    const armHtml = {};
    const deckDigests = {};
    for (const label of ['A', 'B']) {
      const armName = assignment[label];
      const looks = row.arms[armName];
      const lookBlocks = [];
      const lookPayloads = [];
      for (let lookIdx = 0; lookIdx < looks.length; lookIdx += 1) {
        const look = looks[lookIdx];
        const sortedIds = [...look.ids].sort((a, b) =>
          SLOT_ORDER.indexOf(catalog[a]?.slot) - SLOT_ORDER.indexOf(catalog[b]?.slot));
        const itemFigs = [];
        const itemDigests = [];
        for (const sku of sortedIds) {
          const { filename, sha256: photoHash, product } = await resolveSkuAsset(sku);
          itemDigests.push({ slot: product.slot, sku, photoSha256: photoHash, price: product.price, currency: product.currency });
          const caption = `${product.slot} · ${product.name}`;
          itemFigs.push(
            `<figure><img src="assets/${escapeHtml(filename)}" alt="${escapeHtml(caption)}" loading="lazy">` +
            `<figcaption>${escapeHtml(caption)}<br>${Number(product.price).toLocaleString('en-US')} ${escapeHtml(product.currency)}</figcaption></figure>`,
          );
        }
        lookPayloads.push(itemDigests);
        lookBlocks.push(`<article><h3>Deck ${label} · Look ${lookIdx + 1}</h3><div class="items">${itemFigs.join('')}</div></article>`);
      }
      deckDigests[label] = lookPayloads;

      const rubricPrefix = `${scenarioId}.${label}`;
      const rubricFields = [
        renderSelect(`${rubricPrefix}.look_quality`, 'Look quality (1–5)', ['1', '2', '3', '4', '5']),
        renderSelect(`${rubricPrefix}.deck_cohesion`, 'Deck cohesion (1–5)', ['1', '2', '3', '4', '5']),
        renderSelect(`${rubricPrefix}.deck_diversity`, 'Outfit diversity (1–5)', ['1', '2', '3', '4', '5']),
        renderSelect(`${rubricPrefix}.occasion_fit`, 'Style & occasion fit (1–5)', ['1', '2', '3', '4', '5']),
        renderSelect(`${rubricPrefix}.footwear_register`, 'Footwear appropriateness (1–5)', ['1', '2', '3', '4', '5']),
        renderSelect(`${rubricPrefix}.critical`, 'Critical issue in deck?', ['No', 'Yes']),
        `<label>Critical issue detail <input name="${escapeHtml(`${rubricPrefix}.note`)}" maxlength="400" placeholder="Required if Yes"></label>`,
      ].join('');

      armHtml[label] = `<section class="deck-col"><h2>Deck ${label}</h2><div class="looks">${lookBlocks.join('')}</div><fieldset class="rubric"><legend>Deck ${label} rubric</legend>${rubricFields}</fieldset></section>`;
    }

    const auditSha256 = sha256(JSON.stringify({
      scenarioId,
      style: sc.style,
      occasion: sc.occasion,
      decks: deckDigests,
    }));
    manifestScenarios.push({
      id: scenarioId,
      style: sc.style,
      occasion: sc.occasion,
      heldOut: scenarioId.startsWith('S6-H'),
      weatherTemp: sc.weather?.temp ?? null,
      budget: sc.budget ?? null,
      auditSha256,
    });

    const budgetText = sc.budget ? `${Number(sc.budget.amount).toLocaleString('en-US')} ${sc.budget.currency}` : 'No price cap';
    scenarioHtmlBlocks.push(
      `<details><summary>${idx + 1}. ${escapeHtml(scenarioId)} · ${escapeHtml(sc.style.replace(/_/g, ' '))} · ${escapeHtml(sc.occasion)} (${escapeHtml(sc.weather?.temp ?? '?')}°C · ${escapeHtml(budgetText)})</summary>` +
      `<div class="pair">${armHtml.A}${armHtml.B}</div>` +
      `<div class="pref">${renderSelect(`${scenarioId}.preference`, 'Which complete 3-look deck would you rather receive?', ['A', 'B', 'Tie'])}</div>` +
      `</details>`,
    );
  }

  const packageBody = {
    schema: 'aiw.stylist6.human_blind_package.v1',
    generatedAt: new Date().toISOString(),
    gitSha,
    replaySha256,
    scenarioCount: candidates.length,
    heldOutCount: manifestScenarios.filter((s) => s.heldOut).length,
    scenarios: manifestScenarios,
  };
  const packageSha256 = sha256(JSON.stringify(packageBody));
  const manifest = { ...packageBody, packageSha256 };

  const css = `
body{font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:96rem;margin:auto;padding:1rem;color:#1b1c1b;background:#fafbf8}
details{border:1px solid #8a9184;margin:1rem 0;padding:.85rem;border-radius:.4rem;background:#fff}
summary{font-weight:700;cursor:pointer}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,30rem),1fr));gap:1rem;margin-top:.75rem}
.deck-col{border:1px solid #d5d9d0;padding:.75rem;border-radius:.35rem;background:#fcfdfb}
.looks{display:grid;gap:.75rem}
.items{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem}
figure{margin:0;border:1px solid #eceee8;padding:.35rem;background:#fff}
img{width:100%;height:10rem;object-fit:contain;background:#f3f4f0}
figcaption{font-size:12px;margin-top:.25rem;overflow-wrap:anywhere}
.rubric{margin-top:.75rem;border:1px solid #c5cbbd;padding:.65rem;border-radius:.3rem}
label{display:block;margin:.35rem 0}
select,input,button{font:inherit;padding:.35rem .5rem}
button{background:#185b46;color:#fff;border:0;border-radius:.3rem;cursor:pointer;padding:.6rem 1.2rem;font-weight:600}
.pref{margin-top:.75rem;padding:.6rem;background:#f2f6f4;border-radius:.3rem;font-weight:600}
`;

  const js = `
const form=document.querySelector('form'),status=document.querySelector('#status');
const storeKey='aiw-blind-human-'+form.dataset.packageSha;
function readFields(){return Object.fromEntries(new FormData(form));}
form.addEventListener('input',()=>{try{localStorage.setItem(storeKey,JSON.stringify(readFields()));}catch{}});
try{for(const [k,v] of Object.entries(JSON.parse(localStorage.getItem(storeKey)||'{}'))){
  const el=form.elements.namedItem(k);if(el)el.value=v;
}}catch{}
form.addEventListener('submit',(e)=>{
  e.preventDefault();
  const vals=readFields();
  for(const el of form.elements){
    if(el.required && (!el.value || !el.checkValidity())){
      const d=el.closest('details');if(d)d.open=true;
      el.focus();el.reportValidity();status.textContent='Complete all required fields before exporting.';return;
    }
    if(el.name.endsWith('.note') && vals[el.name.replace(/\\.note$/,'.critical')]==='Yes' && !el.value.trim()){
      const d=el.closest('details');if(d)d.open=true;
      el.focus();status.textContent='Explain every Yes critical issue.';return;
    }
  }
  const reviewer=vals.reviewer.trim();
  const blob=new Blob([JSON.stringify({
    schema:'aiw.stylist6.human_blind_submission.v1',
    packageSha256:form.dataset.packageSha,
    replaySha256:form.dataset.replaySha,
    submittedAt:new Date().toISOString(),
    reviewer,
    values:vals
  },null,2)],{type:'application/json'});
  const u=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=u;a.download='human-review-'+reviewer+'.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(u),1000);
  status.textContent='Exported review JSON for '+reviewer+'. Return it privately to the release organizer.';
});
`;

  const html = `<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIWardrobe Independent Blind Fashion Evaluation (${candidates.length} comparisons)</title>
<style>${css}</style><main>
<h1>Independent Blind Outfit Comparison (${candidates.length} scenarios)</h1>
<p>Compare Deck A and Deck B independently for each scenario. Labels A and B are randomly assigned per scenario and do not identify any engine. Rate each metric from 1 (poor) to 5 (excellent), flag any critical defect, and select which 3-look deck you would prefer to receive.</p>
<p>Package SHA-256: <code>${packageSha256}</code> · Replay SHA-256: <code>${replaySha256}</code></p>
<form novalidate data-package-sha="${packageSha256}" data-replay-sha="${replaySha256}">
<label>Reviewer code (2–32 alphanumeric/underscore chars) <input name="reviewer" pattern="[A-Za-z0-9_]{2,32}" maxlength="32" required></label>
${scenarioHtmlBlocks.join('\n')}
<button type="submit">Export blind evaluation JSON</button>
<p id="status" role="status"></p>
</form></main><script>${js}</script></html>\n`;

  writeFileSync(path.join(outDir, 'index.html'), html, { mode: 0o600 });
  writeFileSync(path.join(outDir, 'package_manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { mode: 0o600 });
  const keyDoc = {
    schema: 'aiw.stylist6.human_blind_key.v1',
    generatedAt: manifest.generatedAt,
    gitSha,
    packageSha256,
    replaySha256,
    arms: { baseline: BASELINE_ARM, candidate: CANDIDATE_ARM },
    scenarios: Object.fromEntries(manifestScenarios.map((s) => [s.id, {
      auditSha256: s.auditSha256,
      heldOut: s.heldOut,
      ...keyCases[s.id],
    }])),
  };
  writeFileSync(path.join(outDir, 'blind_key.json'), JSON.stringify(keyDoc, null, 2) + '\n', { mode: 0o600 });

  return {
    outDir,
    packageSha256,
    replaySha256,
    scenarioCount: candidates.length,
    heldOutCount: manifest.heldOutCount,
    uniquePhotos: mediaBySha.size,
  };
}

export function scoreHumanBlindSubmissions({ keyDoc, submissions }) {
  if (!keyDoc || keyDoc.schema !== 'aiw.stylist6.human_blind_key.v1') {
    throw new Error('invalid_blind_key');
  }
  if (!Array.isArray(submissions) || submissions.length < 3) {
    throw new Error('require_at_least_three_independent_reviewers');
  }
  const reviewers = new Set();
  for (const sub of submissions) {
    if (!sub || sub.packageSha256 !== keyDoc.packageSha256) {
      throw new Error('submission_package_sha_mismatch');
    }
    if (typeof sub.reviewer !== 'string' || !/^[A-Za-z0-9_]{2,32}$/.test(sub.reviewer)) {
      throw new Error('invalid_reviewer_id');
    }
    if (reviewers.has(sub.reviewer)) {
      throw new Error(`duplicate_reviewer:${sub.reviewer}`);
    }
    reviewers.add(sub.reviewer);
  }

  const gateScenarios = [];
  let s6Wins = 0, s5Wins = 0, ties = 0;
  for (const [scenarioId, info] of Object.entries(keyDoc.scenarios)) {
    const votes = [];
    let s6 = 0, s5 = 0;
    for (const sub of submissions) {
      const pref = sub.values?.[`${scenarioId}.preference`];
      if (!['A', 'B', 'Tie'].includes(pref)) {
        throw new Error(`missing_preference:${sub.reviewer}:${scenarioId}`);
      }
      let choice = 'tie';
      if (pref === 'A' || pref === 'B') {
        const arm = info[pref];
        choice = arm === CANDIDATE_ARM ? 's6' : 's5';
      }
      if (choice === 's6') s6 += 1;
      else if (choice === 's5') s5 += 1;
      votes.push({ reviewer: sub.reviewer, choice });
    }
    if (s6 > votes.length / 2) s6Wins += 1;
    else if (s5 > votes.length / 2) s5Wins += 1;
    else ties += 1;
    gateScenarios.push({ id: scenarioId, auditSha256: info.auditSha256, votes });
  }
  const total = gateScenarios.length;
  return {
    total,
    s6Wins,
    s5Wins,
    ties,
    winRate: total ? s6Wins / total : 0,
    meets70PercentThreshold: total >= 30 && s6Wins / total >= 0.70,
    humanBlind: {
      independentOfStylistModel: true,
      scenarios: gateScenarios,
    },
  };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const args = process.argv.slice(2);
  const cmd = args[0];
  if (cmd === 'build' && args.length >= 3) {
    const replayPath = args[1];
    const outDir = args[2];
    const flag = (k) => {
      const idx = args.indexOf(`--${k}`);
      return idx >= 0 ? args[idx + 1] : undefined;
    };
    const res = await buildHumanBlindPackage({
      replayPath,
      outDir,
      minScenarios: Number(flag('min') || 30),
      limit: Number(flag('limit') || 0),
      gitSha: flag('git-sha') || null,
    });
    console.log(JSON.stringify(res, null, 2));
  } else if (cmd === 'score' && args.length >= 5) {
    const keyDoc = JSON.parse(readFileSync(args[1], 'utf8'));
    const submissions = args.slice(2).map((p) => JSON.parse(readFileSync(p, 'utf8')));
    console.log(JSON.stringify(scoreHumanBlindSubmissions({ keyDoc, submissions }), null, 2));
  } else {
    console.error('Usage:\n  build <replay.json> <outDir> [--min 30] [--limit N]\n  score <blind_key.json> <sub1.json> <sub2.json> <sub3.json>');
    process.exitCode = 2;
  }
}
