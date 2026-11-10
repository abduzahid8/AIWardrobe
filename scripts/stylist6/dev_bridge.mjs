// Dev-only localhost bridge: runs the REAL Stylist 6 fashion ranker for the
// iOS simulator app. Never deployed, never imported by the app bundle.
//
// Start:  node scripts/stylist6/dev_bridge.mjs [--port 3127]
// Health: GET  http://127.0.0.1:3127/health
// Rank:   POST http://127.0.0.1:3127/generate  { request, catalogRows }
//
// - `request` carries { style, occasion, brief, weather, budget?, preferences?,
//   recentSignatures?, excludeItemIds? } — the same fields the app already
//   collects for the generate-outfits edge function.
// - `catalogRows` are live shop_catalog-shaped rows (id, name, brand, price,
//   currency, image_url, source_url, description, is_active, ...). The bridge
//   adapts them with the real catalog_adapter and ranks with the real
//   recommendFashionDeck. No NVIDIA, no network: visual judging stays
//   NOT_RUN by design (no visualCritic is injected).
// - Failures are reported as success:false with a Stylist 6 code. The app
//   must surface them, never silently substitute a Stylist 5 deck.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { adaptCatalog } from '../../backend/stylist6/catalog_adapter.mjs';
import { recommendFashionDeck } from '../../backend/stylist6/fashion_deck.mjs';
import { referenceFirstBriefs } from '../../backend/stylist6/lookbook_briefs.mjs';
import { resolveStylistWeather } from '../../src/domain/outfitComposition.ts';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');

export const STYLIST6_DEV_ENGINE_VERSION = 'stylist-6-dev-local-20261111';
const DEFAULT_PORT = 3127;
const MAX_BODY_BYTES = 12 * 1024 * 1024;

const indexPath = path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json');
const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));

const seasonOf = (temp) => (temp >= 25 ? 'summer' : temp < 12 ? 'winter' : 'autumn');

// Same metadata-reference match the offline replay uses: the reference is an
// input common to both engines, only when the shortlist contains the
// requested style and occasion.
function pickReference(refs, { style, occasion }) {
  const norm = (v) => String(v || '').toLowerCase().replaceAll(' ', '_');
  return refs.find(({ referenceId }) => {
    const ref = index.images.find((entry) => entry.id === referenceId);
    return ref.styles.includes(style) && ref.occasions.includes(norm(occasion));
  });
}

function mapDeckItem(item, rankScore) {
  return {
    id: String(item.id),
    type: item.type || item.macroCategory,
    macroCategory: item.macroCategory,
    color: item.color || 'unknown',
    name: item.name,
    brand: item.brand || '',
    description: item.description || item.name,
    imageUrl: item.imageUrl,
    image: item.imageUrl,
    recommendation: `Stylist 6 rank · score ${Number(rankScore).toFixed(2)}`,
    isShopItem: true,
    price: item.price,
    currency: item.currency,
    shopUrl: item.shopUrl,
  };
}

async function handleGenerate(payload) {
  const { request = {}, catalogRows = [] } = payload || {};
  const { style, occasion } = request;
  if (typeof style !== 'string' || !style || typeof occasion !== 'string' || !occasion) {
    return { status: 400, body: { success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' } };
  }
  if (!Array.isArray(catalogRows) || !catalogRows.length) {
    return { status: 400, body: { success: false, code: 'stylist6_empty_catalog', stage: 'catalog_adaptation' } };
  }
  // Live discovery rows come from the catalog_discovery_candidates RPC, which
  // filters is_active=true server-side and omits the key. The shared adapter
  // treats a missing key as inactive, so default it here — explicit false
  // still filters out. No production code is touched by this normalization.
  const liveRows = (catalogRows || []).map((row) =>
    row && typeof row === 'object' && !('is_active' in row) ? { ...row, is_active: true } : row);
  const { items } = adaptCatalog(liveRows);
  if (!items.length) {
    return { status: 502, body: { success: false, code: 'stylist6_empty_catalog', stage: 'catalog_adaptation' } };
  }
  const budget = request.budget?.amount && request.budget?.currency ? request.budget : undefined;
  if (budget && !/^[A-Z]{3}$/.test(String(budget.currency))) {
    return { status: 400, body: { success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' } };
  }
  // Same-currency pool, mirroring the offline replay: cross-currency looks
  // cannot be priced honestly without an FX snapshot the ranker owns.
  const pool = budget ? items.filter((item) => item.currency === budget.currency) : items;
  if (!pool.length) {
    return { status: 502, body: { success: false, code: 'stylist6_no_feasible_looks', stage: 'stylist6_ranking' } };
  }
  const weather = resolveStylistWeather(request.brief || '', request.weather || {});
  const temp = Number(weather?.temp);
  const refs = referenceFirstBriefs(index, {
    style, occasion, season: Number.isFinite(temp) ? seasonOf(temp) : undefined,
  });
  const reference = pickReference(refs, { style, occasion });
  const input = {
    brief: request.brief || '',
    weather,
    referenceId: reference?.referenceId,
    recentSignatures: Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : [],
    sourceMode: 'catalog',
    budget,
    style,
    occasion,
    optionalLayer: true,
    preferences: request.preferences && typeof request.preferences === 'object' ? request.preferences : {},
  };
  const deck = await recommendFashionDeck({
    request: input,
    catalog: pool,
    referenceBriefs: refs,
    hardGate: () => ({ ok: true }),
    priceToBudgetCurrency: (item, currency) => (item.currency === currency ? item.price : NaN),
  });
  if ((deck.status !== 'complete' && deck.status !== 'partial') || !deck.outfits.length) {
    return {
      status: 502,
      body: {
        success: false, code: 'stylist6_no_feasible_looks', stage: 'stylist6_ranking',
        engineVersion: STYLIST6_DEV_ENGINE_VERSION, diagnostics: deck.diagnostics || null,
      },
    };
  }
  const excluded = new Set(Array.isArray(request.excludeItemIds) ? request.excludeItemIds.map(String) : []);
  const previousDeckOverlapBySlot = {};
  for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) previousDeckOverlapBySlot[slot] = 0;
  for (const look of deck.outfits) {
    for (const item of look.items) {
      if (excluded.has(String(item.id)) && previousDeckOverlapBySlot[item.macroCategory] !== undefined) {
        previousDeckOverlapBySlot[item.macroCategory] += 1;
      }
    }
  }
  return {
    status: 200,
    body: {
      success: true,
      generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      source: 'stylist-6-dev-local',
      compositionProtocol: 'optional-layer-v1',
      deckTarget: 3,
      status: deck.status,
      previousDeckOverlapBySlot,
      diagnostics: deck.diagnostics || null,
      outfits: deck.outfits.map((look) => ({
        id: look.id,
        items: look.items.map((item) => mapDeckItem(item, look.fashion?.score)),
        description: look.description,
        concept: { id: look.concept?.id, title: look.concept?.title, story: '', score: look.fashion?.score },
        stylingTips: [],
        confidence: look.fashion?.score,
        matchScore: look.fashion?.score,
        engineVersion: STYLIST6_DEV_ENGINE_VERSION,
        fashionDiagnostics: {
          score: look.fashion?.score,
          referenceMetadata: look.fashion?.referenceMetadata || null,
          visualVerdict: deck.diagnostics?.visualVerdict || 'NOT_RUN',
        },
      })),
    },
  };
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(Object.assign(new Error('request_too_large'), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

export function createDevBridgeServer() {
  return http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    try {
      if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200);
        res.end(JSON.stringify({ ok: true, engine: STYLIST6_DEV_ENGINE_VERSION, references: index.images.length }));
        return;
      }
      if (req.method === 'POST' && req.url === '/generate') {
        const raw = await readBody(req);
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch {
          res.writeHead(400);
          res.end(JSON.stringify({ success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' }));
          return;
        }
        const { status, body } = await handleGenerate(payload);
        process.stdout.write(`[stylist6-dev-bridge] POST /generate -> ${status} success=${body?.success} code=${body?.code || '-'} outfits=${body?.outfits?.length ?? 0}\n`);
        res.writeHead(status);
        res.end(JSON.stringify(body));
        return;
      }
      res.writeHead(404);
      res.end(JSON.stringify({ success: false, code: 'not_found', stage: 'stylist6_dev_bridge' }));
    } catch (error) {
      const status = error?.status === 413 ? 413 : 500;
      res.writeHead(status);
      res.end(JSON.stringify({
        success: false,
        code: status === 413 ? 'invalid_generation_request' : 'stylist6_ranking_failed',
        stage: 'stylist6_ranking',
      }));
    }
  });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const argvPort = Number(process.argv.slice(2).find((a) => a.startsWith('--port='))?.slice('--port='.length));
  const port = Number.isFinite(argvPort) && argvPort > 0 ? argvPort
    : Number(process.env.STYLIST6_DEV_PORT) > 0 ? Number(process.env.STYLIST6_DEV_PORT) : DEFAULT_PORT;
  const server = createDevBridgeServer();
  server.listen(port, '127.0.0.1', () => {
    process.stdout.write(`[stylist6-dev-bridge] ${STYLIST6_DEV_ENGINE_VERSION} on http://127.0.0.1:${port} (${index.images.length} references)\n`);
  });
}
