#!/usr/bin/env node
/**
 * Stylist 6 — Scale & Concurrency Load Benchmark (10 / 50 / 100 concurrent).
 *
 * Benchmarks our production learning, quality-flywheel, catalog scoring, and
 * multi-tier distributed cache stack against the frozen live Supabase catalog
 * snapshot (8,554 active SKUs) with zero paid API calls.
 *
 * Usage:
 *   node scripts/stylist6/scale_load_benchmark.mjs
 *   node scripts/stylist6/scale_load_benchmark.mjs --json
 *   node scripts/stylist6/scale_load_benchmark.mjs --tiers 10,50,100
 *   node scripts/stylist6/scale_load_benchmark.mjs --snapshot /path/to/snapshot.json
 */

import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { adaptCatalog } from '../../backend/stylist6/catalog_adapter.mjs';
import {
  buildUnifiedTasteProfile,
  scoreItemTasteAffinity,
  scoreLookTasteAffinity,
  summarizeTasteForPrompt,
  TASTE_EMBEDDING_DIM,
} from '../../backend/stylist6/taste_learning.mjs';
import {
  createQualityFlywheel,
  buildQualityLookupMap,
} from '../../backend/stylist6/catalog_quality_flywheel.mjs';
import {
  createStylistCache,
  buildCreativeDirectorCacheKey,
  buildVisualVerdictCacheKey,
  buildAppearanceCacheKey,
} from '../../backend/stylist6/distributed_cache.mjs';

const DEFAULT_SNAPSHOT_PATH = '/Users/v.zahid/.gemini/antigravity/brain/9c9881b2-32f5-4d7f-a24c-e8b407755794/scratch/live_catalog_snapshot.json';

function parseArgs(argv) {
  const opts = {
    json: false,
    tiers: [10, 50, 100],
    snapshotPath: process.env.STYLIST6_CATALOG_SNAPSHOT || DEFAULT_SNAPSHOT_PATH,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--json') opts.json = true;
    else if (arg === '--snapshot' && argv[i + 1]) opts.snapshotPath = argv[++i];
    else if (arg === '--tiers' && argv[i + 1]) {
      const parsed = argv[++i].split(',').map((x) => parseInt(x.trim(), 10)).filter((n) => n > 0 && n <= 1000);
      if (parsed.length > 0) opts.tiers = parsed;
    }
  }
  return opts;
}

function buildSyntheticCatalog(count = 2000) {
  const slots = [
    { category: 'tops', garment_type: 'upper_body', subtypes: ['shirt', 'polo', 'fine-knit', 't-shirt'] },
    { category: 'tops', garment_type: 'outerwear', subtypes: ['blazer', 'coat', 'cardigan', 'overshirt'] },
    { category: 'bottoms', garment_type: 'lower_body', subtypes: ['tailored-trouser', 'chino', 'jeans'] },
    { category: 'shoes', garment_type: 'shoes', subtypes: ['loafer', 'derby', 'oxford', 'boot', 'sneaker'] },
  ];
  const colors = ['navy', 'camel', 'cream', 'charcoal', 'olive', 'brown', 'white', 'black', 'beige', 'burgundy'];
  const brands = ['Suitsupply', 'Velasca', 'Massimo Dutti', 'Loro Piana', 'Incotex', 'Barbour', 'Zegna', 'AllSaints'];
  const rows = [];
  for (let i = 0; i < count; i++) {
    const s = slots[i % slots.length];
    const sub = s.subtypes[i % s.subtypes.length];
    const color = colors[i % colors.length];
    const brand = brands[i % brands.length];
    rows.push({
      id: `synth-${i}`,
      brand,
      name: `${brand} ${color} ${sub} #${i}`,
      price: 80 + (i % 12) * 35,
      currency: 'USD',
      category: s.category,
      garment_type: s.garment_type,
      description: 'official product photo on plain white background, no model',
      image_url: `https://cdn.example.com/items/synth-${i}.jpg`,
      source_url: `https://shop.example.com/items/synth-${i}`,
      is_active: true,
    });
  }
  return rows;
}

function loadCatalog(snapshotPath) {
  const t0 = performance.now();
  let rawRows = null;
  let source = 'synthetic_fallback';
  if (snapshotPath && fs.existsSync(snapshotPath)) {
    const parsed = JSON.parse(fs.readFileSync(snapshotPath, 'utf8'));
    rawRows = Array.isArray(parsed) ? parsed : (parsed.items || parsed.rows || []);
    source = snapshotPath;
  } else {
    rawRows = buildSyntheticCatalog(2400);
  }
  const adapted = adaptCatalog(rawRows);
  const adaptMs = performance.now() - t0;

  const bySlot = { top: [], outerwear: [], bottom: [], shoes: [] };
  const catalogById = new Map();
  for (const item of adapted.items) {
    catalogById.set(item.id, item);
    if (bySlot[item.macroCategory]) bySlot[item.macroCategory].push(item);
  }
  return {
    source,
    rawCount: rawRows.length,
    admittedCount: adapted.items.length,
    skipped: adapted.skipped,
    adaptMs: +(adaptMs).toFixed(2),
    slotCounts: {
      top: bySlot.top.length,
      outerwear: bySlot.outerwear.length,
      bottom: bySlot.bottom.length,
      shoes: bySlot.shoes.length,
    },
    items: adapted.items,
    bySlot,
    catalogById,
  };
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return +sorted[idx].toFixed(2);
}

function makeDeterministicEmbedding(seed) {
  const vec = new Float32Array(TASTE_EMBEDDING_DIM);
  const a = seed % TASTE_EMBEDDING_DIM;
  const b = (seed * 7 + 13) % TASTE_EMBEDDING_DIM;
  const c = (seed * 31 + 97) % TASTE_EMBEDDING_DIM;
  vec[a] = 0.7;
  vec[b] = 0.5;
  vec[c] = 0.509902;
  return vec;
}

// Simulated Upstash Redis REST backend with realistic ~2ms LAN/regional RTT
function createSimulatedRedisFetch(simulatedRttMs = 2) {
  const kv = new Map();
  return {
    store: kv,
    fetchFn: async (_url, init) => {
      if (simulatedRttMs > 0) {
        await new Promise((r) => setTimeout(r, simulatedRttMs));
      }
      const [cmd, key, val] = JSON.parse(init.body);
      if (cmd === 'GET') {
        return { ok: true, async json() { return { result: kv.has(key) ? kv.get(key) : null }; } };
      }
      if (cmd === 'SET') {
        kv.set(key, val);
        return { ok: true, async json() { return { result: 'OK' }; } };
      }
      if (cmd === 'DEL') {
        kv.delete(key);
        return { ok: true, async json() { return { result: 1 }; } };
      }
      return { ok: true, async json() { return { result: null }; } };
    },
  };
}

const SCENARIOS = [
  { style: 'classic', occasion: 'work', tempC: 16, budget: 600, currency: 'USD', presentation: 'menswear', season: 'autumn' },
  { style: 'old_money', occasion: 'date', tempC: 19, budget: 900, currency: 'USD', presentation: 'menswear', season: 'autumn' },
  { style: 'minimalist', occasion: 'casual', tempC: 24, budget: 450, currency: 'USD', presentation: 'menswear', season: 'spring' },
  { style: 'smart_casual', occasion: 'work', tempC: 12, budget: 750, currency: 'USD', presentation: 'menswear', season: 'winter' },
  { style: 'relaxed_tailoring', occasion: 'travel', tempC: 18, budget: 550, currency: 'USD', presentation: 'unisex', season: 'spring' },
  { style: 'classic', occasion: 'formal', tempC: 14, budget: 1200, currency: 'USD', presentation: 'menswear', season: 'winter' },
];

async function runConcurrencyTier({ concurrency, catalog, qualityMap, embeddingsByImageSha, caches, nowMs }) {
  const memBefore = process.memoryUsage();
  const sumStats = () => caches.reduce((acc, c) => {
    const s = c.stats();
    acc.l1 += s.hits.l1;
    acc.l2 += s.hits.l2;
    acc.coalesced += s.coalescedHits;
    acc.misses += s.misses;
    acc.l1Entries += s.l1Entries;
    acc.l1Bytes += s.l1Bytes;
    acc.evictions += s.evictions;
    return acc;
  }, { l1: 0, l2: 0, coalesced: 0, misses: 0, l1Entries: 0, l1Bytes: 0, evictions: 0 });
  const statsBefore = sumStats();

  const latencies = [];
  const tStart = performance.now();

  const runSingleRequest = async (reqIndex) => {
    const reqT0 = performance.now();
    // Route across stateless Node replicas sharing L2 Upstash Redis
    const cache = caches[reqIndex % caches.length];
    // Zipf-like scenario skew: 65% hit top-2 popular scenarios (realistic prod distribution),
    // 35% spread across tail contexts.
    const scenario = reqIndex % 3 !== 0
      ? SCENARIOS[reqIndex % 2]
      : SCENARIOS[reqIndex % SCENARIOS.length];

    // 1. Build user's unified personal taste profile (explicit + implicit signals)
    const sampleTop = catalog.bySlot.top[(reqIndex * 3) % catalog.bySlot.top.length];
    const sampleOuter = catalog.bySlot.outerwear[(reqIndex * 5) % catalog.bySlot.outerwear.length];
    const sampleBottom = catalog.bySlot.bottom[(reqIndex * 7) % catalog.bySlot.bottom.length];
    const sampleBadShoe = catalog.bySlot.shoes[(reqIndex * 11) % catalog.bySlot.shoes.length];
    const sampleGoodShoe = catalog.bySlot.shoes[(reqIndex * 11 + 1) % catalog.bySlot.shoes.length];

    const tasteProfile = buildUnifiedTasteProfile({
      explicitFeedback: [
        {
          rating: -1,
          item_ids: [sampleTop.id, sampleOuter.id, sampleBottom.id, sampleBadShoe.id],
          target_slot: 'shoes',
          feedback_tags: ['wrong_shoes'],
          created_at: nowMs - 2 * 86_400_000,
        },
        {
          rating: 1,
          item_ids: [sampleTop.id, sampleOuter.id, sampleBottom.id, sampleGoodShoe.id],
          feedback_tags: ['love_palette'],
          created_at: nowMs - 4 * 86_400_000,
        },
      ],
      savedOutfits: [
        { items: [sampleOuter.id, sampleGoodShoe.id], created_at: nowMs - 5 * 86_400_000 },
      ],
      wearLogs: [
        { item_ids: [sampleTop.id, sampleBottom.id], date: new Date(nowMs - 6 * 86_400_000).toISOString() },
      ],
      brandClicks: [
        { item_id: sampleGoodShoe.id, brand: sampleGoodShoe.brand || 'Velasca', price: sampleGoodShoe.price, clicked_at: nowMs - 86_400_000 },
      ],
      catalogById: catalog.catalogById,
      embeddingsByImageSha,
      nowMs,
    });
    const tasteSummary = summarizeTasteForPrompt(tasteProfile);

    // 2. Cache lookup / singleflight compute for Creative Director plan
    const cdKey = buildCreativeDirectorCacheKey({
      version: 'cd-v2',
      modelId: 'nemotron-120b',
      presentation: scenario.presentation,
      style: scenario.style,
      occasion: scenario.occasion,
      temperatureC: scenario.tempC,
      budget: scenario.budget,
      currency: scenario.currency,
      season: scenario.season,
      tasteSignature: tasteSummary.formalityShift,
    });

    await cache.getOrSet(cdKey, async () => {
      // Simulated non-blocking async provider I/O (6ms synthetic wait for local load harness)
      await new Promise((r) => setTimeout(r, 6));
      return {
        directions: ['signature_mastery', 'silhouette_texture_shift', 'editorial_statement'],
        generatedAt: nowMs,
      };
    }, { ttlMs: 30 * 60_000 });

    // 3. Appearance cache lookup (content-addressed photo SHA, simulating ~90% warm cache)
    const userBucket = reqIndex % Math.max(4, Math.floor(concurrency * 0.25));
    const photoSha = createHash('sha256').update(`optin-photo-${userBucket}`).digest('hex');
    const apKey = buildAppearanceCacheKey({ photoSha256: photoSha, analyzerVersion: 'v1', modelId: 'gemini-2.5-flash' });
    await cache.getOrSet(apKey, async () => {
      await new Promise((r) => setTimeout(r, 4));
      return { colorSeason: 'deep_autumn', undertone: 'warm', contrast: 'medium' };
    }, { ttlMs: 7 * 86_400_000 });

    // 4. Score bounded per-slot candidates (60 per slot = 240 SKUs) with quality flywheel + taste learning
    const selectedBySlot = {};
    for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) {
      const pool = catalog.bySlot[slot];
      const stride = Math.max(1, Math.floor(pool.length / 60));
      let bestItem = pool[0];
      let bestScore = -Infinity;
      for (let i = 0; i < Math.min(60, pool.length); i++) {
        const candidate = pool[(i * stride + reqIndex) % pool.length];
        const q = qualityMap.get(candidate.id);
        if (q?.quarantined) continue;
        const qualityScore = q ? q.qualityScore : 0.65;
        const tasteScore = scoreItemTasteAffinity(candidate, slot, tasteProfile).score;
        const total = qualityScore + tasteScore;
        if (total > bestScore) {
          bestScore = total;
          bestItem = candidate;
        }
      }
      selectedBySlot[slot] = bestItem;
    }

    // 5. Score 3 composed candidate looks + Visual Critic verdict cache lookup
    const lookItems = [selectedBySlot.top, selectedBySlot.outerwear, selectedBySlot.bottom, selectedBySlot.shoes];
    scoreLookTasteAffinity(lookItems, tasteProfile);

    const vvKey = buildVisualVerdictCacheKey({
      criticVersion: 'vc-v2',
      modelId: 'kimi-k3.1-pro',
      items: lookItems,
      occasion: scenario.occasion,
      style: scenario.style,
      presentation: scenario.presentation,
      temperatureC: scenario.tempC,
    });
    await cache.getOrSet(vvKey, async () => {
      await new Promise((r) => setTimeout(r, 8));
      return { approved: true, editorialScore: 8.4, weakestSlot: null };
    }, { ttlMs: 60 * 60_000 });

    latencies.push(performance.now() - reqT0);
  };

  await Promise.all(Array.from({ length: concurrency }, (_, i) => runSingleRequest(i)));

  const totalElapsedMs = performance.now() - tStart;
  const statsAfter = sumStats();
  const memAfter = process.memoryUsage();
  latencies.sort((a, b) => a - b);

  const l1HitsDelta = statsAfter.l1 - statsBefore.l1;
  const l2HitsDelta = statsAfter.l2 - statsBefore.l2;
  const coalescedDelta = statsAfter.coalesced - statsBefore.coalesced;
  const missesDelta = statsAfter.misses - statsBefore.misses;
  const effectiveHits = l1HitsDelta + l2HitsDelta + coalescedDelta;
  const totalCacheOps = effectiveHits + Math.max(0, missesDelta - coalescedDelta);

  return {
    concurrency,
    totalElapsedMs: +totalElapsedMs.toFixed(2),
    throughputReqPerSec: +((concurrency / totalElapsedMs) * 1000).toFixed(1),
    latencyMs: {
      p50: percentile(latencies, 50),
      p95: percentile(latencies, 95),
      p99: percentile(latencies, 99),
      max: +latencies[latencies.length - 1].toFixed(2),
    },
    cache: {
      l1Hits: l1HitsDelta,
      l2Hits: l2HitsDelta,
      coalescedSingleflightHits: coalescedDelta,
      coldComputes: Math.max(0, missesDelta - coalescedDelta),
      effectiveHitRatePct: totalCacheOps > 0 ? +((effectiveHits / totalCacheOps) * 100).toFixed(1) : 0,
      l1Entries: statsAfter.l1Entries,
      l1BytesKB: +(statsAfter.l1Bytes / 1024).toFixed(1),
      l1Evictions: statsAfter.evictions,
    },
    memoryDeltaMB: {
      rss: +((memAfter.rss - memBefore.rss) / (1024 * 1024)).toFixed(2),
      heapUsed: +((memAfter.heapUsed - memBefore.heapUsed) / (1024 * 1024)).toFixed(2),
      rssAfterMB: +(memAfter.rss / (1024 * 1024)).toFixed(1),
    },
  };
}

function computeOneMillionMauSizing(tier100Result, catalogCount) {
  // Transparent, auditable 1M-MAU sizing & unit economics model.
  const mau = 1_000_000;
  const generationsPerMauPerMonth = 2.5;
  const monthlyGenerations = mau * generationsPerMauPerMonth; // 2,500,000 / mo
  const dailyGenerations = Math.round(monthlyGenerations / 30); // ~83,333 / day
  const avgRps = +(monthlyGenerations / (30 * 86_400)).toFixed(2); // ~0.96 RPS
  const diurnalPeakMultiplier = 6.0;
  const launchSpikeMultiplier = 15.0;
  const peakRps = +(avgRps * diurnalPeakMultiplier).toFixed(2); // ~5.79 RPS
  const spikeRps = +(avgRps * launchSpikeMultiplier).toFixed(2); // ~14.47 RPS

  // Per-instance capacity:
  // CPU-side taste/quality/retrieval overhead from tier100 benchmark is tiny (~5-20ms/req),
  // so concurrency per 2-vCPU / 2GB Node instance is I/O-wait bound by upstream VLM RTT (~14s avg).
  // With configurable concurrency = 24 in-flight requests per instance (removing the old inflight>=2 wall):
  const avgEndToEndSec = 14;
  const safeInflightPerInstance = 24;
  const sustainableRpsPerInstance = +(safeInflightPerInstance / avgEndToEndSec).toFixed(2); // ~1.71 RPS/instance
  const headroomFactor = 2.5;

  const steadyStateInstances = Math.max(3, Math.ceil((peakRps * 1.5) / sustainableRpsPerInstance));
  const peakAutoscaleInstances = Math.max(steadyStateInstances, Math.ceil((spikeRps * headroomFactor) / sustainableRpsPerInstance));

  // AI Inference cost model (blended NVIDIA NIM / OpenRouter Gemini 2.5 Flash / Kimi K3.1 Pro):
  // Uncached per generation:
  //   - Creative Director (120B/30B text): ~$0.0025
  //   - SKU Composer (120B): ~$0.0035
  //   - Visual Critic (4–6 looks x 1024px board + 1 surgical refine): ~$0.0090
  //   - Portfolio Jury (1800x1000 sheet): ~$0.0030
  //   - Appearance Analyzer (amortized 1x per user photo): ~$0.0015
  const uncachedCostPerGenUsd = 0.0025 + 0.0035 + 0.0090 + 0.0030 + 0.0015; // $0.0195 / gen

  // Cached hit rates (conservative production targets backed by our 3-tier cache + singleflight):
  const cdHitRate = 0.35;
  const vvHitRate = 0.40;
  const apHitRate = 0.95; // content-addressed by photoSha256
  const cachedCostPerGenUsd = +(
    0.0025 * (1 - cdHitRate)
    + 0.0035
    + 0.0090 * (1 - vvHitRate)
    + 0.0030
    + 0.0015 * (1 - apHitRate)
  ).toFixed(5); // ~$0.0136 / gen (~30.3% savings!)

  const monthlyAiCostUncachedUsd = Math.round(monthlyGenerations * uncachedCostPerGenUsd);
  const monthlyAiCostCachedUsd = Math.round(monthlyGenerations * cachedCostPerGenUsd);

  // Infrastructure monthly costs at 1M MAU:
  const nodeComputeMonthlyUsd = steadyStateInstances * 85 + Math.round((peakAutoscaleInstances - steadyStateInstances) * 85 * 0.2);
  const upstashRedisMonthlyUsd = 180; // Pro tier, 2–4GB RAM, ~10M cmds/day
  const supabasePostgresMonthlyUsd = 599; // Dedicated 4–8 vCPU / 16–32GB RAM (pgvector HNSW for 100k SKUs needs ~600MB RAM)
  const totalMonthlyInfraUsd = nodeComputeMonthlyUsd + upstashRedisMonthlyUsd + supabasePostgresMonthlyUsd;
  const totalMonthlyCachedUsd = monthlyAiCostCachedUsd + totalMonthlyInfraUsd;

  // pgvector HNSW RAM estimate for catalog scaling (768-d float32 = 3,072 bytes/vec + ~1.6x HNSW m=16 graph overhead)
  const hnswBytesPerSku = 768 * 4 * 1.6;
  const currentHnswRamMB = +((catalogCount * hnswBytesPerSku) / (1024 * 1024)).toFixed(1);
  const scaled100kHnswRamMB = +((100_000 * hnswBytesPerSku) / (1024 * 1024)).toFixed(1);

  return {
    assumptions: {
      mau,
      generationsPerMauPerMonth,
      monthlyGenerations,
      dailyGenerations,
      avgRps,
      peakRps,
      spikeRps,
      avgEndToEndSec,
      safeInflightPerInstance,
      measuredCpuP95At100ConcurrencyMs: tier100Result?.latencyMs?.p95 ?? null,
    },
    computeSizing: {
      sustainableRpsPerInstance,
      steadyStateInstances2vCpu2Gb: steadyStateInstances,
      peakAutoscaleInstances2vCpu2Gb: peakAutoscaleInstances,
      multiAzMinimum: 3,
      oldBrickWallConcurrencyLimit: 2,
      newConcurrencyPerInstance: safeInflightPerInstance,
      concurrencySpeedupFactor: `${safeInflightPerInstance / 2}x per instance`,
    },
    databaseAndCacheSizing: {
      currentCatalogSkus: catalogCount,
      pgvectorHnswRamCurrentMB: currentHnswRamMB,
      pgvectorHnswRamAt100kSkusMB: scaled100kHnswRamMB,
      redisL2MemoryRecommendation: '4 GB Upstash Redis (global replication + 30–60m TTL)',
      l1BoundedMemoryCapPerInstance: '32 MiB LRU (hard OOM guard)',
    },
    unitEconomicsUsd: {
      uncachedPer1kGenerations: +(uncachedCostPerGenUsd * 1000).toFixed(2),
      cachedPer1kGenerations: +(cachedCostPerGenUsd * 1000).toFixed(2),
      cacheSavingsPct: +(((uncachedCostPerGenUsd - cachedCostPerGenUsd) / uncachedCostPerGenUsd) * 100).toFixed(1),
      monthlyAiUncachedUsd: monthlyAiCostUncachedUsd,
      monthlyAiCachedUsd: monthlyAiCostCachedUsd,
      monthlyComputeUsd: nodeComputeMonthlyUsd,
      monthlyRedisUsd: upstashRedisMonthlyUsd,
      monthlyPostgresUsd: supabasePostgresMonthlyUsd,
      totalMonthlyCostAt1mMauUsd: totalMonthlyCachedUsd,
      totalCostPerMauPerMonthUsd: +(totalMonthlyCachedUsd / mau).toFixed(4),
    },
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const catalog = loadCatalog(opts.snapshotPath);

  // Seed realistic quality flywheel state (~5% of catalog reviewed, ~1.5% quarantined)
  const flywheelStore = new Map();
  const flywheel = createQualityFlywheel(flywheelStore);
  const sampleOutcomes = [];
  for (let i = 0; i < Math.min(400, catalog.items.length); i++) {
    const sku = catalog.items[i];
    if (i % 25 === 0) {
      // Chronic photo defect SKU -> quarantined
      sampleOutcomes.push(
        { approved: false, items: [{ id: sku.id, slot: sku.macroCategory }], weakestSlot: sku.macroCategory, defectCategory: 'broken_photo' },
        { approved: false, items: [{ id: sku.id, slot: sku.macroCategory }], weakestSlot: sku.macroCategory, defectCategory: 'broken_photo' },
        { approved: false, items: [{ id: sku.id, slot: sku.macroCategory }], weakestSlot: sku.macroCategory, defectCategory: 'broken_photo' },
      );
    } else {
      sampleOutcomes.push({
        approved: true,
        items: [{ id: sku.id, slot: sku.macroCategory }],
        editorialScore: 8.2,
      });
    }
  }
  await flywheel.recordOutcome(sampleOutcomes);
  const qualityMap = buildQualityLookupMap(flywheelStore);
  const flywheelStats = await flywheel.stats();

  // Seed sample 768-d embeddings for top 100 items to exercise centroid math realistically
  const embeddingsByImageSha = new Map();
  for (let i = 0; i < Math.min(100, catalog.items.length); i++) {
    embeddingsByImageSha.set(catalog.items[i].id, makeDeterministicEmbedding(i + 1));
  }

  // Create 2 stateless Node replicas (each with 16 MiB bounded L1 LRU) sharing
  // a simulated 2ms L2 Upstash Redis cluster so both L1 and L2 cross-node hits are benchmarked.
  const simRedis = createSimulatedRedisFetch(2);
  const caches = [0, 1].map(() => createStylistCache({
    redisUrl: 'https://sim-upstash.local',
    redisToken: 'sim-token',
    fetchFn: simRedis.fetchFn,
    maxMemoryEntries: 2000,
    maxMemoryBytes: 16 * 1024 * 1024,
    namespace: 'bench',
  }));

  const nowMs = Date.now();
  const tierResults = [];
  for (const concurrency of opts.tiers) {
    const res = await runConcurrencyTier({
      concurrency,
      catalog,
      qualityMap,
      embeddingsByImageSha,
      caches,
      nowMs,
    });
    tierResults.push(res);
  }

  const tier100 = tierResults.find((t) => t.concurrency === 100) || tierResults[tierResults.length - 1];
  const sizing = computeOneMillionMauSizing(tier100, catalog.admittedCount);

  const report = {
    benchmark: 'stylist6-scale-load-benchmark-v1',
    catalog: {
      source: catalog.source,
      rawRows: catalog.rawCount,
      admittedSkus: catalog.admittedCount,
      adaptMs: catalog.adaptMs,
      slotCounts: catalog.slotCounts,
      flywheel: flywheelStats,
    },
    tiers: tierResults,
    oneMillionMauSizing: sizing,
  };

  if (opts.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  console.log('=== Stylist 6 Scale & Concurrency Load Benchmark ===');
  console.log(`Catalog Source : ${report.catalog.source}`);
  console.log(`Catalog Volume : ${report.catalog.admittedSkus}/${report.catalog.rawRows} admitted SKUs (adapted in ${report.catalog.adaptMs}ms)`);
  console.log(`Slot Inventory : top=${catalog.slotCounts.top}, outerwear=${catalog.slotCounts.outerwear}, bottom=${catalog.slotCounts.bottom}, shoes=${catalog.slotCounts.shoes}`);
  console.log(`Quality Flywheel: tracked=${flywheelStats.trackedSkus}, quarantined=${flywheelStats.quarantinedSkus}, avgScore=${flywheelStats.avgQualityScore}\n`);

  console.log('Concurrency Tier Results:');
  console.log('Conc | Elapsed(ms) | Req/s   | P50(ms) | P95(ms) | P99(ms) | CacheHit% | L1/L2/Coal/Miss | RSS Delta(MB)');
  console.log('-----+-------------+---------+---------+---------+---------+-----------+-----------------+--------------');
  for (const t of tierResults) {
    const mix = `${t.cache.l1Hits}/${t.cache.l2Hits}/${t.cache.coalescedSingleflightHits}/${t.cache.coldComputes}`;
    console.log(
      `${String(t.concurrency).padStart(4)} | `
      + `${String(t.totalElapsedMs).padStart(11)} | `
      + `${String(t.throughputReqPerSec).padStart(7)} | `
      + `${String(t.latencyMs.p50).padStart(7)} | `
      + `${String(t.latencyMs.p95).padStart(7)} | `
      + `${String(t.latencyMs.p99).padStart(7)} | `
      + `${String(`${t.cache.effectiveHitRatePct}%`).padStart(9)} | `
      + `${mix.padStart(15)} | `
      + `${String(t.memoryDeltaMB.rss).padStart(12)}`,
    );
  }

  console.log('\n=== 1M-MAU Cost & Instance-Count Sizing ===');
  console.log(`Traffic Model   : ${sizing.assumptions.mau.toLocaleString()} MAU -> ${sizing.assumptions.monthlyGenerations.toLocaleString()} gens/mo (${sizing.assumptions.avgRps} avg RPS, ${sizing.assumptions.peakRps} diurnal peak RPS, ${sizing.assumptions.spikeRps} spike RPS)`);
  console.log(`Compute Sizing  : ${sizing.computeSizing.steadyStateInstances2vCpu2Gb} steady-state / ${sizing.computeSizing.peakAutoscaleInstances2vCpu2Gb} peak-autoscale Node instances (2 vCPU / 2 GB, 24 concurrent in-flight/node = ${sizing.computeSizing.concurrencySpeedupFactor} vs old wall)`);
  console.log(`pgvector HNSW   : ${sizing.databaseAndCacheSizing.pgvectorHnswRamCurrentMB} MB RAM (${catalog.admittedCount} SKUs) -> ${sizing.databaseAndCacheSizing.pgvectorHnswRamAt100kSkusMB} MB RAM at 100k SKUs`);
  console.log(`AI Unit Cost    : $${sizing.unitEconomicsUsd.uncachedPer1kGenerations}/1k uncached -> $${sizing.unitEconomicsUsd.cachedPer1kGenerations}/1k cached (${sizing.unitEconomicsUsd.cacheSavingsPct}% AI cost reduction)`);
  console.log(`Monthly Breakdown: AI=$${sizing.unitEconomicsUsd.monthlyAiCachedUsd.toLocaleString()} + Compute=$${sizing.unitEconomicsUsd.monthlyComputeUsd.toLocaleString()} + Redis=$${sizing.unitEconomicsUsd.monthlyRedisUsd} + Postgres=$${sizing.unitEconomicsUsd.monthlyPostgresUsd}`);
  console.log(`TOTAL @ 1M MAU  : $${sizing.unitEconomicsUsd.totalMonthlyCostAt1mMauUsd.toLocaleString()}/mo ($${sizing.unitEconomicsUsd.totalCostPerMauPerMonthUsd}/MAU/mo)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
