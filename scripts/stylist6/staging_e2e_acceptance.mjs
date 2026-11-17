#!/usr/bin/env node
/**
 * Isolated Stylist 6 staging E2E acceptance matrix & evidence capture harness.
 *
 * Covers Tasks 6 & 7:
 *   - catalog, wardrobe, mixed, locked anchor, UZS budget, EN/RU/UZ locales,
 *     AI Outfit Maker (`{ request }` contract), and Daily AI Outfit (top-level
 *     `generate-outfits` contract).
 *   - Supports zero-cost `--dry-run` (validates matrix normalization and
 *     receipt capture without network or quota) and `--execute-authorized`
 *     (gated by explicit `AIW_S6_E2E_ACK=I_ACCEPT_AUTHORIZED_S6_E2E_COSTS`
 *     against ONLY `aiw-stylist6-node-staging`, never production).
 */
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import { normalizeStylist6CloudPayload } from '../../backend/stylist6/cloud_handler.mjs';
import { referenceAmount, validateBudgetFxSnapshot } from '../../src/domain/budgetFx.ts';

export const EXPECTED_PROJECT_HOST = 'fyqpifmrsftsfqibhwhy.supabase.co';
export const STAGING_FUNCTION_PATH = '/functions/v1/aiw-stylist6-node-staging';
export const AUTHORIZED_E2E_ACK = 'I_ACCEPT_AUTHORIZED_S6_E2E_COSTS';

export function buildAcceptanceMatrix({
  runPrefix = `s6-e2e-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`,
  anchorItemId = 'ANCHOR_WARDROBE_OR_CATALOG_ID',
  budgetAmountUzs = 3_500_000,
} = {}) {
  return [
    {
      id: 'ai_maker_catalog_en',
      surface: 'ai_outfit_maker',
      locale: 'en',
      sourceMode: 'catalog',
      expectVisualRequested: true,
      payload: {
        request: {
          style: 'old_money',
          occasion: 'Work',
          brief: 'Work, old_money, tailored smart casual look',
          sourceMode: 'catalog',
          weather: { temp: 18, condition: 'clear' },
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [],
          enableVisual: true,
          language: 'en',
          generationRequestId: `${runPrefix}-catalog-en`,
        },
      },
    },
    {
      id: 'ai_maker_wardrobe_en',
      surface: 'ai_outfit_maker',
      locale: 'en',
      sourceMode: 'wardrobe',
      expectVisualRequested: false,
      payload: {
        request: {
          style: 'casual',
          occasion: 'Weekend',
          brief: 'Weekend, casual',
          sourceMode: 'wardrobe',
          weather: { temp: 20, condition: 'clear' },
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [],
          enableVisual: false,
          language: 'en',
          generationRequestId: `${runPrefix}-wardrobe-en`,
        },
      },
    },
    {
      id: 'ai_maker_mixed_en',
      surface: 'ai_outfit_maker',
      locale: 'en',
      sourceMode: 'mixed',
      expectVisualRequested: false,
      payload: {
        request: {
          style: 'smart_casual',
          occasion: 'Dinner',
          brief: 'Dinner, smart_casual',
          sourceMode: 'mixed',
          weather: { temp: 16, condition: 'clear' },
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [],
          enableVisual: false,
          language: 'en',
          generationRequestId: `${runPrefix}-mixed-en`,
        },
      },
    },
    {
      id: 'ai_maker_locked_anchor_mixed',
      surface: 'ai_outfit_maker',
      locale: 'en',
      sourceMode: 'mixed',
      requiredAnchorId: anchorItemId,
      expectVisualRequested: false,
      payload: {
        request: {
          style: 'smart_casual',
          occasion: 'Work',
          brief: 'Work, smart_casual, build around locked piece',
          sourceMode: 'mixed',
          weather: { temp: 18, condition: 'clear' },
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [anchorItemId],
          enableVisual: false,
          language: 'en',
          generationRequestId: `${runPrefix}-anchor-mixed`,
        },
      },
    },
    {
      id: 'ai_maker_budget_uzs',
      surface: 'ai_outfit_maker',
      locale: 'en',
      sourceMode: 'catalog',
      budget: { amount: budgetAmountUzs, currency: 'UZS' },
      expectVisualRequested: false,
      payload: {
        request: {
          style: 'classic',
          occasion: 'Work',
          brief: 'Work, classic under budget',
          sourceMode: 'catalog',
          budget: { amount: budgetAmountUzs, currency: 'UZS' },
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [],
          enableVisual: false,
          language: 'en',
          generationRequestId: `${runPrefix}-budget-uzs`,
        },
      },
    },
    {
      id: 'ai_maker_catalog_ru',
      surface: 'ai_outfit_maker',
      locale: 'ru',
      sourceMode: 'catalog',
      expectVisualRequested: false,
      payload: {
        request: {
          style: 'minimalist',
          occasion: 'Date night',
          brief: 'Date night, minimalist',
          sourceMode: 'catalog',
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [],
          enableVisual: false,
          language: 'ru',
          generationRequestId: `${runPrefix}-catalog-ru`,
        },
      },
    },
    {
      id: 'ai_maker_catalog_uz',
      surface: 'ai_outfit_maker',
      locale: 'uz',
      sourceMode: 'catalog',
      expectVisualRequested: false,
      payload: {
        request: {
          style: 'semi_classic',
          occasion: 'Work',
          brief: 'Work, semi_classic',
          sourceMode: 'catalog',
          preferences: {},
          recentSignatures: [],
          excludeItemIds: [],
          requiredIds: [],
          enableVisual: false,
          language: 'uz',
          generationRequestId: `${runPrefix}-catalog-uz`,
        },
      },
    },
    {
      id: 'daily_ai_outfit_top_level',
      surface: 'daily_ai_outfit',
      locale: 'en',
      sourceMode: 'mixed',
      expectVisualRequested: true,
      payload: {
        stylePreferences: 'business_casual',
        occasion: 'Work',
        weather: { temp: 18, condition: 'clear' },
        limit: 3,
        allowRecentFallback: true,
        catalogOnly: false,
        preferences: {},
        enableVisual: true,
        language: 'en',
        generationRequestId: `${runPrefix}-daily-en`,
      },
    },
  ];
}

const stripShopPrefix = (id) => (String(id).startsWith('shop_') ? String(id).slice(5) : String(id));

function checkLanguageCopy(locale, outfits) {
  const texts = outfits.map((o) => String(o?.description || '').trim()).filter(Boolean);
  if (!texts.length) return { ok: false, reason: 'missing_descriptions' };
  if (locale === 'ru') {
    const hasCyrillic = texts.some((t) => /[\u0400-\u04FF]/.test(t));
    return { ok: hasCyrillic, reason: hasCyrillic ? 'ok' : 'missing_cyrillic_in_ru_copy' };
  }
  const hasLatin = texts.some((t) => /[A-Za-z]/.test(t));
  return { ok: hasLatin, reason: hasLatin ? 'ok' : `missing_latin_in_${locale}_copy` };
}

export function captureAcceptanceRecord(scenario, status, body, { latencyMs = null, serverTiming = null, quotaBefore = null, quotaAfter = null } = {}) {
  const failures = [];
  const engineVersion = String(body?.generationVersion || '');
  if (status !== 200 || body?.success !== true) {
    failures.push(`http_${status}_${body?.code || body?.error || 'failed'}`);
  }
  if (!engineVersion.startsWith('stylist-6-')) {
    failures.push(`unexpected_engine_${engineVersion || 'none'}`);
  }
  const outfits = Array.isArray(body?.outfits) ? body.outfits : [];
  if (status === 200 && outfits.length !== 3) {
    failures.push(`expected_3_outfits_got_${outfits.length}`);
  }

  const outfitSignatures = [];
  const capturedOutfits = outfits.map((outfit, idx) => {
    const items = Array.isArray(outfit?.items) ? outfit.items : [];
    const ids = items.map((item) => String(item?.id || ''));
    const sig = [...ids].sort().join('|');
    outfitSignatures.push(sig);
    if (new Set(ids).size !== ids.length) {
      failures.push(`outfit_${idx + 1}_duplicate_item`);
    }
    for (const item of items) {
      if (scenario.sourceMode === 'catalog' && item.isShopItem !== true) {
        failures.push(`outfit_${idx + 1}_wardrobe_leak_in_catalog`);
      }
      if (scenario.sourceMode === 'wardrobe' && item.isShopItem !== false) {
        failures.push(`outfit_${idx + 1}_shop_leak_in_wardrobe`);
      }
    }
    if (scenario.requiredAnchorId) {
      const target = stripShopPrefix(scenario.requiredAnchorId);
      if (!ids.some((id) => stripShopPrefix(id) === target)) {
        failures.push(`outfit_${idx + 1}_missing_anchor_${scenario.requiredAnchorId}`);
      }
    }
    if (scenario.budget) {
      let fxSnapshot = null;
      if (body?.currencyConversion?.active && body.currencyConversion.snapshot) {
        try { fxSnapshot = validateBudgetFxSnapshot(body.currencyConversion.snapshot); } catch { fxSnapshot = null; }
      }
      let total = 0;
      for (const item of items) {
        if (item.isShopItem !== true) continue;
        const price = Number(item.price);
        const cur = String(item.currency || '');
        if (!Number.isFinite(price) || price <= 0) {
          failures.push(`outfit_${idx + 1}_invalid_shop_price`);
          continue;
        }
        if (cur === scenario.budget.currency) {
          total += price;
        } else if (fxSnapshot) {
          const converted = referenceAmount(price, cur, scenario.budget.currency, fxSnapshot);
          if (!Number.isFinite(converted)) failures.push(`outfit_${idx + 1}_unconverted_currency_${cur}`);
          else total += converted;
        } else {
          failures.push(`outfit_${idx + 1}_foreign_currency_without_fx_${cur}`);
        }
      }
      if (total > scenario.budget.amount + 1e-6) {
        failures.push(`outfit_${idx + 1}_budget_exceeded_${Math.round(total)}`);
      }
    }
    return {
      id: outfit?.id || `outfit_${idx + 1}`,
      selectionSource: outfit?.selectionSource ?? null,
      engineVersion: outfit?.engineVersion ?? null,
      concept: outfit?.concept ?? null,
      description: outfit?.description ?? null,
      visualVerdict: outfit?.fashionDiagnostics?.visualVerdict ?? null,
      referenceEvaluationMode: outfit?.fashionDiagnostics?.referenceEvaluationMode ?? null,
      items: items.map((item) => ({
        id: item.id,
        type: item.type,
        macroCategory: item.macroCategory ?? null,
        isShopItem: item.isShopItem,
        price: item.price ?? null,
        currency: item.currency ?? null,
        priceFreshness: item.priceFreshness ?? null,
        imageUrl: item.imageUrl ?? null,
        shopUrl: item.shopUrl ?? null,
      })),
    };
  });

  if (outfits.length === 3 && new Set(outfitSignatures).size < 3) {
    failures.push('non_distinct_three_outfit_composition');
  }
  const localeCheck = status === 200 ? checkLanguageCopy(scenario.locale, outfits) : { ok: false, reason: 'skipped_non_200' };
  if (status === 200 && !localeCheck.ok) {
    failures.push(localeCheck.reason);
  }

  return {
    id: scenario.id,
    surface: scenario.surface,
    locale: scenario.locale,
    sourceMode: scenario.sourceMode,
    status,
    pass: failures.length === 0,
    failures,
    engineVersion: engineVersion || null,
    serverInventory: body?.serverInventory ?? null,
    languageReturned: body?.language ?? null,
    latencyMs,
    serverTiming,
    quotaOutcome: quotaBefore !== null && quotaAfter !== null
      ? { before: quotaBefore, after: quotaAfter, delta: quotaAfter - quotaBefore }
      : null,
    visualStatus: body?.visualStatus ?? null,
    visualReceipt: body?.visualReceipt ?? null,
    merchantVerification: body?.offerVerification ?? null,
    currencyConversion: body?.currencyConversion ?? null,
    error: status === 200 ? null : {
      code: body?.code ?? body?.error ?? null,
      stage: body?.stage ?? null,
      diagnostics: body?.diagnostics ?? null,
    },
    outfits: capturedOutfits,
  };
}

export function validateDryMatrix(options = {}) {
  const matrix = buildAcceptanceMatrix(options);
  return matrix.map((scenario) => {
    const normalized = normalizeStylist6CloudPayload(scenario.payload, { STYLIST6_DEFAULT_VISUAL_EVAL: 'false' });
    assert.ok(normalized.style, `${scenario.id} missing normalized style`);
    assert.ok(normalized.occasion, `${scenario.id} missing normalized occasion`);
    assert.equal(normalized.language, scenario.locale, `${scenario.id} language mismatch`);
    if (scenario.budget) {
      assert.deepEqual(normalized.budget, scenario.budget, `${scenario.id} budget mismatch`);
    }
    if (scenario.requiredAnchorId) {
      assert.deepEqual(normalized.requiredIds, [scenario.requiredAnchorId], `${scenario.id} anchor mismatch`);
    }
    return {
      id: scenario.id,
      surface: scenario.surface,
      normalized,
    };
  });
}

async function runAuthorizedLive(options) {
  if (process.env.AIW_S6_E2E_ACK !== AUTHORIZED_E2E_ACK) {
    throw new Error(`Refusing live metered E2E: set AIW_S6_E2E_ACK=${AUTHORIZED_E2E_ACK}`);
  }
  const envFile = existsSync('.env') ? dotenv.parse(readFileSync('.env', 'utf8')) : {};
  const supabaseUrl = process.env.SUPABASE_URL || envFile.EXPO_PUBLIC_SUPABASE_URL || '';
  const anonKey = process.env.SUPABASE_ANON_KEY || envFile.EXPO_PUBLIC_SUPABASE_ANON_KEY || '';
  const serviceKey = process.env.AIW_TEST_SERVICE_ROLE_KEY || '';
  assert.ok(supabaseUrl && anonKey && serviceKey, 'Missing SUPABASE_URL, SUPABASE_ANON_KEY, or AIW_TEST_SERVICE_ROLE_KEY');
  const origin = new URL(supabaseUrl);
  assert.equal(origin.protocol, 'https:');
  assert.equal(origin.hostname, EXPECTED_PROJECT_HOST, 'Refusing unexpected Supabase host');
  const endpoint = new URL(STAGING_FUNCTION_PATH, origin).toString();

  const authClient = createClient(origin.origin, anonKey, { auth: { persistSession: false, autoRefreshToken: false } });
  const admin = createClient(origin.origin, serviceKey, { auth: { persistSession: false, autoRefreshToken: false } });

  let token = process.env.AIW_TEST_ACCESS_TOKEN || '';
  if (!token) {
    const email = process.env.AIW_TEST_EMAIL;
    const password = process.env.AIW_TEST_PASSWORD;
    assert.ok(email && password, 'Supply AIW_TEST_ACCESS_TOKEN or AIW_TEST_EMAIL/PASSWORD');
    const { data, error } = await authClient.auth.signInWithPassword({ email, password });
    assert.ok(!error && data?.session?.access_token, 'Test account sign-in failed');
    token = data.session.access_token;
  }
  const identity = await authClient.auth.getUser(token);
  assert.ok(!identity.error && identity.data?.user?.id, 'Failed to validate test JWT');
  const userId = identity.data.user.id;
  if (process.env.AIW_TEST_USER_ID) assert.equal(userId, process.env.AIW_TEST_USER_ID);

  let anchorItemId = options.anchorItemId || process.env.AIW_TEST_ANCHOR_ID || '';
  if (!anchorItemId) {
    const { data: wardrobe } = await admin.from('clothing_items')
      .select('id').eq('user_id', userId).limit(1);
    if (wardrobe?.[0]?.id) {
      anchorItemId = String(wardrobe[0].id);
    } else {
      const { data: shop } = await admin.from('shop_catalog')
        .select('id').eq('is_active', true).eq('category', 'tops').limit(1);
      anchorItemId = shop?.[0]?.id ? `shop_${shop[0].id}` : '';
    }
  }
  assert.ok(anchorItemId, 'No anchor item found for locked-anchor acceptance case');

  const day = () => new Date().toISOString().slice(0, 10);
  async function readUsed(date) {
    const { data, error } = await admin.from('usage_counters')
      .select('used').eq('user_id', userId).eq('feature', 'edge:aiOutfits')
      .eq('day', date).maybeSingle();
    assert.equal(error, null, 'Failed to read usage_counters');
    return data?.used ?? 0;
  }

  const matrix = buildAcceptanceMatrix({ anchorItemId });
  const records = [];
  for (const scenario of matrix) {
    const d = day();
    const before = await readUsed(d);
    const t0 = performance.now();
    const res = await fetch(endpoint, {
      method: 'POST',
      redirect: 'error',
      headers: {
        apikey: anonKey,
        authorization: `Bearer ${token}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify(scenario.payload),
      signal: AbortSignal.timeout(65000),
    });
    const latencyMs = Math.round(performance.now() - t0);
    const body = await res.json().catch(() => ({}));
    const after = await readUsed(d);
    records.push(captureAcceptanceRecord(scenario, res.status, body, {
      latencyMs,
      serverTiming: res.headers.get('server-timing'),
      quotaBefore: before,
      quotaAfter: after,
    }));
  }

  const outPath = `.private/staging-e2e-acceptance-${new Date().toISOString().slice(0, 10)}.json`;
  const report = {
    schema: 'aiw.stylist6.staging_e2e.v1',
    executedAt: new Date().toISOString(),
    endpoint,
    totalScenarios: records.length,
    passed: records.filter((r) => r.pass).length,
    failed: records.filter((r) => !r.pass).length,
    records,
  };
  writeFileSync(outPath, JSON.stringify(report, null, 2), { mode: 0o600 });
  console.log(JSON.stringify({ outPath, totalScenarios: report.totalScenarios, passed: report.passed, failed: report.failed }));
  if (report.failed > 0) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args.includes('--dry-run')) {
    const checked = validateDryMatrix();
    console.log(JSON.stringify({ mode: 'dry-run', scenarios: checked.map((c) => c.id), count: checked.length }));
  } else if (args.includes('--execute-authorized')) {
    const anchorFlag = args.find((a) => a.startsWith('--anchor='))?.slice('--anchor='.length);
    await runAuthorizedLive({ anchorItemId: anchorFlag });
  } else {
    console.error('Usage: node scripts/stylist6/staging_e2e_acceptance.mjs --dry-run | --execute-authorized');
    process.exitCode = 2;
  }
}
