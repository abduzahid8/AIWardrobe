import { OutfitGenerationFailure } from './outfitGenerationFailure';
import { normalizeStylistLanguage } from '../domain/stylingContext';

/**
 * Development-only Stylist 6 path for the iOS simulator.
 *
 * Reachability is closed in release builds by construction:
 * - `STYLIST6_DEV_ENABLED` is false unless the bundle is a dev bundle
 *   (`__DEV__`) AND the operator explicitly opted in with
 *   `EXPO_PUBLIC_STYLIST6_DEV_BRIDGE=true`. Both are build-time constants,
 *   so every call site below is dead code in release.
 * - The bridge URL is loopback-only (127.0.0.1); there is no production
 *   endpoint for Stylist 6.
 * - `invokeStylist6Bridge` throws when the gate is off, and surfaces the
 *   bridge's real failure otherwise. It never falls back to Stylist 5: a
 *   Stylist 6 failure stays a Stylist 6 failure.
 */
export const STYLIST6_DEV_ENABLED =
  __DEV__ && process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE === 'true';

export type DevEngine = 'stylist5' | 'stylist6' | 'stylist6cloud';

const devPort = (): number => {
  const raw = Number(process.env.EXPO_PUBLIC_STYLIST6_DEV_PORT || 3127);
  return Number.isFinite(raw) && raw > 0 ? raw : 3127;
};

export const STYLIST6_DEV_URL = `http://127.0.0.1:${devPort()}/generate`;

export interface Stylist6BridgeRequest {
  style: string;
  occasion: string;
  brief: string;
  weather?: { temp: number; condition: string };
  budget?: { amount: number; currency: string };
  preferences?: unknown;
  recentSignatures?: string[];
  excludeItemIds?: string[];
  /** Dev-only opt-in: run the hosted NVIDIA visual review on the backend. */
  enableVisual?: boolean;
}

export function buildStylist6BridgeBody(args: {
  request: Stylist6BridgeRequest;
  catalogRows: Record<string, unknown>[];
}): { request: Stylist6BridgeRequest; catalogRows: Record<string, unknown>[] } {
  return { request: args.request, catalogRows: args.catalogRows };
}

function failureForBridge(code: unknown, stage = 'stylist6_ranking'): OutfitGenerationFailure {
  const known: Record<string, 'outfitMaker.generateFailed' | 'outfitMaker.noMatchingClothing' | 'outfitMaker.catalogUnavailable' | 'outfitMaker.requestTooLarge' | 'outfitMaker.invalidRequest'> = {
    stylist6_no_feasible_looks: 'outfitMaker.noMatchingClothing',
    stylist6_empty_catalog: 'outfitMaker.catalogUnavailable',
    stylist6_ranking_failed: 'outfitMaker.generateFailed',
    invalid_stylist6_request: 'outfitMaker.invalidRequest',
    invalid_generation_request: 'outfitMaker.requestTooLarge',
  };
  const key = typeof code === 'string' && code in known ? known[code] : 'outfitMaker.generateFailed';
  return new OutfitGenerationFailure(key, typeof code === 'string' ? code : 'stylist6_ranking_failed', stage);
}

/**
 * POST the dev request to the local Stylist 6 bridge and return the
 * edge-shaped payload (success, outfits, generationVersion, ...).
 * Throws OutfitGenerationFailure on any transport or ranking failure.
 */
export async function invokeStylist6Bridge(
  body: { request: Stylist6BridgeRequest; catalogRows: Record<string, unknown>[] },
  options?: { signal?: AbortSignal }
): Promise<Record<string, any>> {
  if (!STYLIST6_DEV_ENABLED) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_dev_disabled', 'stylist6_dev');
  }
  let response: Response;
  try {
    response = await fetch(STYLIST6_DEV_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: options?.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_bridge_unreachable', 'stylist6_dev');
  }
  let data: Record<string, any>;
  try {
    data = (await response.json()) as Record<string, any>;
  } catch {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_bad_response', 'stylist6_ranking');
  }
  if (!response.ok || data?.success !== true || !Array.isArray(data?.outfits)) {
    throw failureForBridge(data?.code);
  }
  return data;
}

/** True only for decks genuinely produced by the local Stylist 6 bridge. */
export function isStylist6DevVersion(generationVersion: unknown): boolean {
  return typeof generationVersion === 'string' && generationVersion.startsWith('stylist-6-dev-');
}

/** True only for decks genuinely produced by Stylist 6 cloud staging. */
export function isStylist6CloudVersion(generationVersion: unknown): boolean {
  return typeof generationVersion === 'string' && generationVersion.startsWith('stylist-6-cloud-');
}

/**
 * Developer-only receipt summary for manual iOS visual reviews.
 *
 * The cloud has always returned the visual provider's bounded receipt, but
 * the simulator previously displayed only "NVIDIA_ABSTAINED", hiding whether
 * the cause was an unavailable reference, an image format, a deadline or an
 * actual provider response. Never render product URLs, JWTs, image bytes,
 * arbitrary error strings, token billing metadata or request fingerprints.
 */
const VISUAL_REASON_LABELS: Record<string, string> = {
  reference_unresolved: 'reference_unresolved',
  product_image_unresolved: 'product_image_unresolved',
  image_toolchain_unavailable: 'image_toolchain_unavailable',
  image_format_or_dimensions_invalid: 'image_format',
  provider_transport: 'provider_transport',
  provider_deadline: 'provider_deadline',
  batch_deadline_media: 'batch_deadline_media',
  batch_deadline_in_flight: 'batch_deadline_in_flight',
  batch_deadline_queued: 'batch_deadline_queued',
  batch_budget_before_dispatch: 'batch_budget',
  invalid_candidate: 'invalid_candidate',
};
const boundedReceiptCount = (value: unknown): number | null =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 && value <= 12
    ? value : null;

export function stylist6VisualDiagnostic(response: unknown): string | null {
  if (!response || typeof response !== 'object') return null;
  const body = response as Record<string, unknown>;
  const status = typeof body.visualStatus === 'string' && /^NVIDIA_(?:ABSTAINED|APPLIED|NOT_RUN|EVALUATED_NO_CHANGE)$/.test(body.visualStatus)
    ? body.visualStatus : null;
  if (!status) return null;
  const receipt = body.visualReceipt;
  if (!receipt || typeof receipt !== 'object') return status;
  const r = receipt as Record<string, unknown>;
  const calls = boundedReceiptCount(r.providerCalls);
  const valid = boundedReceiptCount(r.validVerdicts);
  const abstained = boundedReceiptCount(r.abstentions);
  const counters = calls !== null && valid !== null && abstained !== null
    ? ` c${calls}/v${valid}/a${abstained}` : '';
  const shown = boundedReceiptCount(r.displayedLooks);
  const judged = boundedReceiptCount(r.displayedJudged);
  // A valid verdict for an alternate or later-rejected look is not proof
  // that all *displayed* outfits were reviewed. Keep those counts distinct.
  const coverage = shown !== null && shown > 0 && judged !== null && judged <= shown
    ? ` · shown${judged}/${shown}` : '';
  const events = Array.isArray(r.events) ? r.events.slice(0, 4) : [];
  const failure = events.find(x =>
    !!x && typeof x === 'object' &&
    Object.hasOwn(VISUAL_REASON_LABELS, (x as Record<string, unknown>).reason as string));
  const entry = failure && typeof failure === 'object'
    ? failure as Record<string, unknown> : null;
  const reason = entry && typeof entry.reason === 'string'
    ? ` · ${VISUAL_REASON_LABELS[entry.reason]}` : '';
  const slot = ['top', 'outerwear', 'bottom', 'shoes'].includes(String(entry?.mediaSlot || ''))
    ? String(entry!.mediaSlot) : null;
  const cause = [
    'http_401', 'http_403', 'http_404', 'http_429', 'http_5xx',
    'invalid_media','oversize','decode_failed','fetch_failed','host_not_allowlisted',
  ].includes(String(entry?.mediaFailure || '')) ? String(entry!.mediaFailure) : null;
  const media = entry?.reason === 'product_image_unresolved' &&
    (slot || cause) ? ` (${[slot, cause].filter(Boolean).join('/')})` : '';
  return `${status}${counters}${coverage}${reason}${media}`;
}

export interface Stylist6CloudRequest {
  style: string;
  occasion: string;
  brief: string;
  weather?: { temp: number; condition: string };
  budget?: { amount: number; currency: string };
  preferences?: unknown;
  recentSignatures?: string[];
  excludeItemIds?: string[];
  requiredIds?: string[];
  sourceMode?: 'wardrobe' | 'catalog' | 'mixed';
  /** Dev-only opt-in: run the hosted NVIDIA visual review on the backend. */
  enableVisual?: boolean;
  /** Idempotency identity: redelivery replays the recorded outcome. */
  generationRequestId?: string;
  /** Active app locale: the server writes deck copy in this language
   * (unsupported values degrade to English server-side). */
  language?: 'en' | 'ru' | 'uz';
}

/** Normalized copy language for a Stylist 6 request, from the active i18n
 * locale (e.g. i18n.language). Unsupported/missing values map to 'en'. */
export function stylist6Language(locale: unknown): 'en' | 'ru' | 'uz' {
  return normalizeStylistLanguage(locale);
}

function failureForCloud(code: unknown, stage = 'stylist6_cloud'): OutfitGenerationFailure {
  const known: Record<string, 'outfitMaker.generateFailed' | 'outfitMaker.noMatchingClothing' | 'outfitMaker.catalogUnavailable' | 'outfitMaker.requestTooLarge' | 'outfitMaker.invalidRequest'> = {
    stylist6_no_feasible_looks: 'outfitMaker.noMatchingClothing',
    stylist6_empty_catalog: 'outfitMaker.catalogUnavailable',
    stylist6_ranking_failed: 'outfitMaker.generateFailed',
    invalid_stylist6_request: 'outfitMaker.invalidRequest',
    invalid_generation_request: 'outfitMaker.requestTooLarge',
    anchor_not_found: 'outfitMaker.invalidRequest',
    unauthorized: 'outfitMaker.generateFailed',
    rate_limited: 'outfitMaker.generateFailed',
    feature_limit_reached: 'outfitMaker.generateFailed',
    staging_transport_unconfigured: 'outfitMaker.generateFailed',
    staging_compute_unavailable: 'outfitMaker.generateFailed',
  };
  const key = typeof code === 'string' && code in known ? known[code] : 'outfitMaker.generateFailed';
  return new OutfitGenerationFailure(key, typeof code === 'string' ? code : 'stylist6_ranking_failed', stage);
}

/**
 * The staging proxy answers application-JSON errors (4xx/5xx). supabase-js
 * surfaces those as FunctionsHttpError and does NOT parse the body into
 * `data`, so the real server code must be recovered from the raw Response
 * (readable exactly once). The failure keeps the genuine code and stage so
 * the screen presents the server's verdict, not a generic transport label.
 */
async function cloudFailureFromInvokeError(invokeError: unknown): Promise<OutfitGenerationFailure> {
  const context = (invokeError as { context?: unknown } | null)?.context;
  if (context && typeof (context as Response).json === 'function') {
    try {
      const body = await (context as Response).json() as { code?: unknown; stage?: unknown } | null;
      if (body && typeof body.code === 'string') {
        return failureForCloud(body.code, typeof body.stage === 'string' ? body.stage : 'stylist6_cloud');
      }
    } catch {
      // Body already consumed or not JSON: fall through to the generic form.
    }
  }
  const message = (invokeError as { message?: unknown } | null)?.message;
  return failureForCloud(typeof message === 'string' ? message : 'stylist6_ranking_failed');
}

/**
 * POST the staging request to the isolated Stylist 6 cloud path
 * (authenticated Supabase staging proxy -> signed Vercel Node compute ->
 * server-side Stylist 6 engine). No client inventory is sent: the server
 * fetches shop and wardrobe rows itself. Failures surface intact and are
 * never replaced with a Stylist 5 deck. The optional abort signal
 * propagates through supabase-js so a cancelled screen stops the transport
 * (a request already dispatched server-side may still complete there).
 */
export async function invokeStylist6Cloud(
  body: { request: Stylist6CloudRequest },
  options?: { signal?: AbortSignal },
): Promise<Record<string, any>> {
  if (!STYLIST6_DEV_ENABLED) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_dev_disabled', 'stylist6_cloud');
  }
  // Call-time require keeps lib/supabase out of this module's import graph
  // (the dev-only cloud path must not force a Supabase client everywhere),
  // while staying interceptable by the test module registry.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { supabase } = require('../../lib/supabase') as typeof import('../../lib/supabase');
  const { data: sessionData } = await supabase.auth.getSession();
  const token = sessionData?.session?.access_token;
  if (!token) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'unauthorized', 'stylist6_cloud');
  }
  let data: Record<string, any> | null = null;
  let invokeError: unknown = null;
  try {
    const result = await supabase.functions.invoke('aiw-stylist6-node-staging', {
      headers: { Authorization: `Bearer ${token}` },
      body,
      ...(options?.signal ? { signal: options.signal } : {}),
    });
    data = result.data as Record<string, any> | null;
    invokeError = result.error;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'staging_compute_unavailable', 'stylist6_cloud');
  }
  if (invokeError) {
    throw await cloudFailureFromInvokeError(invokeError);
  }
  if (!data || data?.success !== true || !Array.isArray(data?.outfits)) {
    throw failureForCloud((data as any)?.code);
  }
  if (!isStylist6CloudVersion(data.generationVersion)) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_wrong_engine', 'stylist6_cloud');
  }
  return data;
}

export interface Stylist6DevAcceptance {
  accepted: Record<string, any>[];
  rejections: Record<string, number>;
}

const DEV_SLOTS = ['top', 'outerwear', 'bottom', 'shoes'] as const;

function isHttpsPhoto(value: unknown): boolean {
  try {
    const url = new URL(String(value));
    return url.protocol === 'https:' && !!url.hostname;
  } catch {
    return false;
  }
}

/** Structure/identity contract shared by both exact-deck acceptances. */
function lookStructureRejection(items: unknown[]): string | null {
  if (items.length !== 3 && items.length !== 4) return 'structure';
  const slots = items.map((item: any) => item?.macroCategory);
  if (slots.some((slot: unknown) => !(DEV_SLOTS as readonly unknown[]).includes(slot))) {
    return 'structure';
  }
  const count = (slot: string) => slots.filter((value: unknown) => value === slot).length;
  if (count('top') !== 1 || count('bottom') !== 1 || count('shoes') !== 1 || count('outerwear') > 1) {
    return 'structure';
  }
  if (items.some((item: any) => typeof item?.id !== 'string' || !item.id.trim())) {
    return 'invalid_product_identity';
  }
  if (new Set(items.map((item: any) => item.id)).size !== items.length) {
    return 'invalid_product_identity';
  }
  return null;
}

/**
 * Rendering-contract acceptance for exact Stylist 6 dev decks. The shared
 * Stylist 5 taste gates (base-shirt-only tops, style-judge repair,
 * closest-match recovery) would rewrite or reject the ranker's signature
 * looks — fine-knit base layers in particular — so the dev path validates
 * only what the outfit card needs to render truthfully: 3-4 slotted items
 * with unique string ids, reachable photos, shop provenance and prices.
 * Budgets are still enforced downstream by the screen's fitsCap filter;
 * anything that fails here is counted, never repaired or substituted.
 */
export function acceptStylist6DevDeck(outfits: unknown): Stylist6DevAcceptance {
  const rejections: Record<string, number> = {};
  const reject = (reason: string) => {
    rejections[reason] = (rejections[reason] || 0) + 1;
  };
  const accepted: Record<string, any>[] = [];
  for (const look of Array.isArray(outfits) ? outfits : []) {
    const items = (look as { items?: unknown })?.items;
    if (!Array.isArray(items)) {
      reject('structure');
      continue;
    }
    const structural = lookStructureRejection(items);
    if (structural) {
      reject(structural);
      continue;
    }
    if (items.some((item: any) => item?.isShopItem !== true)) {
      reject('source_or_anchor');
      continue;
    }
    if (items.some((item: any) => !isHttpsPhoto(item?.imageUrl || item?.image))) {
      reject('unrenderable_photo');
      continue;
    }
    accepted.push(look as Record<string, any>);
  }
  return { accepted, rejections };
}

/**
 * Rendering-contract acceptance for exact Stylist 6 CLOUD decks, per the
 * response's own sourceMode. The cloud deck flows to the card verbatim —
 * the shared Stylist 5 taste gates (repair, style judge, closest-match
 * recovery) must not rewrite the server-ranked plan — so this validates
 * only what the outfit card needs to render truthfully:
 * - catalog: every piece is a shop SKU with a merchant URL, price, currency;
 * - wardrobe: every piece is an owned item (no shoppable provenance);
 * - mixed: owned and shop pieces coexist, each keeping its own contract.
 * Anything that fails is counted, never repaired or substituted; budgets
 * still apply downstream through the screen's fitsCap filter.
 */
export function acceptStylist6CloudDeck(
  outfits: unknown,
  sourceMode: unknown,
): Stylist6DevAcceptance {
  const mode = sourceMode === 'wardrobe' || sourceMode === 'mixed' ? sourceMode : 'catalog';
  const rejections: Record<string, number> = {};
  const reject = (reason: string) => {
    rejections[reason] = (rejections[reason] || 0) + 1;
  };
  const accepted: Record<string, any>[] = [];
  for (const look of Array.isArray(outfits) ? outfits : []) {
    const items = (look as { items?: unknown })?.items;
    if (!Array.isArray(items)) {
      reject('structure');
      continue;
    }
    const structural = lookStructureRejection(items);
    if (structural) {
      reject(structural);
      continue;
    }
    const provenance = items.every((item: any) => {
      if (typeof item?.isShopItem !== 'boolean') return false;
      if (item.isShopItem) {
        // Shop SKUs keep the full merchant contract in every mode.
        if (mode === 'wardrobe') return false;
        return isHttpsPhoto(item?.shopUrl) && typeof item?.price === 'number'
          && Number.isFinite(item.price) && /^[A-Z]{3}$/.test(String(item?.currency || ''));
      }
      return mode !== 'catalog';
    });
    if (!provenance) {
      reject('source_or_anchor');
      continue;
    }
    if (items.some((item: any) => !isHttpsPhoto(item?.imageUrl || item?.image))) {
      reject('unrenderable_photo');
      continue;
    }
    accepted.push(look as Record<string, any>);
  }
  return { accepted, rejections };
}
