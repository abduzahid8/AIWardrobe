import { OutfitGenerationFailure } from './outfitGenerationFailure';

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
 * POST the staging request to the isolated Stylist 6 cloud path
 * (authenticated Supabase staging proxy -> signed Vercel Node compute ->
 * server-side Stylist 6 engine). No client inventory is sent: the server
 * fetches shop and wardrobe rows itself. Failures surface intact and are
 * never replaced with a Stylist 5 deck.
 */
export async function invokeStylist6Cloud(
  body: { request: Stylist6CloudRequest },
): Promise<Record<string, any>> {
  if (!STYLIST6_DEV_ENABLED) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_dev_disabled', 'stylist6_cloud');
  }
  const { supabase } = await import('../../lib/supabase');
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
    });
    data = result.data as Record<string, any> | null;
    invokeError = result.error;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'staging_compute_unavailable', 'stylist6_cloud');
  }
  if (invokeError || !data || data?.success !== true || !Array.isArray(data?.outfits)) {
    throw failureForCloud((data as any)?.code ?? (invokeError as any)?.message);
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
    if (!Array.isArray(items) || (items.length !== 3 && items.length !== 4)) {
      reject('structure');
      continue;
    }
    const slots = items.map((item: any) => item?.macroCategory);
    if (slots.some((slot: unknown) => !(DEV_SLOTS as readonly unknown[]).includes(slot))) {
      reject('structure');
      continue;
    }
    const count = (slot: string) => slots.filter((value: unknown) => value === slot).length;
    if (count('top') !== 1 || count('bottom') !== 1 || count('shoes') !== 1 || count('outerwear') > 1) {
      reject('structure');
      continue;
    }
    if (items.some((item: any) => typeof item?.id !== 'string' || !item.id.trim())) {
      reject('invalid_product_identity');
      continue;
    }
    if (new Set(items.map((item: any) => item.id)).size !== items.length) {
      reject('invalid_product_identity');
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
