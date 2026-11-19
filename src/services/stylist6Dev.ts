import { OutfitGenerationFailure } from './outfitGenerationFailure';
import {
  normalizeStylistLanguage,
  type StylingPresentation,
  type StylingBodyProfile,
  type StylingAppearanceProfile,
  type OwnedWardrobeSummary,
} from '../domain/stylingContext';
import { budgetFxContext, validateBudgetFxSnapshot } from '../domain/budgetFx';
import type { ParsedBudget } from '../utils/outfitBudget';

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

/** Opt-in TestFlight engine, never an implicit App Store rollout.
 * Release builds use the server-owned JWT/HMAC gateway and do NOT expose
 * developer controls or the loopback bridge. Both opt-ins must be explicit;
 * production EAS profile has neither one.
 */
export const STYLIST6_CLOUD_ENABLED = STYLIST6_DEV_ENABLED || (
  !__DEV__ && process.env.EXPO_PUBLIC_STYLIST6_BETA === 'true' &&
  process.env.EXPO_PUBLIC_APP_STAGE === 'testflight'
);

/** Dev-simulator-only canary switch. Never routes TestFlight or production
 * releases to the experimental Render service. Existing Vercel staging is
 * the unchanged default. Only the dedicated Render Supabase proxy is used. */
export const STYLIST6_RENDER_CANARY_ENABLED =
  STYLIST6_DEV_ENABLED && process.env.EXPO_PUBLIC_STYLIST6_RENDER_CANARY === 'true';

/** Stylist 6 recency must not permanently exhaust a finite real catalog.
 * A professional stylist can reuse great shoes in NEW complete outfits
 * after a cooling-off deck. Reject exact products shown in the PREVIOUS
 * displayed deck; preserve older full-look signatures as a soft fashion
 * repetition penalty handled by the backend. Keep real user preferences,
 * explicitly forbidden colors/brands, exact budget and anchor hard.
 * Stylist 5 keeps its existing separate strict SKU-history logic.
 */
export function stylist6CloudRecentExclusions(
  history: {previousDeckProductKeys?: readonly string[]; olderRecentProductKeys?: readonly string[]} | null,
  anchorId?: string | null,
): string[] {
  const keys=Array.isArray(history?.previousDeckProductKeys)
    ? history.previousDeckProductKeys : [];
  const required=typeof anchorId==='string'?anchorId.trim():'';
  const ids=[...new Set(keys
    .filter((x): x is string => typeof x==='string' && x.startsWith('id:'))
    .map(x=>x.slice(3).trim())
    .filter(x=>x.length>0 && x.length<=200 && x!==required))];
  return ids.slice(-12);
}

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
  presentation?: StylingPresentation;
  bodyProfile?: StylingBodyProfile;
  appearanceProfile?: StylingAppearanceProfile;
  ownedWardrobeSummary?: OwnedWardrobeSummary;
  dislikedPatterns?: string[];
  recentSignatures?: string[];
  excludeItemIds?: string[];
  requiredIds?: string[];
  sourceMode?: 'wardrobe' | 'catalog' | 'mixed' | 'shop' | 'hybrid';
  language?: 'en' | 'ru' | 'uz';
  /** Dev-only opt-in: run the hosted NVIDIA visual review on the backend. */
  enableVisual?: boolean;
}

const VALID_PRESENTATIONS = new Set<StylingPresentation>(['menswear', 'womenswear', 'unisex']);
const SHA256_HEX_RE = /^[a-f0-9]{64}$/i;

/**
 * Normalizes a color token for Stylist 6 requests.
 * Fixes legacy style-quiz Midnight Blue hex (`#0A1931` -> `navy`) and
 * American/British gray spelling (`gray` -> `grey`) without mutating
 * Stylist 5 callers.
 */
export function normalizeStylistColorToken(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim().toLowerCase();
  if (!trimmed) return null;
  if (trimmed === '#0a1931' || trimmed === 'midnight blue' || trimmed === 'midnight_blue' || trimmed === 'midnight-blue') {
    return 'navy';
  }
  if (trimmed === 'gray') return 'grey';
  if (trimmed === 'unknown' || trimmed === 'none') return null;
  return trimmed.slice(0, 40);
}

/** Derives a safe Stylist 6 presentation value from an explicit gender field. */
export function deriveStylist6Presentation(gender: unknown): StylingPresentation | undefined {
  if (typeof gender !== 'string') return undefined;
  const normalized = gender.trim().toLowerCase();
  if (normalized === 'male' || normalized === 'man' || normalized === 'men' || normalized === 'menswear') {
    return 'menswear';
  }
  if (normalized === 'female' || normalized === 'woman' || normalized === 'women' || normalized === 'womenswear') {
    return 'womenswear';
  }
  if (normalized === 'other' || normalized === 'unisex' || normalized === 'nonbinary' || normalized === 'non-binary') {
    return 'unisex';
  }
  return undefined;
}

function sanitizeStringList(
  values: unknown,
  maxItems: number,
  maxChars: number,
  normalizer?: (raw: unknown) => string | null,
): string[] {
  if (!Array.isArray(values)) return [];
  const out: string[] = [];
  for (const item of values) {
    const cleaned = normalizer
      ? normalizer(item)
      : typeof item === 'string'
        ? item.trim().toLowerCase().slice(0, maxChars)
        : null;
    if (cleaned && !out.includes(cleaned)) {
      out.push(cleaned);
      if (out.length >= maxItems) break;
    }
  }
  return out;
}

/**
 * Normalizes stored StylePreference values for Stylist 6 without losing
 * legacy `#0A1931` Midnight Blue or `gray` selections.
 */
export function normalizeStylist6Preferences(rawPrefs: unknown): Record<string, unknown> | undefined {
  if (!rawPrefs || typeof rawPrefs !== 'object' || Array.isArray(rawPrefs)) return undefined;
  const src = rawPrefs as Record<string, unknown>;
  const favoriteColors = sanitizeStringList(src.favoriteColors, 16, 40, normalizeStylistColorToken);
  const avoidColors = sanitizeStringList(
    Array.isArray(src.avoidColors) ? src.avoidColors : src.avoidedColors,
    16,
    40,
    normalizeStylistColorToken,
  );
  const likedPatterns = sanitizeStringList(src.likedPatterns, 16, 40);
  const dislikedPatterns = sanitizeStringList(src.dislikedPatterns, 20, 40);
  const favoriteBrands = sanitizeStringList(src.favoriteBrands, 20, 60);
  const avoidBrands = sanitizeStringList(src.avoidBrands, 20, 60);
  const preferredFits = sanitizeStringList(src.preferredFits, 6, 40);
  const fitPreference = typeof src.fitPreference === 'string' && src.fitPreference.trim()
    ? src.fitPreference.trim().toLowerCase().slice(0, 40)
    : preferredFits[0];

  const out: Record<string, unknown> = {};
  if (typeof src.stylePersonality === 'string' && src.stylePersonality.trim()) {
    out.stylePersonality = src.stylePersonality.trim().slice(0, 40);
  }
  if (typeof src.bodyType === 'string' && src.bodyType.trim()) {
    out.bodyType = src.bodyType.trim().toLowerCase().slice(0, 40);
  }
  if (favoriteColors.length > 0) out.favoriteColors = favoriteColors;
  if (avoidColors.length > 0) out.avoidColors = avoidColors;
  if (likedPatterns.length > 0) out.likedPatterns = likedPatterns;
  if (dislikedPatterns.length > 0) out.dislikedPatterns = dislikedPatterns;
  if (fitPreference) out.fitPreference = fitPreference;
  if (preferredFits.length > 0) out.preferredFits = preferredFits;
  if (favoriteBrands.length > 0) out.favoriteBrands = favoriteBrands;
  if (avoidBrands.length > 0) out.avoidBrands = avoidBrands;
  if (typeof src.workDressCode === 'string' && src.workDressCode.trim()) {
    out.workDressCode = src.workDressCode.trim().toLowerCase().slice(0, 40);
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/**
 * Summarizes owned wardrobe items into aggregate style DNA (no user IDs,
 * item IDs, or private photo URLs) so catalog-only Stylist 6 requests still
 * honor the wearer's real closet palette, categories, brands, and formality.
 */
export function summarizeOwnedWardrobeForStylist6(
  items: readonly Record<string, unknown>[] | null | undefined,
): OwnedWardrobeSummary | undefined {
  if (!Array.isArray(items) || items.length === 0) return undefined;
  const colorCounts = new Map<string, number>();
  const categoryCounts = new Map<string, number>();
  const brandCounts = new Map<string, number>();
  const formalityCounts: Record<'formal' | 'smart' | 'casual' | 'athletic', number> = {
    formal: 0,
    smart: 0,
    casual: 0,
    athletic: 0,
  };
  let classifiedFormality = 0;

  const bump = (map: Map<string, number>, key: string | null) => {
    if (!key) return;
    map.set(key, (map.get(key) || 0) + 1);
  };

  for (const item of items.slice(0, 250)) {
    if (!item || typeof item !== 'object') continue;
    const rawColor = item.primaryColor ?? item.primary_color ?? (Array.isArray(item.color) ? item.color[0] : item.color);
    const normColor = normalizeStylistColorToken(rawColor);
    if (normColor && normColor !== 'neutral') bump(colorCounts, normColor);

    const rawCat = item.category ?? item.macroCategory ?? item.subCategory ?? item.type;
    if (typeof rawCat === 'string' && rawCat.trim()) {
      bump(categoryCounts, rawCat.trim().toLowerCase().slice(0, 40));
    }

    if (typeof item.brand === 'string' && item.brand.trim()) {
      bump(brandCounts, item.brand.trim().slice(0, 50));
    }

    const occasions = Array.isArray(item.occasions)
      ? (item.occasions as unknown[])
          .filter((o: unknown): o is string => typeof o === 'string')
          .map((o: string) => o.toLowerCase())
      : [];
    const text = `${String(item.subCategory ?? item.type ?? '')} ${String(item.name ?? '')} ${String(item.style ?? '')}`.toLowerCase();
    let bucket: 'formal' | 'smart' | 'casual' | 'athletic' | null = null;
    if (occasions.includes('sport') || /\b(gym|sport|running|jogger|track|athletic|sneaker)\b/.test(text)) {
      bucket = 'athletic';
    } else if (occasions.includes('formal') || /\b(tuxedo|suit|oxford|dress shirt|evening|gown)\b/.test(text)) {
      bucket = 'formal';
    } else if (occasions.includes('work') || occasions.includes('date') || /\b(blazer|loafer|derby|trouser|chino|button|coat|cardigan|cashmere|merino)\b/.test(text)) {
      bucket = 'smart';
    } else if (occasions.includes('casual') || occasions.includes('travel') || /\b(tee|t-shirt|jeans|denim|hoodie|shorts|polo|sweater)\b/.test(text)) {
      bucket = 'casual';
    }
    if (bucket) {
      formalityCounts[bucket] += 1;
      classifiedFormality += 1;
    }
  }

  const topKeys = (map: Map<string, number>, limit: number): string[] =>
    [...map.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, limit)
      .map(([k]) => k);

  const formalityMix: Record<string, number> = {};
  if (classifiedFormality > 0) {
    for (const [key, count] of Object.entries(formalityCounts)) {
      if (count > 0) {
        formalityMix[key] = Math.round((count / classifiedFormality) * 100) / 100;
      }
    }
  }

  return {
    totalItems: items.length,
    dominantColors: topKeys(colorCounts, 5),
    dominantCategories: topKeys(categoryCounts, 5),
    topBrands: topKeys(brandCounts, 5),
    formalityMix,
  };
}

function sanitizeBodyProfile(input: StylingBodyProfile | null | undefined): StylingBodyProfile | undefined {
  if (!input || typeof input !== 'object') return undefined;
  const out: StylingBodyProfile = {};
  if (typeof input.gender === 'string') {
    const g = input.gender.trim().toLowerCase();
    if (g && g !== 'prefer_not_to_say' && g.length <= 32) out.gender = g;
  }
  if (typeof input.heightCm === 'number' && Number.isFinite(input.heightCm) && input.heightCm >= 100 && input.heightCm <= 250) {
    out.heightCm = Math.round(input.heightCm * 10) / 10;
  }
  if (typeof input.weightKg === 'number' && Number.isFinite(input.weightKg) && input.weightKg >= 25 && input.weightKg <= 300) {
    out.weightKg = Math.round(input.weightKg * 10) / 10;
  }
  if (typeof input.bodyType === 'string' && input.bodyType.trim()) {
    out.bodyType = input.bodyType.trim().toLowerCase().slice(0, 40);
  }
  if (input.measurements && typeof input.measurements === 'object' && !Array.isArray(input.measurements)) {
    const cleanMeasurements: Record<string, number> = {};
    let count = 0;
    for (const [key, value] of Object.entries(input.measurements)) {
      if (count >= 16) break;
      if (/^[a-zA-Z][a-zA-Z0-9_]{0,31}$/.test(key) && typeof value === 'number' && Number.isFinite(value) && value >= 10 && value <= 250) {
        cleanMeasurements[key] = Math.round(value * 10) / 10;
        count += 1;
      }
    }
    if (count > 0) out.measurements = cleanMeasurements;
  }
  if (typeof input.fitPreference === 'string' && input.fitPreference.trim()) {
    out.fitPreference = input.fitPreference.trim().toLowerCase().slice(0, 40);
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function sanitizeAppearanceProfile(input: StylingAppearanceProfile | null | undefined): StylingAppearanceProfile | undefined {
  // Strictly opt-in: never forward appearance attributes without explicit consent.
  if (!input || typeof input !== 'object' || input.consentGranted !== true) return undefined;
  const out: StylingAppearanceProfile = { consentGranted: true };
  if (typeof input.skinUndertone === 'string' && input.skinUndertone.trim()) {
    out.skinUndertone = input.skinUndertone.trim().toLowerCase().slice(0, 32);
  }
  if (typeof input.skinDepth === 'string' && input.skinDepth.trim()) {
    out.skinDepth = input.skinDepth.trim().toLowerCase().slice(0, 32);
  }
  if (typeof input.contrastLevel === 'string' && input.contrastLevel.trim()) {
    out.contrastLevel = input.contrastLevel.trim().toLowerCase().slice(0, 32);
  }
  if (typeof input.colorSeason === 'string' && input.colorSeason.trim()) {
    out.colorSeason = input.colorSeason.trim().toLowerCase().slice(0, 48);
  }
  const flattering = sanitizeStringList(input.flatteringColors, 12, 40, normalizeStylistColorToken);
  if (flattering.length > 0) out.flatteringColors = flattering;
  const avoid = sanitizeStringList(input.avoidColors, 12, 40, normalizeStylistColorToken);
  if (avoid.length > 0) out.avoidColors = avoid;
  if (typeof input.metalTone === 'string' && input.metalTone.trim()) {
    out.metalTone = input.metalTone.trim().toLowerCase().slice(0, 32);
  }
  if (typeof input.photoSha256 === 'string' && SHA256_HEX_RE.test(input.photoSha256.trim())) {
    out.photoSha256 = input.photoSha256.trim().toLowerCase();
  }
  return out;
}

function sanitizeOwnedWardrobeSummary(input: OwnedWardrobeSummary | null | undefined): OwnedWardrobeSummary | undefined {
  if (!input || typeof input !== 'object') return undefined;
  if (typeof input.totalItems !== 'number' || !Number.isSafeInteger(input.totalItems) || input.totalItems <= 0) {
    return undefined;
  }
  const formalityMix: Record<string, number> = {};
  if (input.formalityMix && typeof input.formalityMix === 'object' && !Array.isArray(input.formalityMix)) {
    let count = 0;
    for (const [k, v] of Object.entries(input.formalityMix)) {
      if (count >= 8) break;
      if (/^[a-z_]{1,24}$/i.test(k) && typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1) {
        formalityMix[k.toLowerCase()] = Math.round(v * 100) / 100;
        count += 1;
      }
    }
  }
  return {
    totalItems: Math.min(input.totalItems, 10000),
    dominantColors: sanitizeStringList(input.dominantColors, 8, 40, normalizeStylistColorToken),
    dominantCategories: sanitizeStringList(input.dominantCategories, 8, 40),
    topBrands: sanitizeStringList(input.topBrands, 8, 50, (raw) =>
      typeof raw === 'string' && raw.trim() ? raw.trim().slice(0, 50) : null,
    ),
    formalityMix,
  };
}

function applyStylist6Personalization<T extends Partial<Stylist6BridgeRequest | Stylist6CloudRequest>>(
  source: T,
): T {
  const req = { ...source };
  delete req.presentation;
  delete req.bodyProfile;
  delete req.appearanceProfile;
  delete req.ownedWardrobeSummary;
  delete req.dislikedPatterns;

  if (source.presentation && VALID_PRESENTATIONS.has(source.presentation)) {
    req.presentation = source.presentation;
  }
  const cleanBody = sanitizeBodyProfile(source.bodyProfile);
  if (cleanBody) req.bodyProfile = cleanBody;

  const cleanAppearance = sanitizeAppearanceProfile(source.appearanceProfile);
  if (cleanAppearance) req.appearanceProfile = cleanAppearance;

  const cleanWardrobe = sanitizeOwnedWardrobeSummary(source.ownedWardrobeSummary);
  if (cleanWardrobe) req.ownedWardrobeSummary = cleanWardrobe;

  const cleanDislikedPatterns = sanitizeStringList(source.dislikedPatterns, 20, 40);
  if (cleanDislikedPatterns.length > 0) req.dislikedPatterns = cleanDislikedPatterns;

  return req;
}

function normalizeBridgeSourceMode(
  mode: unknown,
): 'wardrobe' | 'catalog' | 'mixed' | undefined {
  if (typeof mode !== 'string') return undefined;
  const normalized = mode.trim().toLowerCase();
  if (normalized === 'shop' || normalized === 'catalog') return 'catalog';
  if (normalized === 'hybrid' || normalized === 'mixed') return 'mixed';
  if (normalized === 'wardrobe') return 'wardrobe';
  return undefined;
}

export function buildStylist6BridgeBody(args: {
  request: Stylist6BridgeRequest;
  catalogRows: Record<string, unknown>[];
  wardrobeItems?: Record<string, unknown>[];
}): {
  request: Stylist6BridgeRequest;
  catalogRows: Record<string, unknown>[];
  wardrobeItems?: Record<string, unknown>[];
} {
  const req: Stylist6BridgeRequest = applyStylist6Personalization(args.request);
  if (Array.isArray(args.request.requiredIds)) {
    const cleanRequired = [...new Set(
      args.request.requiredIds
        .filter((id): id is string => typeof id === 'string' && id.trim().length > 0)
        .map((id) => id.trim()),
    )].slice(0, 4);
    if (cleanRequired.length > 0) {
      req.requiredIds = cleanRequired;
    } else {
      delete req.requiredIds;
    }
  }
  const normalizedSourceMode = normalizeBridgeSourceMode(args.request.sourceMode);
  if (normalizedSourceMode) {
    req.sourceMode = normalizedSourceMode;
  } else {
    delete req.sourceMode;
  }
  if (args.request.language !== undefined) {
    req.language = normalizeStylistLanguage(args.request.language);
  }
  const wardrobeItems = Array.isArray(args.wardrobeItems) && args.wardrobeItems.length > 0
    ? args.wardrobeItems
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
      .slice(0, 120)
      .map((item) => ({ ...item, isWardrobe: true, isShopItem: false }))
    : undefined;
  return {
    request: req,
    catalogRows: args.catalogRows,
    ...(wardrobeItems && wardrobeItems.length > 0 ? { wardrobeItems } : {}),
  };
}

function failureForBridge(code: unknown, stage = 'stylist6_ranking'): OutfitGenerationFailure {
  const known: Record<string, 'outfitMaker.generateFailed' | 'outfitMaker.noMatchingClothing' | 'outfitMaker.visualQualityUnmet' | 'outfitMaker.catalogUnavailable' | 'outfitMaker.requestTooLarge' | 'outfitMaker.invalidRequest'> = {
    stylist6_no_feasible_looks: 'outfitMaker.noMatchingClothing',
    stylist6_visual_quality_unmet: 'outfitMaker.visualQualityUnmet',
    stylist6_empty_catalog: 'outfitMaker.catalogUnavailable',
    stylist6_ranking_failed: 'outfitMaker.generateFailed',
    invalid_stylist6_request: 'outfitMaker.invalidRequest',
    invalid_generation_request: 'outfitMaker.requestTooLarge',
    anchor_not_found: 'outfitMaker.invalidRequest',
    anchor_outfit_unavailable: 'outfitMaker.noMatchingClothing',
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
  body: {
    request: Stylist6BridgeRequest;
    catalogRows: Record<string, unknown>[];
    wardrobeItems?: Record<string, unknown>[];
  },
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
  presentation?: StylingPresentation;
  bodyProfile?: StylingBodyProfile;
  appearanceProfile?: StylingAppearanceProfile;
  ownedWardrobeSummary?: OwnedWardrobeSummary;
  dislikedPatterns?: string[];
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

/**
 * Reconciles the client's budget FX context with the server's validated CBU
 * conversion snapshot so cross-currency cloud decks are validated against the
 * exact server rates, and tampered or missing snapshots fail closed.
 */
export function applyStylist6CloudBudgetFx(
  shopBudget: ParsedBudget | null | undefined,
  currencyConversion: unknown,
): ParsedBudget | null | undefined {
  if (!shopBudget) return shopBudget;
  const conv = currencyConversion && typeof currencyConversion === 'object'
    ? (currencyConversion as { active?: unknown; snapshot?: unknown })
    : null;
  if (conv?.active !== true) {
    if (!('fx' in shopBudget)) return shopBudget;
    const { fx: _fx, ...rest } = shopBudget;
    return rest;
  }
  if (!conv.snapshot) {
    throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable', 'budget_rates_unavailable', 'stylist6_cloud');
  }
  try {
    const validated = validateBudgetFxSnapshot(conv.snapshot);
    return { ...shopBudget, fx: budgetFxContext(validated) };
  } catch {
    throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable', 'budget_rates_unavailable', 'stylist6_cloud');
  }
}

function failureForCloud(code: unknown, stage = 'stylist6_cloud'): OutfitGenerationFailure {
  const known: Record<string, 'outfitMaker.generateFailed' | 'outfitMaker.noMatchingClothing' | 'outfitMaker.visualQualityUnmet' | 'outfitMaker.catalogUnavailable' | 'outfitMaker.requestTooLarge' | 'outfitMaker.invalidRequest' | 'outfitMaker.subscriptionRequired'> = {
    stylist6_no_feasible_looks: 'outfitMaker.noMatchingClothing',
    stylist6_visual_quality_unmet: 'outfitMaker.visualQualityUnmet',
    stylist6_empty_catalog: 'outfitMaker.catalogUnavailable',
    stylist6_ranking_failed: 'outfitMaker.generateFailed',
    invalid_stylist6_request: 'outfitMaker.invalidRequest',
    invalid_generation_request: 'outfitMaker.requestTooLarge',
    anchor_not_found: 'outfitMaker.invalidRequest',
    anchor_outfit_unavailable: 'outfitMaker.noMatchingClothing',
    unauthorized: 'outfitMaker.generateFailed',
    rate_limited: 'outfitMaker.generateFailed',
    feature_limit_reached: 'outfitMaker.subscriptionRequired',
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
      const body = await (context as Response).json() as {
        code?: unknown; stage?: unknown; diagnostics?: {
          modelFashionSelection?: unknown; visualReviewCoverage?: unknown;
          visualQualityDecisions?: Array<{reason?: string; quality?: number; scores?: unknown}>;
        };
      } | null;
      if (body && typeof body.code === 'string') {
        // Development QA only: a paid photo-quality failure must reveal the
        // safe aggregate AI decision receipt; never user IDs, SKU names,
        // merchant links, JWTs or images.
        if (__DEV__ && body.diagnostics) {
          console.warn('[S6 FASHION QA FAIL]', JSON.stringify({
            code:body.code,
            director:body.diagnostics.modelFashionSelection,
            coverage:body.diagnostics.visualReviewCoverage,
            decisions:(body.diagnostics.visualQualityDecisions || []).map(row=>({
              reason:row.reason,quality:row.quality,scores:row.scores,
            })),
          }));
        }
        return failureForCloud(body.code, typeof body.stage === 'string' ? body.stage : 'stylist6_cloud');
      }
    } catch {
      // Body already consumed or not JSON: fall through to the generic form.
    }
  }
  // SDK fetch/relay failures have NO server payload. They may happen AFTER
  // an already-paid generation was completed and logged HTTP200 by Supabase.
  // Classify only this transport failure for READ-ONLY same-ID recovery.
  // Never classify a genuine FunctionsHttpError / photo-veto as recoverable.
  const sdkName=(invokeError as {name?:unknown}|null)?.name;
  if(sdkName==='FunctionsFetchError'||sdkName==='FunctionsRelayError')
    return failureForCloud('staging_compute_unavailable','stylist6_transport');
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
  if (!STYLIST6_CLOUD_ENABLED) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_release_disabled', 'stylist6_cloud');
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
    const target = STYLIST6_RENDER_CANARY_ENABLED
      ? 'aiw-stylist6-node-render-staging' : 'aiw-stylist6-node-staging';
    const sanitizedBody = {
      ...body,
      request: applyStylist6Personalization(body.request),
    };
    const result = await supabase.functions.invoke(target, {
      headers: { Authorization: `Bearer ${token}` },
      body: sanitizedBody,
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

/**
 * READ-ONLY recovery of an ALREADY CHARGED, COMPLETE Stylist 6 generation.
 * Never calls the paid Edge Function, never retries POST and never bypasses
 * original source/PHOTO acceptance. Supabase RLS restricts SELECT to the
 * current authenticated owner, and rows expire after 45 minutes.
 * A missing/stale/invalid row gives null, never a fabricated outfit.
 */
export async function recoverStylist6CloudResult(
  requestId: string,options?: {signal?: AbortSignal},
): Promise<Record<string, any>|null> {
  if (!STYLIST6_CLOUD_ENABLED ||
      !/^[A-Za-z0-9_-]{8,128}$/.test(String(requestId||'')) ||
      options?.signal?.aborted) return null;
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {supabase}=require('../../lib/supabase') as typeof import('../../lib/supabase');
  try{
    const {data:session}=await supabase.auth.getSession();
    if(!session?.session?.access_token || options?.signal?.aborted)return null;
    let query=supabase.from('stylist6_completed_results')
      .select('response_payload').eq('request_id',requestId);
    if(options?.signal)query=query.abortSignal(options.signal);
    const {data,error}=await query.maybeSingle();
    if(error || !data || options?.signal?.aborted)return null;
    const response=(data as {response_payload?:unknown}).response_payload;
    if(!response || typeof response!=='object' || Array.isArray(response))
      return null;
    const deck=response as Record<string,any>;
    if(deck.generationRequestId!==requestId ||
       !stylist6CloudCompletion(deck) ||
       deck.visualStatus!=='NVIDIA_APPLIED' ||
       deck.visualReceipt?.displayedJudged!==deck.outfits.length ||
       !deck.outfits.every((look:any)=>
         look?.fashionDiagnostics?.visualVerdict==='JUDGED' &&
         Array.isArray(look?.items) &&
         (deck.sourceMode!=='catalog'||look.items.length===4)))
      return null;
    const accepted=acceptStylist6CloudDeck(deck.outfits,deck.sourceMode);
    if(accepted.accepted.length!==deck.outfits.length ||
       Object.keys(accepted.rejections).length>0)return null;
    return deck;
  }catch{return null;}
}

/**
 * Cloud completion contract: an honestly partial result is only admissible
 * when each shown card has its own model-approved visual evidence. The
 * authoritative backend cannot charge the user and then ask the client to
 * invent a third outfit. A normal full deck retains the existing contract.
 */
/** Distinguish a legitimate FASHION EDITOR rejection of a merely repetitive
 * third look from a lack of individually photo-approved candidates. Both are
 * safe partial decks, but they deserve different, truthful explanations.
 */
export function stylist6CuratedPartial(response: unknown, shownCount: number): boolean {
  if (stylist6CloudCompletion(response) !== 'verified_partial' ||
    !Number.isInteger(shownCount) || shownCount < 1 || shownCount > 2) return false;
  const r = response as Record<string, any>;
  const final = r.diagnostics?.modelFashionSelection?.photoEditorial;
  return final?.status === 'applied' && final?.chosen === shownCount &&
    Array.isArray(r.outfits) && r.outfits.length === shownCount &&
    r.visualReceipt?.displayedJudged === shownCount;
}

export function stylist6CloudCompletion(response: unknown): 'complete' | 'verified_partial' | null {
  if (!response || typeof response !== 'object') return null;
  const body = response as Record<string, any>;
  if (body.success !== true || !isStylist6CloudVersion(body.generationVersion) ||
      !Array.isArray(body.outfits)) return null;
  if (body.status === 'complete' && body.outfits.length === 3) return 'complete';
  if (body.status !== 'partial' || body.outfits.length < 1 ||
      body.outfits.length > 2 || body.visualStatus !== 'NVIDIA_APPLIED') return null;
  const receipt = body.visualReceipt;
  return receipt && receipt.visualApproval === 'verified_partial' &&
    receipt.displayedJudged === body.outfits.length &&
    receipt.displayedLooks === body.outfits.length &&
    body.outfits.every((look: any) => look?.fashionDiagnostics?.visualVerdict === 'JUDGED')
    ? 'verified_partial' : null;
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
export function acceptStylist6DevDeck(
  outfits: unknown,
  sourceMode?: unknown,
): Stylist6DevAcceptance {
  const mode = normalizeBridgeSourceMode(sourceMode) ?? 'catalog';
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
    const provenanceOk = mode === 'wardrobe'
      ? items.every((item: any) => item?.isShopItem === false)
      : mode === 'mixed'
        ? items.every((item: any) => typeof item?.isShopItem === 'boolean')
        : items.every((item: any) => item?.isShopItem === true);
    if (!provenanceOk) {
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

export type Stylist6SwapSlot = 'top' | 'outerwear' | 'bottom' | 'shoes' | 'accessory' | 'dress';

const VALID_SWAP_SLOTS = new Set<Stylist6SwapSlot>([
  'top',
  'outerwear',
  'bottom',
  'shoes',
  'accessory',
  'dress',
]);

export interface Stylist6SlotSwapParams {
  outfit: Record<string, unknown>;
  targetSlot: Stylist6SwapSlot;
  reason?: string;
  context?: Partial<Stylist6CloudRequest>;
  excludedIds?: readonly string[];
  authToken?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface Stylist6SlotSwapResult {
  outfit: Record<string, unknown>;
  swappedSlot: Stylist6SwapSlot;
  previousItemId: string | null;
  replacementItem: Record<string, unknown>;
  generationVersion?: string;
}

export interface Stylist6AppearanceAnalysisParams {
  photoUrl: string;
  consentGranted: boolean;
  authToken?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface Stylist6AppearanceAnalysisResult {
  status: 'analyzed' | 'consent_required' | 'unavailable';
  appearance: StylingAppearanceProfile | null;
}

function withBoundedSignal(
  timeoutMs: number,
  externalSignal?: AbortSignal,
): { signal: AbortSignal; wasTimeout: () => boolean; cleanup: () => void } {
  const controller = new AbortController();
  let timedOut = false;
  const onExternalAbort = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener('abort', onExternalAbort, { once: true });
    }
  }
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, Math.max(500, timeoutMs));
  const cleanup = () => {
    clearTimeout(timer);
    if (externalSignal) {
      externalSignal.removeEventListener('abort', onExternalAbort);
    }
  };
  return { signal: controller.signal, wasTimeout: () => timedOut, cleanup };
}

async function resolveStylist6AuthToken(
  explicitToken: string | undefined,
  stage: string,
): Promise<{
  token: string;
  supabase: typeof import('../../lib/supabase').supabase;
}> {
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
  const { supabase } = require('../../lib/supabase') as typeof import('../../lib/supabase');
  const trimmed = typeof explicitToken === 'string' ? explicitToken.trim() : '';
  if (trimmed) return { token: trimmed, supabase };
  const { data: sessionData } = await supabase.auth.getSession();
  const token = sessionData?.session?.access_token;
  if (!token) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'unauthorized', stage);
  }
  return { token, supabase };
}

function slotItemOf(item: unknown): Record<string, unknown> | null {
  return item && typeof item === 'object' && !Array.isArray(item)
    ? (item as Record<string, unknown>)
    : null;
}

function hasValidSwapProvenance(
  items: Record<string, unknown>[],
  mode: 'wardrobe' | 'catalog' | 'mixed',
): boolean {
  return items.every((item) => {
    if (typeof item.isShopItem !== 'boolean') return false;
    if (!isHttpsPhoto(item.imageUrl || item.image)) return false;
    if (item.isShopItem) {
      if (mode === 'wardrobe') return false;
      return isHttpsPhoto(item.shopUrl)
        && typeof item.price === 'number'
        && Number.isFinite(item.price)
        && /^[A-Z]{3}$/.test(String(item.currency || ''));
    }
    return mode !== 'catalog';
  });
}

function verifySwappedOutfit(args: {
  candidate: Record<string, unknown>;
  sourceMode: 'wardrobe' | 'catalog' | 'mixed';
  targetSlot: Stylist6SwapSlot;
  previousItemId: string | null;
  originalItems: Record<string, unknown>[];
  excludedSet: Set<string>;
}): Record<string, unknown> | null {
  const rawItems = args.candidate.items;
  if (!Array.isArray(rawItems)) return null;
  const items = rawItems.map(slotItemOf).filter((x): x is Record<string, unknown> => x !== null);
  if (items.length !== rawItems.length) return null;

  // Catalog mode strictly requires 4 unique shoppable SKUs.
  if (args.sourceMode === 'catalog' && items.length !== 4) return null;
  if (items.length < 3 || items.length > 4) return null;

  const ids = items.map((item) => (typeof item.id === 'string' ? item.id.trim() : ''));
  if (ids.some((id) => !id) || new Set(ids).size !== items.length) return null;

  const slots = items.map((item) => String(item.macroCategory || '').trim().toLowerCase());
  const usesStandardSlots = slots.every((s) => (DEV_SLOTS as readonly string[]).includes(s));
  if (usesStandardSlots) {
    const check = acceptStylist6CloudDeck([args.candidate], args.sourceMode);
    if (check.accepted.length !== 1 || Object.keys(check.rejections).length > 0) return null;
  } else {
    // Extended 4-item grammars (warm_weather_separates / one_piece_editorial)
    if (items.length !== 4) return null;
    if (slots.some((s) => !VALID_SWAP_SLOTS.has(s as Stylist6SwapSlot))) return null;
    if (new Set(slots).size !== items.length) return null;
    if (slots.includes('dress') && slots.includes('bottom')) return null;
    if (!hasValidSwapProvenance(items, args.sourceMode)) return null;
  }

  const replacement = items.find(
    (item) => String(item.macroCategory || '').trim().toLowerCase() === args.targetSlot,
  );
  if (!replacement || typeof replacement.id !== 'string') return null;
  const replacementId = replacement.id.trim();
  if (!replacementId) return null;
  if (args.previousItemId && replacementId === args.previousItemId) return null;
  if (args.excludedSet.has(replacementId)) return null;

  // Ensure locked non-target slots stayed intact (no silent rewrite of other items).
  const expectedLocked = args.originalItems
    .filter((item) => String(item.macroCategory || '').trim().toLowerCase() !== args.targetSlot)
    .map((item) => ({
      slot: String(item.macroCategory || '').trim().toLowerCase(),
      id: typeof item.id === 'string' ? item.id.trim() : '',
    }))
    .filter((entry) => entry.slot && entry.id);

  for (const locked of expectedLocked) {
    const matching = items.find(
      (item) => String(item.macroCategory || '').trim().toLowerCase() === locked.slot,
    );
    if (!matching || String(matching.id || '').trim() !== locked.id) return null;
  }

  return replacement;
}

/**
 * Requests a surgical 1-slot replacement for an existing Stylist 6 look
 * (`action: 'swap_slot'`). Locks the other 3 items, bounds transport timeout,
 * and enforces strict 4-SKU verification before returning any replacement.
 */
export async function requestStylist6SlotSwap(
  params: Stylist6SlotSwapParams,
): Promise<Stylist6SlotSwapResult> {
  if (!STYLIST6_CLOUD_ENABLED) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_release_disabled', 'stylist6_swap');
  }
  if (!VALID_SWAP_SLOTS.has(params.targetSlot)) {
    throw new OutfitGenerationFailure('outfitMaker.invalidRequest', 'invalid_stylist6_request', 'stylist6_swap');
  }
  const rawOriginalItems = params.outfit && Array.isArray(params.outfit.items)
    ? params.outfit.items
    : null;
  if (!rawOriginalItems || rawOriginalItems.length === 0) {
    throw new OutfitGenerationFailure('outfitMaker.invalidRequest', 'invalid_stylist6_request', 'stylist6_swap');
  }
  const originalItems = rawOriginalItems
    .map(slotItemOf)
    .filter((x): x is Record<string, unknown> => x !== null);
  const previousItem = originalItems.find(
    (item) => String(item.macroCategory || '').trim().toLowerCase() === params.targetSlot,
  );
  const previousItemId = typeof previousItem?.id === 'string' && previousItem.id.trim()
    ? previousItem.id.trim()
    : null;

  const excludedIds = [...new Set([
    ...(Array.isArray(params.excludedIds) ? params.excludedIds : []),
    ...(previousItemId ? [previousItemId] : []),
  ]
    .filter((id): id is string => typeof id === 'string' && id.trim().length > 0)
    .map((id) => id.trim().slice(0, 200)))].slice(0, 40);

  const reason = typeof params.reason === 'string' && params.reason.trim()
    ? params.reason.trim().slice(0, 160)
    : undefined;
  const sanitizedContext = params.context
    ? applyStylist6Personalization(params.context)
    : undefined;

  const { token, supabase } = await resolveStylist6AuthToken(params.authToken, 'stylist6_swap');
  const target = STYLIST6_RENDER_CANARY_ENABLED
    ? 'aiw-stylist6-node-render-staging'
    : 'aiw-stylist6-node-staging';
  const { signal, wasTimeout, cleanup } = withBoundedSignal(params.timeoutMs ?? 25_000, params.signal);

  let data: Record<string, unknown> | null = null;
  let invokeError: unknown = null;
  try {
    const result = await supabase.functions.invoke(target, {
      headers: { Authorization: `Bearer ${token}` },
      body: {
        action: 'swap_slot',
        outfit: params.outfit,
        targetSlot: params.targetSlot,
        ...(reason ? { reason } : {}),
        ...(sanitizedContext ? { context: sanitizedContext } : {}),
        excludedIds,
      },
      signal,
    });
    data = result.data as Record<string, unknown> | null;
    invokeError = result.error;
  } catch (error) {
    if (wasTimeout()) {
      throw new OutfitGenerationFailure('outfitMaker.requestTimeout', 'request_timeout', 'stylist6_swap');
    }
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'staging_compute_unavailable', 'stylist6_swap');
  } finally {
    cleanup();
  }

  if (wasTimeout()) {
    throw new OutfitGenerationFailure('outfitMaker.requestTimeout', 'request_timeout', 'stylist6_swap');
  }
  if (invokeError) {
    throw await cloudFailureFromInvokeError(invokeError);
  }
  if (!data || data.success !== true) {
    throw failureForCloud(data?.code, 'stylist6_swap');
  }

  const candidate = (data.outfit && typeof data.outfit === 'object' && !Array.isArray(data.outfit)
    ? data.outfit
    : Array.isArray(data.outfits) && data.outfits[0] && typeof data.outfits[0] === 'object'
      ? data.outfits[0]
      : null) as Record<string, unknown> | null;

  if (!candidate) {
    throw new OutfitGenerationFailure('outfitMaker.noMatchingClothing', 'stylist6_swap_invalid_outfit', 'stylist6_swap');
  }

  const rawSourceMode = data.sourceMode ?? params.context?.sourceMode;
  const sourceMode: 'wardrobe' | 'catalog' | 'mixed' =
    rawSourceMode === 'wardrobe' || rawSourceMode === 'mixed' ? rawSourceMode : 'catalog';

  const replacementItem = verifySwappedOutfit({
    candidate,
    sourceMode,
    targetSlot: params.targetSlot,
    previousItemId,
    originalItems,
    excludedSet: new Set(excludedIds),
  });
  if (!replacementItem) {
    throw new OutfitGenerationFailure('outfitMaker.noMatchingClothing', 'stylist6_swap_invalid_outfit', 'stylist6_swap');
  }

  const generationVersion = typeof data.generationVersion === 'string'
    ? data.generationVersion
    : undefined;
  return {
    outfit: candidate,
    swappedSlot: params.targetSlot,
    previousItemId,
    replacementItem,
    ...(generationVersion ? { generationVersion } : {}),
  };
}

/**
 * Calls opt-in Stylist 6 appearance analysis (`action: 'analyze_appearance'`).
 * Strictly enforces explicit consent before touching auth or the network,
 * validates HTTPS photo URLs, and sanitizes the response.
 */
export async function requestStylist6AppearanceAnalysis(
  params: Stylist6AppearanceAnalysisParams,
): Promise<Stylist6AppearanceAnalysisResult> {
  if (params.consentGranted !== true) {
    return { status: 'consent_required', appearance: null };
  }
  if (!STYLIST6_CLOUD_ENABLED) {
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'stylist6_release_disabled', 'stylist6_appearance');
  }
  const photoUrl = typeof params.photoUrl === 'string' ? params.photoUrl.trim() : '';
  if (!isHttpsPhoto(photoUrl)) {
    throw new OutfitGenerationFailure('outfitMaker.invalidRequest', 'invalid_stylist6_request', 'stylist6_appearance');
  }

  const { token, supabase } = await resolveStylist6AuthToken(params.authToken, 'stylist6_appearance');
  const target = STYLIST6_RENDER_CANARY_ENABLED
    ? 'aiw-stylist6-node-render-staging'
    : 'aiw-stylist6-node-staging';
  const { signal, wasTimeout, cleanup } = withBoundedSignal(params.timeoutMs ?? 20_000, params.signal);

  let data: Record<string, unknown> | null = null;
  let invokeError: unknown = null;
  try {
    const result = await supabase.functions.invoke(target, {
      headers: { Authorization: `Bearer ${token}` },
      body: {
        action: 'analyze_appearance',
        photoUrl,
        consentGranted: true,
      },
      signal,
    });
    data = result.data as Record<string, unknown> | null;
    invokeError = result.error;
  } catch (error) {
    if (wasTimeout()) {
      throw new OutfitGenerationFailure('outfitMaker.requestTimeout', 'request_timeout', 'stylist6_appearance');
    }
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new OutfitGenerationFailure('outfitMaker.generateFailed', 'staging_compute_unavailable', 'stylist6_appearance');
  } finally {
    cleanup();
  }

  if (wasTimeout()) {
    throw new OutfitGenerationFailure('outfitMaker.requestTimeout', 'request_timeout', 'stylist6_appearance');
  }
  if (invokeError) {
    throw await cloudFailureFromInvokeError(invokeError);
  }
  if (!data) {
    return { status: 'unavailable', appearance: null };
  }
  if (data.status === 'consent_required') {
    return { status: 'consent_required', appearance: null };
  }
  if (data.success === false) {
    throw failureForCloud(data.code, 'stylist6_appearance');
  }
  const rawAppearance = data.appearance && typeof data.appearance === 'object' && !Array.isArray(data.appearance)
    ? (data.appearance as StylingAppearanceProfile)
    : null;
  const sanitized = rawAppearance
    ? sanitizeAppearanceProfile({ ...rawAppearance, consentGranted: true })
    : undefined;
  if (!sanitized) {
    return { status: 'unavailable', appearance: null };
  }
  return { status: 'analyzed', appearance: sanitized };
}
