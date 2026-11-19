/**
 * Stylist 6 — Strictly Opt-In User Appearance & Color Analysis Analyzer.
 *
 * Privacy & Safety Contract (AIWardrobe North Star):
 *  1. Strictly opt-in: `consent === true` is mandatory. Any other value returns
 *     `{ status: 'consent_required', appearance: null }` with zero I/O.
 *  2. SSRF-safe URL guard: HTTPS only, no credentials/non-443 ports, rejects
 *     loopback/RFC1918/link-local/metadata/IP literals, checks DNS resolution
 *     against private CIDRs, `redirect: 'error'`, bounded byte & magic-byte validation.
 *  3. Content-addressed caching: SHA-256 hash of validated photo bytes only
 *     (`appearance:v1:<sha256>`). Raw photo bytes and userId are NEVER persisted
 *     or sent to the vision provider.
 *  4. Zero PII to provider: provider receives ONLY validated image bytes, static
 *     fashion color-analysis prompt, and strict JSON schema.
 *  5. Strict JSON schema validation + confidence gating: never infers protected
 *     attributes (race, ethnicity, nationality, religion, age, medical/health)
 *     and never invents body measurements. Low confidence (<0.60) or poor
 *     lighting degrades cleanly to `'unknown'`.
 */

import { createHash } from 'node:crypto';
import { lookup as defaultDnsLookup } from 'node:dns/promises';
import { isIP } from 'node:net';
import { canonicalizeStyleColor } from '../../src/domain/personalStyleDna.ts';

export const APPEARANCE_SCHEMA_VERSION = 'user-appearance-color-v1';
export const APPEARANCE_CACHE_VERSION = 'appearance:v1';
export const MIN_APPEARANCE_CONFIDENCE = 0.6;
export const MAX_APPEARANCE_PHOTO_BYTES = 4 * 1024 * 1024; // 4 MiB
export const MIN_APPEARANCE_PHOTO_BYTES = 64;
export const APPEARANCE_FETCH_TIMEOUT_MS = 8_000;
export const APPEARANCE_CACHE_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

const ALLOWED_UNDERTONES = Object.freeze(['warm', 'cool', 'neutral', 'olive', 'unknown']);
const ALLOWED_DEPTHS = Object.freeze(['fair', 'light', 'medium', 'tan', 'deep', 'unknown']);
const ALLOWED_CONTRASTS = Object.freeze(['low', 'medium', 'high', 'unknown']);
const ALLOWED_METALS = Object.freeze(['gold', 'silver', 'mixed', 'unknown']);
const ALLOWED_LIGHTING = Object.freeze(['good', 'mixed', 'poor', 'unknown']);
const ALLOWED_SEASONS = Object.freeze([
  'light_spring', 'warm_spring', 'bright_spring', 'spring',
  'light_summer', 'cool_summer', 'soft_summer', 'summer',
  'soft_autumn', 'warm_autumn', 'deep_autumn', 'autumn',
  'cool_winter', 'deep_winter', 'bright_winter', 'winter',
  'unknown',
]);

const UNDERTONE_SET = new Set(ALLOWED_UNDERTONES);
const DEPTH_SET = new Set(ALLOWED_DEPTHS);
const CONTRAST_SET = new Set(ALLOWED_CONTRASTS);
const METAL_SET = new Set(ALLOWED_METALS);
const LIGHTING_SET = new Set(ALLOWED_LIGHTING);
const SEASON_SET = new Set(ALLOWED_SEASONS);

/** Strict JSON schema passed to OpenAI-compatible vision providers. */
export const APPEARANCE_RESPONSE_FORMAT = Object.freeze({
  type: 'json_schema',
  json_schema: {
    name: 'aiw_opt_in_personal_color_analysis',
    strict: true,
    schema: {
      type: 'object',
      properties: {
        skinUndertone: { type: 'string', enum: [...ALLOWED_UNDERTONES] },
        skinDepth: { type: 'string', enum: [...ALLOWED_DEPTHS] },
        contrastLevel: { type: 'string', enum: [...ALLOWED_CONTRASTS] },
        colorSeason: { type: 'string', enum: [...ALLOWED_SEASONS] },
        flatteringColors: { type: 'array', items: { type: 'string' }, maxItems: 10 },
        powerNeutrals: { type: 'array', items: { type: 'string' }, maxItems: 6 },
        avoidColors: { type: 'array', items: { type: 'string' }, maxItems: 8 },
        metalTone: { type: 'string', enum: [...ALLOWED_METALS] },
        lightingQuality: { type: 'string', enum: [...ALLOWED_LIGHTING] },
        confidence: { type: 'number', minimum: 0, maximum: 1 },
      },
      required: [
        'skinUndertone',
        'skinDepth',
        'contrastLevel',
        'colorSeason',
        'flatteringColors',
        'powerNeutrals',
        'avoidColors',
        'metalTone',
        'lightingQuality',
        'confidence',
      ],
      additionalProperties: false,
    },
  },
});

const APPEARANCE_ANALYSIS_PROMPT = [
  'You are a professional personal color analyst assisting an opt-in fashion styling session.',
  'Assess ONLY optical wardrobe color-harmony attributes visible in the photograph:',
  '- skinUndertone (warm, cool, neutral, olive, or unknown)',
  '- skinDepth (fair, light, medium, tan, deep, or unknown)',
  '- contrastLevel between hair/eyes/skin (low, medium, high, or unknown)',
  '- colorSeason (12-season palette or unknown)',
  '- metalTone (gold, silver, mixed, or unknown)',
  '- flatteringColors, powerNeutrals, avoidColors (standard fashion color names only)',
  '- lightingQuality (good, mixed, poor, or unknown) and numeric confidence (0..1).',
  'STRICT RULES:',
  '- NEVER infer or output race, ethnicity, nationality, religion, age, medical/health traits, gender identity, weight, height, or body measurements.',
  '- If lighting has heavy color casts, is dark/blurry, or the subject is not clearly visible, set confidence below 0.55 and return "unknown" rather than guessing.',
  '- Return ONLY valid JSON matching the strict schema.',
].join('\n');

const BLOCKED_HOSTNAMES = new Set([
  'localhost',
  'localhost.localdomain',
  'metadata',
  'metadata.google.internal',
  'metadata.goog',
  'instance-data',
]);

const BLOCKED_HOST_SUFFIXES = [
  '.localhost',
  '.local',
  '.internal',
  '.localdomain',
  '.home.arpa',
];

function isPrivateOrReservedIpv4(ip) {
  const parts = ip.split('.').map(Number);
  if (parts.length !== 4 || parts.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) {
    return true;
  }
  const [a, b] = parts;
  if (a === 0) return true;                          // 0.0.0.0/8
  if (a === 10) return true;                         // 10.0.0.0/8 RFC1918
  if (a === 100 && b >= 64 && b <= 127) return true; // 100.64.0.0/10 CGNAT / Alibaba metadata
  if (a === 127) return true;                        // 127.0.0.0/8 loopback
  if (a === 169 && b === 254) return true;           // 169.254.0.0/16 link-local & AWS/GCP metadata
  if (a === 172 && b >= 16 && b <= 31) return true;  // 172.16.0.0/12 RFC1918
  if (a === 192 && b === 168) return true;           // 192.168.0.0/16 RFC1918
  if (a === 192 && b === 0 && parts[2] === 0) return true; // 192.0.0.0/24
  if (a === 192 && b === 0 && parts[2] === 2) return true; // TEST-NET-1
  if (a === 198 && (b === 18 || b === 19)) return true;    // 198.18.0.0/15 benchmark
  if (a >= 224) return true;                         // multicast & reserved
  return false;
}

export function isPrivateOrReservedIp(ip) {
  const normalized = String(ip || '').trim().toLowerCase();
  const family = isIP(normalized);
  if (family === 4) return isPrivateOrReservedIpv4(normalized);
  if (family === 6) {
    if (normalized === '::' || normalized === '::1') return true;
    // IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1 or ::ffff:7f00:1)
    if (normalized.startsWith('::ffff:')) {
      const rest = normalized.slice(7);
      if (isIP(rest) === 4) return isPrivateOrReservedIpv4(rest);
      const hexPair = rest.match(/^([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
      if (hexPair) {
        const hi = parseInt(hexPair[1], 16);
        const lo = parseInt(hexPair[2], 16);
        return isPrivateOrReservedIpv4(`${(hi >> 8) & 255}.${hi & 255}.${(lo >> 8) & 255}.${lo & 255}`);
      }
      return true;
    }
    const firstHextet = parseInt(normalized.split(':')[0] || '0', 16);
    if (Number.isNaN(firstHextet)) return true;
    if ((firstHextet & 0xfe00) === 0xfc00) return true; // fc00::/7 ULA
    if ((firstHextet & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
    if ((firstHextet & 0xff00) === 0xff00) return true; // ff00::/8 multicast
    return false;
  }
  return true;
}

/**
 * SSRF-safe URL validator for user-submitted opt-in photo URLs.
 */
export function validateAppearancePhotoUrl(rawUrl, options = {}) {
  if (typeof rawUrl !== 'string' || !rawUrl.trim() || rawUrl.length > 2048) {
    return { ok: false, reason: 'invalid_url' };
  }
  let parsed;
  try {
    parsed = new URL(rawUrl.trim());
  } catch {
    return { ok: false, reason: 'invalid_url' };
  }

  if (parsed.protocol !== 'https:') {
    return { ok: false, reason: 'https_required' };
  }
  if (parsed.username || parsed.password) {
    return { ok: false, reason: 'credentials_forbidden_in_url' };
  }
  if (parsed.port && parsed.port !== '443') {
    return { ok: false, reason: 'non_standard_port_forbidden' };
  }

  const hostname = parsed.hostname.toLowerCase().replace(/\.$/, '');
  if (!hostname || hostname.length > 253) {
    return { ok: false, reason: 'invalid_hostname' };
  }
  if (BLOCKED_HOSTNAMES.has(hostname) || BLOCKED_HOST_SUFFIXES.some((s) => hostname.endsWith(s))) {
    return { ok: false, reason: 'private_or_internal_host_forbidden' };
  }

  // Block raw IPv4, bracketed IPv6, and integer/hex/octal IP representations
  const unbracketed = hostname.startsWith('[') && hostname.endsWith(']')
    ? hostname.slice(1, -1)
    : hostname;
  if (isIP(unbracketed) !== 0) {
    return { ok: false, reason: 'ip_literal_forbidden' };
  }
  if (/^(?:0x[0-9a-f]+|\d+)$/i.test(hostname) || hostname.split('.').every((seg) => /^(?:0x[0-9a-f]+|0[0-7]*|\d+)$/i.test(seg))) {
    return { ok: false, reason: 'ip_literal_forbidden' };
  }
  if (!hostname.includes('.')) {
    return { ok: false, reason: 'single_label_host_forbidden' };
  }

  if (Array.isArray(options.allowedHosts) && options.allowedHosts.length > 0) {
    const allowed = options.allowedHosts.some((entry) => {
      const h = String(entry || '').trim().toLowerCase();
      if (!h) return false;
      return h.startsWith('*.') ? hostname.endsWith(h.slice(1)) : hostname === h;
    });
    if (!allowed) {
      return { ok: false, reason: 'host_not_allowlisted' };
    }
  }

  return { ok: true, url: parsed.toString(), hostname };
}

function detectImageMimeType(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 12) return null;
  // JPEG: FF D8 FF
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return 'image/jpeg';
  }
  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (
    buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47 &&
    buffer[4] === 0x0d && buffer[5] === 0x0a && buffer[6] === 0x1a && buffer[7] === 0x0a
  ) {
    return 'image/png';
  }
  // WebP: RIFF....WEBP
  if (
    buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46 &&
    buffer[8] === 0x57 && buffer[9] === 0x45 && buffer[10] === 0x42 && buffer[11] === 0x50
  ) {
    return 'image/webp';
  }
  return null;
}

async function readBoundedResponseBytes(response, maxBytes = MAX_APPEARANCE_PHOTO_BYTES) {
  const contentLength = response?.headers?.get?.('content-length');
  if (contentLength && /^\d+$/.test(contentLength) && Number(contentLength) > maxBytes) {
    void response.body?.cancel?.().catch(() => {});
    throw new Error('photo_oversize');
  }
  if (response?.body?.getReader) {
    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value?.byteLength) continue;
        total += value.byteLength;
        if (total > maxBytes) {
          await reader.cancel().catch(() => {});
          throw new Error('photo_oversize');
        }
        chunks.push(Buffer.from(value));
      }
      return Buffer.concat(chunks, total);
    } finally {
      reader.releaseLock();
    }
  }
  if (typeof response?.arrayBuffer === 'function') {
    const buf = Buffer.from(await response.arrayBuffer());
    if (buf.length > maxBytes) throw new Error('photo_oversize');
    return buf;
  }
  throw new Error('photo_fetch_invalid_body');
}

function normalizeColorArray(list, max) {
  if (!Array.isArray(list)) return null;
  const seen = new Set();
  const out = [];
  for (const raw of list.slice(0, max * 2)) {
    if (typeof raw !== 'string') continue;
    const c = canonicalizeStyleColor(raw);
    if (c !== 'unknown' && !seen.has(c)) {
      seen.add(c);
      out.push(c);
      if (out.length >= max) break;
    }
  }
  return out;
}

function buildUnknownFallbackAppearance(confidence, lightingQuality) {
  return {
    skinUndertone: 'unknown',
    skinDepth: 'unknown',
    contrastLevel: 'unknown',
    colorSeason: 'unknown',
    flatteringColors: [],
    powerNeutrals: [],
    avoidColors: [],
    metalTone: 'unknown',
    lightingQuality,
    confidence,
    lowConfidenceFallback: true,
    provenance: 'opt_in_photo_analysis',
  };
}

/**
 * Strictly validate and sanitize raw VLM output into an allowlisted appearance
 * profile. Strips any extra/forbidden keys, enforces exact enums & bounds, and
 * degrades cleanly to `'unknown'` whenever confidence is below threshold or
 * lighting quality is `'poor'`.
 */
export function validateAppearanceAnalysisPayload(rawOutput, options = {}) {
  const minConfidence =
    typeof options.minConfidence === 'number' && Number.isFinite(options.minConfidence)
      ? Math.max(0, Math.min(1, options.minConfidence))
      : MIN_APPEARANCE_CONFIDENCE;

  let payload = rawOutput;
  if (typeof payload === 'string') {
    try {
      payload = JSON.parse(payload);
    } catch {
      return { ok: false, reason: 'invalid_json', appearance: null };
    }
  } else if (payload && typeof payload === 'object' && typeof payload.content === 'string') {
    try {
      payload = JSON.parse(payload.content);
    } catch {
      return { ok: false, reason: 'invalid_json', appearance: null };
    }
  }

  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return { ok: false, reason: 'payload_not_object', appearance: null };
  }

  const normEnum = (val) => (typeof val === 'string' ? val.trim().toLowerCase().replace(/[\s-]+/g, '_') : '');

  const skinUndertone = normEnum(payload.skinUndertone);
  const skinDepth = normEnum(payload.skinDepth);
  const contrastLevel = normEnum(payload.contrastLevel);
  const colorSeason = normEnum(payload.colorSeason);
  const metalTone = normEnum(payload.metalTone);
  const lightingQuality = payload.lightingQuality == null ? 'unknown' : normEnum(payload.lightingQuality);
  const confidence = payload.confidence;

  if (
    !UNDERTONE_SET.has(skinUndertone) ||
    !DEPTH_SET.has(skinDepth) ||
    !CONTRAST_SET.has(contrastLevel) ||
    !SEASON_SET.has(colorSeason) ||
    !METAL_SET.has(metalTone) ||
    !LIGHTING_SET.has(lightingQuality) ||
    typeof confidence !== 'number' ||
    !Number.isFinite(confidence) ||
    confidence < 0 ||
    confidence > 1
  ) {
    return { ok: false, reason: 'schema_validation_failed', appearance: null };
  }

  const flatteringColors = normalizeColorArray(payload.flatteringColors, 10);
  const powerNeutrals = normalizeColorArray(payload.powerNeutrals, 6);
  const rawAvoidColors = normalizeColorArray(payload.avoidColors, 8);
  if (!flatteringColors || !powerNeutrals || !rawAvoidColors) {
    return { ok: false, reason: 'color_arrays_required', appearance: null };
  }
  const positiveSet = new Set([...flatteringColors, ...powerNeutrals]);
  const avoidColors = rawAvoidColors.filter((c) => !positiveSet.has(c));

  const roundedConfidence = Math.round(confidence * 1000) / 1000;

  if (roundedConfidence < minConfidence || lightingQuality === 'poor') {
    return {
      ok: true,
      lowConfidence: true,
      appearance: buildUnknownFallbackAppearance(roundedConfidence, lightingQuality),
    };
  }

  // Strictly allowlisted output object — never spreads raw model output, so
  // hallucinated protected attributes or invented body measurements can never leak.
  return {
    ok: true,
    lowConfidence: false,
    appearance: {
      skinUndertone,
      skinDepth,
      contrastLevel,
      colorSeason,
      flatteringColors,
      powerNeutrals,
      avoidColors,
      metalTone,
      lightingQuality,
      confidence: roundedConfidence,
      lowConfidenceFallback: false,
      provenance: 'opt_in_photo_analysis',
    },
  };
}

/**
 * Perform strictly opt-in appearance color analysis on a user-provided photo.
 *
 * @param {object} params
 * @param {string} [params.photoUrl]      HTTPS URL (validated by SSRF guard)
 * @param {Buffer|Uint8Array} [params.photoBytes] Direct image bytes (preferred when already uploaded)
 * @param {boolean} params.consent        MUST be strictly `true`
 * @param {string} [params.userId]        Accepted for caller ergonomics; NEVER sent to provider or cache key
 * @param {Function} params.providerFn    Async VLM callback receiving zero-PII payload
 * @param {Map|object} [params.cache]     Optional sync Map or async distributed cache
 * @param {Function} [params.fetchFn]     Optional fetch override (for tests)
 * @param {Function} [params.dnsLookupFn] Optional DNS lookup override (for tests)
 * @param {string[]} [params.allowedHosts] Optional host allowlist
 * @param {number} [params.minConfidence] Minimum confidence floor (default 0.60)
 */
export async function analyzeUserAppearance({
  photoUrl,
  photoBytes,
  consent,
  userId,
  providerFn,
  cache,
  fetchFn,
  dnsLookupFn,
  allowedHosts,
  minConfidence = MIN_APPEARANCE_CONFIDENCE,
} = {}) {
  // Explicitly unused for privacy: userId must NEVER leave this boundary.
  void userId;

  // 1. Strict opt-in consent gate — zero I/O, zero cache touch, zero provider call
  if (consent !== true) {
    return {
      status: 'consent_required',
      appearance: null,
      cached: false,
      photoSha256: null,
      reason: 'explicit_opt_in_consent_required',
    };
  }

  // 2. Obtain & validate photo bytes (either directly supplied or via SSRF-safe HTTPS fetch)
  let buffer = null;
  if (photoBytes != null) {
    if (Buffer.isBuffer(photoBytes)) {
      buffer = photoBytes;
    } else if (photoBytes instanceof Uint8Array) {
      buffer = Buffer.from(photoBytes);
    } else {
      return {
        status: 'invalid_photo',
        appearance: null,
        cached: false,
        photoSha256: null,
        reason: 'photo_bytes_invalid_type',
      };
    }
  } else if (photoUrl != null) {
    const check = validateAppearancePhotoUrl(photoUrl, { allowedHosts });
    if (!check.ok) {
      return {
        status: 'blocked_url',
        appearance: null,
        cached: false,
        photoSha256: null,
        reason: check.reason,
      };
    }

    // DNS rebinding & private-network guard before opening socket
    const shouldCheckDns = typeof dnsLookupFn === 'function' || typeof fetchFn !== 'function';
    if (shouldCheckDns) {
      const lookup = typeof dnsLookupFn === 'function' ? dnsLookupFn : defaultDnsLookup;
      try {
        const records = await lookup(check.hostname, { all: true });
        const list = Array.isArray(records) ? records : records ? [records] : [];
        if (list.length === 0 || list.some((r) => isPrivateOrReservedIp(r.address || r))) {
          return {
            status: 'blocked_url',
            appearance: null,
            cached: false,
            photoSha256: null,
            reason: 'dns_resolved_private_ip',
          };
        }
      } catch {
        return {
          status: 'blocked_url',
          appearance: null,
          cached: false,
          photoSha256: null,
          reason: 'dns_lookup_failed',
        };
      }
    }

    const doFetch = typeof fetchFn === 'function' ? fetchFn : globalThis.fetch;
    if (typeof doFetch !== 'function') {
      return {
        status: 'fetch_failed',
        appearance: null,
        cached: false,
        photoSha256: null,
        reason: 'fetch_unavailable',
      };
    }

    try {
      const response = await doFetch(check.url, {
        method: 'GET',
        redirect: 'error',
        headers: {
          'User-Agent': 'AIWardrobe-AppearanceAnalyzer/1.0',
          Accept: 'image/jpeg,image/png,image/webp',
        },
        signal: AbortSignal.timeout(APPEARANCE_FETCH_TIMEOUT_MS),
      });
      if (!response?.ok) {
        void response?.body?.cancel?.().catch(() => {});
        return {
          status: 'fetch_failed',
          appearance: null,
          cached: false,
          photoSha256: null,
          reason: `http_${Number(response?.status) || 'error'}`,
        };
      }
      buffer = await readBoundedResponseBytes(response, MAX_APPEARANCE_PHOTO_BYTES);
    } catch (err) {
      return {
        status: 'fetch_failed',
        appearance: null,
        cached: false,
        photoSha256: null,
        reason: err?.message === 'photo_oversize' ? 'photo_oversize' : 'photo_fetch_error',
      };
    }
  } else {
    return {
      status: 'invalid_photo',
      appearance: null,
      cached: false,
      photoSha256: null,
      reason: 'photo_required',
    };
  }

  if (buffer.length < MIN_APPEARANCE_PHOTO_BYTES || buffer.length > MAX_APPEARANCE_PHOTO_BYTES) {
    return {
      status: 'invalid_photo',
      appearance: null,
      cached: false,
      photoSha256: null,
      reason: buffer.length > MAX_APPEARANCE_PHOTO_BYTES ? 'photo_oversize' : 'photo_too_small',
    };
  }

  const mimeType = detectImageMimeType(buffer);
  if (!mimeType) {
    return {
      status: 'invalid_photo',
      appearance: null,
      cached: false,
      photoSha256: null,
      reason: 'unsupported_image_format',
    };
  }

  // 3. Content-addressed SHA-256 cache lookup (zero PII in cache key)
  const photoSha256 = createHash('sha256').update(buffer).digest('hex');
  const cacheKey = `${APPEARANCE_CACHE_VERSION}:${photoSha256}`;

  if (cache && typeof cache.get === 'function') {
    try {
      const hit = await cache.get(cacheKey);
      if (hit && typeof hit === 'object') {
        const revalidated = validateAppearanceAnalysisPayload(hit, { minConfidence });
        if (revalidated.ok) {
          return {
            status: revalidated.lowConfidence ? 'low_confidence' : 'ok',
            appearance: revalidated.appearance,
            cached: true,
            photoSha256,
            cacheKey,
          };
        }
      }
    } catch {
      // Cache read failure degrades cleanly to provider invocation.
    }
  }

  // 4. Invoke VLM provider with strictly zero-PII request payload
  if (typeof providerFn !== 'function') {
    return {
      status: 'provider_unavailable',
      appearance: null,
      cached: false,
      photoSha256,
      cacheKey,
      reason: 'provider_fn_required',
    };
  }

  let rawProviderResult;
  try {
    rawProviderResult = await providerFn({
      mimeType,
      imageBase64: buffer.toString('base64'),
      prompt: APPEARANCE_ANALYSIS_PROMPT,
      responseFormat: APPEARANCE_RESPONSE_FORMAT,
      schemaVersion: APPEARANCE_SCHEMA_VERSION,
    });
  } catch {
    return {
      status: 'provider_error',
      appearance: null,
      cached: false,
      photoSha256,
      cacheKey,
      reason: 'provider_call_failed',
    };
  }

  // 5. Strictly validate VLM output & apply confidence gating
  const validated = validateAppearanceAnalysisPayload(rawProviderResult, { minConfidence });
  if (!validated.ok) {
    return {
      status: 'invalid_provider_output',
      appearance: null,
      cached: false,
      photoSha256,
      cacheKey,
      reason: validated.reason,
    };
  }

  // Cache only schema-valid high-confidence results (never cache failures or malformed output)
  if (!validated.lowConfidence && cache && typeof cache.set === 'function') {
    try {
      await cache.set(cacheKey, validated.appearance, APPEARANCE_CACHE_TTL_MS);
    } catch {
      // Non-fatal cache write failure
    }
  }

  return {
    status: validated.lowConfidence ? 'low_confidence' : 'ok',
    appearance: validated.appearance,
    cached: false,
    photoSha256,
    cacheKey,
  };
}
