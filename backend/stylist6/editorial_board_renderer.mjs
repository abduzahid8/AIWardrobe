/**
 * Stylist 6 — High-Resolution Proportion-Aware Editorial Board Renderer.
 *
 * Renders studio-grade flat-lay boards via Sharp so both Vision-Language Models
 * and human reviewers see true garment proportions, silhouettes, textures,
 * color relationships, and footwear grounding — instead of tiny disconnected
 * thumbnails where a shoe looks the same scale as an overcoat.
 *
 * Security & honesty guarantees:
 *   - Enforces SSRF host validation (`HOST_ALLOWLIST` + private-IP / loopback /
 *     metadata / non-HTTPS / port / credential rejection) before ANY fetch.
 *   - Enforces byte (4 MiB) and pixel (36 MP / 8192 edge / static-only) caps.
 *   - Transparently decodes AVIF, WebP, PNG, and JPEG (stripping EXIF/alpha).
 *   - Never fabricates product imagery; missing/unauthorized slots are recorded
 *     explicitly in `slotStatus` and fail closed when `requireAllPhotos: true`.
 *   - Injectable bounded LRU photo byte cache (entry + byte-size bounded) gives
 *     0ms CDN overhead when re-rendering boards during surgical slot swaps.
 *   - Single raw-RGB composite + single strictly XML-escaped SVG overlay.
 */

import { createHash } from 'node:crypto';
import sharp from 'sharp';
import { HOST_ALLOWLIST } from '../../scripts/stylist6/image_access.mjs';

export const EDITORIAL_BOARD_RENDERER_VERSION = 's6-editorial-board-renderer-v1';

export const STUDIO_BACKDROP_HEX = '#F6F5F2';
export const STUDIO_CARD_HEX = '#FFFFFF';
export const STUDIO_BORDER_HEX = '#E4E0D8';
export const STUDIO_INK_HEX = '#1F1E1C';
export const STUDIO_MUTED_HEX = '#6D6860';
export const STUDIO_PLACEHOLDER_HEX = '#ECE9E2';

export const LOOK_BOARD_WIDTH = 1024;
export const LOOK_BOARD_HEIGHT = 1024;
export const PORTFOLIO_SHEET_WIDTH = 1800;
export const PORTFOLIO_SHEET_HEIGHT = 1000;
export const SHOWROOM_BOARD_WIDTH = 1600;
export const SHOWROOM_BOARD_HEIGHT = 1600;
export const SWAP_BOARD_WIDTH = 1400;
export const SWAP_BOARD_HEIGHT = 900;

export const MAX_INPUT_IMAGE_BYTES = 4 * 1024 * 1024;
export const MAX_INPUT_IMAGE_PIXELS = 36 * 1024 * 1024;
export const MAX_INPUT_EDGE = 8192;
export const MIN_INPUT_EDGE = 16;
export const DEFAULT_LOOK_BOARD_MAX_BYTES = 512 * 1024;
export const DEFAULT_PORTFOLIO_MAX_BYTES = 850 * 1024;
export const DEFAULT_SHOWROOM_MAX_BYTES = 1024 * 1024;
export const DEFAULT_SWAP_MAX_BYTES = 768 * 1024;
export const DEFAULT_FETCH_TIMEOUT_MS = 8000;

const SUPPORTED_DECODE_FORMATS = new Set(['jpeg', 'png', 'webp', 'avif']);
const CANONICAL_SLOTS = ['outerwear', 'top', 'bottom', 'shoes'];

const PRIVATE_HOST_PATTERNS = [
  /^localhost$/i,
  /\.local$/i,
  /\.internal$/i,
  /^127\./,
  /^10\./,
  /^0\./,
  /^169\.254\./,
  /^192\.168\./,
  /^172\.(?:1[6-9]|2\d|3[01])\./,
  /^\[?::1\]?$/,
  /^\[?f[cd][0-9a-f]{2}:/i,
  /^\[?fe[89ab][0-9a-f]:/i,
];

function escapeXml(raw) {
  return String(raw || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function sanitizeLabelText(raw, maxLen = 40) {
  if (typeof raw !== 'string') return '';
  return raw
    .replace(/https?:\/\/[^\s]+|\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b/gi, '')
    .replace(/[\u0000-\u001f\u007f<>]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxLen);
}

function formatPriceTag(item) {
  const price = Number(item?.price);
  const currency = typeof item?.currency === 'string' && /^[A-Z]{3}$/.test(item.currency.trim())
    ? item.currency.trim()
    : '';
  if (!Number.isFinite(price) || price <= 0 || !currency) return '';
  const rounded = price >= 1000 ? Math.round(price).toLocaleString('en-US') : String(Math.round(price * 100) / 100);
  if (currency === 'USD') return `$${rounded}`;
  if (currency === 'EUR') return `EUR ${rounded}`;
  if (currency === 'GBP') return `GBP ${rounded}`;
  return `${rounded} ${currency}`;
}

/**
 * SSRF-hardened image URL validation preserving existing HOST_ALLOWLIST controls.
 */
export function validateEditorialImageUrl(url, {
  authorizePhoto = null,
  allowlist = HOST_ALLOWLIST,
} = {}) {
  if (typeof url !== 'string' || !url.trim()) {
    return { ok: false, reason: 'invalid_url', host: null };
  }
  let parsed;
  try {
    parsed = new URL(url.trim());
  } catch {
    return { ok: false, reason: 'invalid_url', host: null };
  }
  if (parsed.protocol !== 'https:' || parsed.username || parsed.password || parsed.port) {
    return { ok: false, reason: 'unsafe_url', host: parsed.hostname || null };
  }
  const host = parsed.hostname.toLowerCase();
  if (!host || PRIVATE_HOST_PATTERNS.some((re) => re.test(host))) {
    return { ok: false, reason: 'ssrf_blocked', host };
  }
  if (typeof authorizePhoto === 'function') {
    let authorized = false;
    try {
      authorized = authorizePhoto(url) === true;
    } catch {
      authorized = false;
    }
    if (!authorized) return { ok: false, reason: 'host_not_allowlisted', host };
    return { ok: true, reason: null, host };
  }
  if (!(allowlist instanceof Set) || !allowlist.has(host)) {
    return { ok: false, reason: 'host_not_allowlisted', host };
  }
  return { ok: true, reason: null, host };
}

/**
 * Bounded LRU photo byte cache (capped by BOTH entry count AND total byte size).
 * Eliminates repeat merchant CDN fetches across surgical slot swaps and portfolio sheets.
 */
export function createPhotoByteCache({
  maxEntries = 256,
  maxBytes = 64 * 1024 * 1024,
  ttlMs = 15 * 60_000,
  nowFn = () => Date.now(),
} = {}) {
  const entryCap = Math.max(1, Math.floor(Number(maxEntries) || 256));
  const byteCap = Math.max(64 * 1024, Math.floor(Number(maxBytes) || 64 * 1024 * 1024));
  const ttl = Math.max(1000, Math.floor(Number(ttlMs) || 15 * 60_000));

  const store = new Map();
  let currentBytes = 0;
  let hits = 0;
  let misses = 0;
  let evictions = 0;

  function sizeOf(value) {
    if (Buffer.isBuffer(value)) return value.length;
    if (Buffer.isBuffer(value?.bytes)) return value.bytes.length;
    if (Buffer.isBuffer(value?.jpegBuffer)) return value.jpegBuffer.length;
    return 0;
  }

  function removeKey(key) {
    const existing = store.get(key);
    if (!existing) return false;
    currentBytes = Math.max(0, currentBytes - existing.size);
    store.delete(key);
    return true;
  }

  function evictIfNeeded(incomingSize) {
    while ((store.size >= entryCap || currentBytes + incomingSize > byteCap) && store.size > 0) {
      const oldestKey = store.keys().next().value;
      removeKey(oldestKey);
      evictions++;
    }
  }

  function get(key) {
    const entry = store.get(key);
    if (!entry) {
      misses++;
      return null;
    }
    if (nowFn() - entry.at > ttl) {
      removeKey(key);
      misses++;
      return null;
    }
    // LRU promotion
    store.delete(key);
    store.set(key, entry);
    hits++;
    return entry.value;
  }

  function set(key, value) {
    const sz = sizeOf(value);
    if (sz <= 0 || sz > byteCap) return false;
    if (store.has(key)) removeKey(key);
    evictIfNeeded(sz);
    store.set(key, { value, size: sz, at: nowFn() });
    currentBytes += sz;
    return true;
  }

  function has(key) {
    const entry = store.get(key);
    if (!entry) return false;
    if (nowFn() - entry.at > ttl) {
      removeKey(key);
      return false;
    }
    return true;
  }

  function clear() {
    store.clear();
    currentBytes = 0;
  }

  function stats() {
    return {
      entries: store.size,
      bytes: currentBytes,
      maxEntries: entryCap,
      maxBytes: byteCap,
      hits,
      misses,
      evictions,
    };
  }

  return { get, set, has, delete: removeKey, clear, stats };
}

/**
 * Decode and validate AVIF/WebP/PNG/JPEG bytes under strict byte & pixel caps.
 * Returns metadata + validated original bytes (or resized raw RGB when box is specified).
 */
export async function decodeNormalizedPhotoBytes(rawBytes, {
  width = null,
  height = null,
  background = STUDIO_CARD_HEX,
  sharpImpl = sharp,
} = {}) {
  if (!Buffer.isBuffer(rawBytes) || rawBytes.length < 32 || rawBytes.length > MAX_INPUT_IMAGE_BYTES) {
    const err = new Error('oversize_or_invalid_media');
    err.code = Buffer.isBuffer(rawBytes) && rawBytes.length > MAX_INPUT_IMAGE_BYTES ? 'oversize' : 'invalid_media';
    throw err;
  }

  const sharpOpts = {
    failOn: 'truncated',
    limitInputPixels: MAX_INPUT_IMAGE_PIXELS,
    animated: false,
    sequentialRead: true,
  };

  let meta;
  try {
    meta = await sharpImpl(rawBytes, sharpOpts).metadata();
  } catch {
    const err = new Error('decode_failed');
    err.code = 'decode_failed';
    throw err;
  }

  const isAvif = meta.format === 'avif' || (meta.format === 'heif' && meta.compression === 'av1');
  if (!(SUPPORTED_DECODE_FORMATS.has(meta.format) || isAvif)
      || !Number.isSafeInteger(meta.width) || !Number.isSafeInteger(meta.height)
      || meta.width < MIN_INPUT_EDGE || meta.height < MIN_INPUT_EDGE
      || meta.width > MAX_INPUT_EDGE || meta.height > MAX_INPUT_EDGE
      || meta.width * meta.height > MAX_INPUT_IMAGE_PIXELS
      || (meta.pages != null && meta.pages !== 1)) {
    const err = new Error('invalid_media_dimensions_or_format');
    err.code = 'decode_failed';
    throw err;
  }

  const sourceFormat = isAvif ? 'avif' : meta.format;

  if (Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0) {
    try {
      const rendered = await sharpImpl(rawBytes, sharpOpts)
        .rotate()
        .flatten({ background })
        .resize(width, height, {
          fit: 'contain',
          background,
          withoutEnlargement: false,
        })
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      return {
        raw: rendered.data,
        info: rendered.info,
        sourceFormat,
        sourceWidth: meta.width,
        sourceHeight: meta.height,
      };
    } catch {
      const err = new Error('decode_failed');
      err.code = 'decode_failed';
      throw err;
    }
  }

  return {
    bytes: rawBytes,
    sourceFormat,
    sourceWidth: meta.width,
    sourceHeight: meta.height,
  };
}

async function readBoundedImageBody(response, maxBytes = MAX_INPUT_IMAGE_BYTES, signal = null) {
  const headerLen = Number(response.headers?.get?.('content-length'));
  if (Number.isFinite(headerLen) && headerLen > maxBytes) {
    void response.body?.cancel?.().catch(() => {});
    const err = new Error('oversize');
    err.code = 'oversize';
    throw err;
  }
  if (!response.body?.getReader) {
    const buf = Buffer.from(await response.arrayBuffer());
    if (buf.length < 32) {
      const err = new Error('invalid_media');
      err.code = 'invalid_media';
      throw err;
    }
    if (buf.length > maxBytes) {
      const err = new Error('oversize');
      err.code = 'oversize';
      throw err;
    }
    return buf;
  }
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal?.addEventListener?.('abort', cancel, { once: true });
  const chunks = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      size += value.byteLength;
      if (size > maxBytes) {
        const err = new Error('oversize');
        err.code = 'oversize';
        throw err;
      }
      chunks.push(Buffer.from(value));
    }
    if (size < 32) {
      const err = new Error('invalid_media');
      err.code = 'invalid_media';
      throw err;
    }
    return Buffer.concat(chunks, size);
  } finally {
    signal?.removeEventListener?.('abort', cancel);
    cancel();
    reader.releaseLock();
  }
}

async function resolveItemRawBytes(item, {
  resolvePhoto = null,
  authorizePhoto = null,
  allowlist = HOST_ALLOWLIST,
  fetchImpl = globalThis.fetch,
  photoCache = null,
  signal = null,
  fetchTimeoutMs = DEFAULT_FETCH_TIMEOUT_MS,
} = {}) {
  // 1. Pre-supplied in-memory Buffer (synthetic tests / local cache)
  const inlineBuffer = Buffer.isBuffer(item?.bytes)
    ? item.bytes
    : Buffer.isBuffer(item?.jpegBuffer)
      ? item.jpegBuffer
      : Buffer.isBuffer(item?.buffer)
        ? item.buffer
        : null;
  if (inlineBuffer) {
    return { ok: true, bytes: inlineBuffer, cacheHit: false };
  }

  const imageUrl = typeof item?.imageUrl === 'string'
    ? item.imageUrl
    : (typeof item?.image_url === 'string' ? item.image_url : null);
  const cacheKey = imageUrl || (item?.id ? `item:${item.id}` : null);

  // 2. Validate SSRF / authorization BEFORE touching cache or network when a URL is present
  if (imageUrl) {
    const authCheck = validateEditorialImageUrl(imageUrl, { authorizePhoto, allowlist });
    if (!authCheck.ok) {
      return { ok: false, status: authCheck.reason, cacheHit: false };
    }
  } else if (typeof resolvePhoto !== 'function') {
    return { ok: false, status: 'missing', cacheHit: false };
  }

  // 3. Check injectable bounded LRU cache (supports sync Map/LRU or async cache)
  if (photoCache && cacheKey && typeof photoCache.get === 'function') {
    const cached = await Promise.resolve(photoCache.get(cacheKey));
    const cachedBuf = Buffer.isBuffer(cached)
      ? cached
      : Buffer.isBuffer(cached?.bytes)
        ? cached.bytes
        : null;
    if (cachedBuf && cachedBuf.length >= 32) {
      return { ok: true, bytes: cachedBuf, cacheHit: true };
    }
  }

  if (signal?.aborted) {
    return { ok: false, status: 'deadline', cacheHit: false };
  }

  // 4. Injectable trusted photo resolver (e.g. request-scoped cloud cache)
  if (typeof resolvePhoto === 'function') {
    try {
      const resolved = await Promise.resolve(resolvePhoto(item, signal));
      const buf = Buffer.isBuffer(resolved)
        ? resolved
        : Buffer.isBuffer(resolved?.bytes)
          ? resolved.bytes
          : null;
      if (!buf || buf.length < 32) {
        return { ok: false, status: 'source_missing', cacheHit: false };
      }
      if (buf.length > MAX_INPUT_IMAGE_BYTES) {
        return { ok: false, status: 'oversize', cacheHit: false };
      }
      if (photoCache && cacheKey && typeof photoCache.set === 'function') {
        await Promise.resolve(photoCache.set(cacheKey, buf));
      }
      return { ok: true, bytes: buf, cacheHit: false };
    } catch (err) {
      const code = err?.mediaFailure || err?.code || 'fetch_failed';
      return { ok: false, status: code, cacheHit: false };
    }
  }

  // 5. Direct SSRF-validated HTTPS fetch
  if (!imageUrl || typeof fetchImpl !== 'function') {
    return { ok: false, status: 'missing', cacheHit: false };
  }
  const fetchController = new AbortController();
  const timer = setTimeout(() => fetchController.abort(), Math.max(300, Math.min(15_000, fetchTimeoutMs)));
  const fetchSignal = signal ? AbortSignal.any([signal, fetchController.signal]) : fetchController.signal;
  try {
    const response = await fetchImpl(imageUrl, {
      method: 'GET',
      redirect: 'error',
      signal: fetchSignal,
      headers: {
        'User-Agent': 'AIWardrobe-Stylist6-EditorialBoard/1.0',
        Accept: 'image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8',
      },
    });
    if (!response?.ok) {
      void response?.body?.cancel?.().catch(() => {});
      const code = Number(response?.status);
      const status = [401, 403, 404, 429].includes(code)
        ? `http_${code}`
        : code >= 500 ? 'http_5xx' : 'fetch_failed';
      return { ok: false, status, cacheHit: false };
    }
    const raw = await readBoundedImageBody(response, MAX_INPUT_IMAGE_BYTES, fetchSignal);
    if (photoCache && cacheKey && typeof photoCache.set === 'function') {
      await Promise.resolve(photoCache.set(cacheKey, raw));
    }
    return { ok: true, bytes: raw, cacheHit: false };
  } catch (err) {
    if (signal?.aborted) return { ok: false, status: 'deadline', cacheHit: false };
    return { ok: false, status: err?.code || 'fetch_failed', cacheHit: false };
  } finally {
    clearTimeout(timer);
  }
}

function mapItemsToBoardSlots(items) {
  const list = Array.isArray(items) ? items.filter(Boolean) : [];
  const bySlot = { outerwear: null, top: null, bottom: null, shoes: null };
  const slotLabels = { outerwear: 'OUTERWEAR', top: 'TOP', bottom: 'BOTTOM', shoes: 'SHOES' };

  for (const item of list) {
    const rawSlot = String(item.slot || item.macroCategory || '').toLowerCase().trim();
    if (CANONICAL_SLOTS.includes(rawSlot) && !bySlot[rawSlot]) {
      bySlot[rawSlot] = item;
    } else if (rawSlot === 'dress' && !bySlot.outerwear) {
      bySlot.outerwear = item;
      slotLabels.outerwear = 'DRESS';
    } else if ((rawSlot === 'accessory' || rawSlot === 'bag') && !bySlot.outerwear) {
      bySlot.outerwear = item;
      slotLabels.outerwear = rawSlot === 'bag' ? 'BAG' : 'ACCESSORY';
    }
  }
  return { bySlot, slotLabels };
}

async function encodeBoundedJpeg(sharpPipeline, maxBytes, qualities = [82, 72, 60]) {
  let lastBuf = null;
  for (const quality of qualities) {
    const buf = await sharpPipeline.clone().jpeg({ quality, progressive: false, mozjpeg: false }).toBuffer();
    lastBuf = buf;
    if (buf.length <= maxBytes) return { ok: true, buffer: buf, quality };
  }
  return { ok: false, buffer: lastBuf, quality: qualities.at(-1) };
}

// Proportion-aware studio zones for 1024x1024 Look Board
const LOOK_BOARD_ZONES = Object.freeze({
  outerwear: Object.freeze({
    cardX: 36, cardY: 104, cardW: 336, cardH: 864,
    imgX: 52, imgY: 146, imgW: 304, imgH: 734,
    labelX: 52, labelY: 130,
    footerY1: 912, footerY2: 938,
  }),
  top: Object.freeze({
    cardX: 392, cardY: 104, cardW: 280, cardH: 474,
    imgX: 408, imgY: 146, imgW: 248, imgH: 348,
    labelX: 408, labelY: 130,
    footerY1: 524, footerY2: 550,
  }),
  shoes: Object.freeze({
    cardX: 392, cardY: 598, cardW: 280, cardH: 370,
    imgX: 408, imgY: 640, imgW: 248, imgH: 244,
    labelX: 408, labelY: 624,
    footerY1: 912, footerY2: 938,
  }),
  bottom: Object.freeze({
    cardX: 692, cardY: 104, cardW: 296, cardH: 864,
    imgX: 708, imgY: 146, imgW: 264, imgH: 734,
    labelX: 708, labelY: 130,
    footerY1: 912, footerY2: 938,
  }),
});

/**
 * Render a clean 1024x1024 proportion-aware 4-item studio flat-lay board on
 * warm neutral `#F6F5F2` backdrop.
 */
export async function renderEditorialLookBoard(items, options = {}) {
  const started = performance.now();
  const {
    title = 'STUDIO LOOK BOARD',
    subtitle = '',
    lookToken = null,
    anonymizeLabels = false,
    includePriceTags = true,
    requireAllPhotos = false,
    maxOutputBytes = DEFAULT_LOOK_BOARD_MAX_BYTES,
    deadlineMs = 6000,
    signal = null,
    photoCache = null,
    resolvePhoto = null,
    authorizePhoto = null,
    allowlist = HOST_ALLOWLIST,
    fetchImpl = globalThis.fetch,
    sharpImpl = sharp,
  } = options;

  const list = Array.isArray(items) ? items : (Array.isArray(items?.items) ? items.items : []);
  if (!list.length) {
    throw new Error('invalid_board_items');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(300, Math.min(20_000, deadlineMs)));
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;

  const { bySlot, slotLabels } = mapItemsToBoardSlots(list);
  const slotStatus = {};
  let cacheHits = 0;
  let cacheMisses = 0;

  try {
    const resolvedEntries = await Promise.all(CANONICAL_SLOTS.map(async (slot) => {
      const item = bySlot[slot];
      const zone = LOOK_BOARD_ZONES[slot];
      if (!item) {
        slotStatus[slot] = { status: 'absent', cacheHit: false };
        return { slot, item: null, zone, rawComposite: null };
      }

      const fetched = await resolveItemRawBytes(item, {
        resolvePhoto,
        authorizePhoto,
        allowlist,
        fetchImpl,
        photoCache,
        signal: combinedSignal,
        fetchTimeoutMs: Math.min(DEFAULT_FETCH_TIMEOUT_MS, deadlineMs),
      });

      if (fetched.cacheHit) cacheHits++;
      else cacheMisses++;

      if (!fetched.ok) {
        slotStatus[slot] = { status: fetched.status, cacheHit: fetched.cacheHit };
        return { slot, item, zone, rawComposite: null };
      }

      try {
        const decoded = await decodeNormalizedPhotoBytes(fetched.bytes, {
          width: zone.imgW,
          height: zone.imgH,
          background: STUDIO_CARD_HEX,
          sharpImpl,
        });
        slotStatus[slot] = {
          status: 'rendered',
          cacheHit: fetched.cacheHit,
          sourceFormat: decoded.sourceFormat,
          sourceWidth: decoded.sourceWidth,
          sourceHeight: decoded.sourceHeight,
        };
        return {
          slot,
          item,
          zone,
          rawComposite: {
            input: decoded.raw,
            raw: {
              width: decoded.info.width,
              height: decoded.info.height,
              channels: decoded.info.channels,
            },
            left: zone.imgX,
            top: zone.imgY,
          },
        };
      } catch (err) {
        slotStatus[slot] = { status: err?.code || 'decode_failed', cacheHit: fetched.cacheHit };
        return { slot, item, zone, rawComposite: null };
      }
    }));

    const presentSlots = CANONICAL_SLOTS.filter((slot) => bySlot[slot] !== null);
    const failedSlots = presentSlots.filter((slot) => slotStatus[slot]?.status !== 'rendered');
    const allSlotsRendered = failedSlots.length === 0 && presentSlots.length > 0;

    if (requireAllPhotos && !allSlotsRendered) {
      const firstFail = failedSlots[0];
      return {
        ok: false,
        error: slotStatus[firstFail]?.status || 'source_missing',
        jpegBuffer: null,
        bytes: null,
        width: LOOK_BOARD_WIDTH,
        height: LOOK_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        slotStatus,
        allSlotsRendered: false,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    // Build crisp studio card frames + minimal typography in ONE SVG layer.
    // Order: backdrop -> SVG cards (underneath) -> raw item images -> SVG labels/borders (above)
    // Even faster: paint card fill + hairline + labels in one SVG under? No, SVG placed AFTER
    // images can paint borders & text without obscuring images, while cards themselves are
    // white rectangles drawn in a single base SVG UNDER the images! Two tiny SVGs total.
    const cardRects = CANONICAL_SLOTS.map((slot) => {
      const z = LOOK_BOARD_ZONES[slot];
      const rendered = slotStatus[slot]?.status === 'rendered';
      const fill = rendered ? STUDIO_CARD_HEX : STUDIO_PLACEHOLDER_HEX;
      return `<rect x="${z.cardX}" y="${z.cardY}" width="${z.cardW}" height="${z.cardH}" rx="10" fill="${fill}" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`;
    });

    const headerTitle = anonymizeLabels
      ? escapeXml(sanitizeLabelText(lookToken || 'LOOK BOARD', 28).toUpperCase())
      : escapeXml(sanitizeLabelText(title, 52).toUpperCase() || 'STUDIO LOOK BOARD');
    const headerSub = anonymizeLabels
      ? 'PROPORTION-SCALED FLAT-LAY'
      : escapeXml(sanitizeLabelText(subtitle, 84).toUpperCase());

    const baseSvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${LOOK_BOARD_WIDTH}" height="${LOOK_BOARD_HEIGHT}">`
      + `<rect width="${LOOK_BOARD_WIDTH}" height="${LOOK_BOARD_HEIGHT}" fill="${STUDIO_BACKDROP_HEX}"/>`
      + `<text x="36" y="${headerSub ? 42 : 54}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="20" font-weight="700" letter-spacing="1.6">${headerTitle}</text>`
      + (headerSub ? `<text x="36" y="66" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="600" letter-spacing="1.2">${headerSub}</text>` : '')
      + `<line x1="36" y1="84" x2="988" y2="84" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`
      + cardRects.join('')
      + `</svg>`,
    );

    const textLines = [];
    for (const entry of resolvedEntries) {
      const { slot, item, zone } = entry;
      const status = slotStatus[slot]?.status;
      const roleLabel = escapeXml(slotLabels[slot]);
      textLines.push(
        `<text x="${zone.labelX}" y="${zone.labelY}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="700" letter-spacing="1.8">${roleLabel}</text>`,
      );

      if (status !== 'rendered') {
        const msg = status === 'absent' ? 'NO OUTER LAYER' : 'PHOTO UNAVAILABLE';
        const cx = Math.round(zone.imgX + zone.imgW / 2);
        const cy = Math.round(zone.imgY + zone.imgH / 2);
        textLines.push(
          `<text x="${cx}" y="${cy}" text-anchor="middle" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="13" font-weight="600" letter-spacing="1.2">${msg}</text>`,
        );
      }

      if (item && !anonymizeLabels) {
        const brand = sanitizeLabelText(item.brand || item.subtype || '', 26);
        const price = includePriceTags ? formatPriceTag(item) : '';
        if (brand) {
          textLines.push(
            `<text x="${zone.labelX}" y="${zone.footerY1}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="14" font-weight="700">${escapeXml(brand)}</text>`,
          );
        }
        if (price) {
          textLines.push(
            `<text x="${zone.labelX}" y="${zone.footerY2}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="13" font-weight="500">${escapeXml(price)}</text>`,
          );
        }
      } else if (item && anonymizeLabels && item.productToken) {
        textLines.push(
          `<text x="${zone.labelX}" y="${zone.footerY1}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="13" font-weight="700">${escapeXml(sanitizeLabelText(item.productToken, 12))}</text>`,
        );
      }
    }

    // Subtle studio footer
    const footerNote = anonymizeLabels
      ? 'REAL SOURCE GARMENT PHOTOGRAPHS'
      : 'STUDIO FLAT-LAY · PROPORTION-SCALED REAL PRODUCT PHOTOGRAPHS';
    textLines.push(
      `<text x="36" y="1002" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600" letter-spacing="1.4">${footerNote}</text>`,
    );

    const overlaySvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${LOOK_BOARD_WIDTH}" height="${LOOK_BOARD_HEIGHT}">${textLines.join('')}</svg>`,
    );

    const composites = [
      ...resolvedEntries.filter((e) => e.rawComposite).map((e) => e.rawComposite),
      { input: overlaySvg, left: 0, top: 0 },
    ];

    const pipeline = sharpImpl(baseSvg).composite(composites);
    const encoded = await encodeBoundedJpeg(pipeline, maxOutputBytes, [82, 72, 60, 50]);
    if (!encoded.ok) {
      return {
        ok: false,
        error: 'oversized',
        jpegBuffer: null,
        bytes: null,
        width: LOOK_BOARD_WIDTH,
        height: LOOK_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        slotStatus,
        allSlotsRendered,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const sha256 = createHash('sha256').update(encoded.buffer).digest('hex');
    return {
      ok: true,
      jpegBuffer: encoded.buffer,
      bytes: encoded.buffer,
      width: LOOK_BOARD_WIDTH,
      height: LOOK_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256,
      slotStatus,
      allSlotsRendered,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}

function computePortfolioCardLayout(lookCount) {
  if (lookCount <= 3) {
    const cols = 3;
    const marginX = 24;
    const gapX = 18;
    const colW = 572;
    const cardY = 66;
    const cardH = 912;
    return Array.from({ length: lookCount }, (_, idx) => {
      const cardX = marginX + idx * (colW + gapX);
      const innerX = cardX + 12;
      const innerY = cardY + 50;
      return {
        cardX, cardY, cardW: colW, cardH,
        headerY: cardY + 32,
        zones: {
          outerwear: { x: innerX, y: innerY, w: 186, h: 774, imgX: innerX + 6, imgY: innerY + 24, imgW: 174, imgH: 736 },
          top: { x: innerX + 196, y: innerY, w: 166, h: 428, imgX: innerX + 202, imgY: innerY + 24, imgW: 154, imgH: 392 },
          shoes: { x: innerX + 196, y: innerY + 440, w: 166, h: 334, imgX: innerX + 202, imgY: innerY + 464, imgW: 154, imgH: 298 },
          bottom: { x: innerX + 372, y: innerY, w: 176, h: 774, imgX: innerX + 378, imgY: innerY + 24, imgW: 164, imgH: 736 },
        },
        footerY: cardY + cardH - 22,
        compact: false,
      };
    });
  }

  const cols = lookCount <= 6 ? 3 : 4;
  const marginX = cols === 3 ? 24 : 20;
  const gapX = cols === 3 ? 18 : 12;
  const colW = cols === 3 ? 572 : 431;
  const startY = 66;
  const rowH = 448;
  const rowGap = 16;

  return Array.from({ length: lookCount }, (_, idx) => {
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    const cardX = marginX + col * (colW + gapX);
    const cardY = startY + row * (rowH + rowGap);
    const innerX = cardX + 10;
    const innerY = cardY + 36;
    const outerW = cols === 3 ? 186 : 138;
    const midW = cols === 3 ? 166 : 123;
    const bottomW = cols === 3 ? 176 : 134;
    const innerGap = cols === 3 ? 10 : 8;
    return {
      cardX, cardY, cardW: colW, cardH: rowH,
      headerY: cardY + 24,
      zones: {
        outerwear: { x: innerX, y: innerY, w: outerW, h: 398, imgX: innerX + 5, imgY: innerY + 20, imgW: outerW - 10, imgH: 372 },
        top: { x: innerX + outerW + innerGap, y: innerY, w: midW, h: 214, imgX: innerX + outerW + innerGap + 5, imgY: innerY + 20, imgW: midW - 10, imgH: 188 },
        shoes: { x: innerX + outerW + innerGap, y: innerY + 222, w: midW, h: 176, imgX: innerX + outerW + innerGap + 5, imgY: innerY + 242, imgW: midW - 10, imgH: 150 },
        bottom: { x: innerX + outerW + midW + innerGap * 2, y: innerY, w: bottomW, h: 398, imgX: innerX + outerW + midW + innerGap * 2 + 5, imgY: innerY + 20, imgW: bottomW - 10, imgH: 372 },
      },
      footerY: cardY + rowH - 8,
      compact: true,
    };
  });
}

/**
 * Render a 1800x1000 high-resolution side-by-side portfolio comparison sheet
 * supporting 1–8 looks (3-column hero layout for 1–3 looks, 2x3 for 4–6, 2x4 for 7–8).
 */
export async function renderPortfolioComparisonSheet(looks, options = {}) {
  const started = performance.now();
  const {
    title = 'PORTFOLIO COMPARISON',
    subtitle = '',
    anonymizeLabels = false,
    requireAllPhotos = false,
    maxOutputBytes = DEFAULT_PORTFOLIO_MAX_BYTES,
    deadlineMs = 6000,
    signal = null,
    photoCache = null,
    resolvePhoto = null,
    authorizePhoto = null,
    allowlist = HOST_ALLOWLIST,
    fetchImpl = globalThis.fetch,
    sharpImpl = sharp,
    onStatus = null,
  } = options;

  const note = (status) => { try { onStatus?.(status); } catch { /* noop */ } };

  if (!Array.isArray(looks) || looks.length < 1 || looks.length > 8) {
    note('invalid_input');
    throw new Error('invalid_portfolio_looks');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(300, Math.min(20_000, deadlineMs)));
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;

  const layouts = computePortfolioCardLayout(looks.length);
  const lookTokens = looks.map((look, i) =>
    typeof look?.token === 'string' && /^L\d{2}$/.test(look.token)
      ? look.token
      : `L${String(i).padStart(2, '0')}`,
  );
  const slotStatuses = looks.map(() => ({}));
  let cacheHits = 0;
  let cacheMisses = 0;

  try {
    const lookPlans = looks.map((look, idx) => {
      const items = Array.isArray(look?.items) ? look.items : [];
      const { bySlot, slotLabels } = mapItemsToBoardSlots(items);
      return { look, idx, token: lookTokens[idx], layout: layouts[idx], bySlot, slotLabels };
    });

    const cellResults = await Promise.all(
      lookPlans.flatMap((plan) =>
        CANONICAL_SLOTS.map(async (slot) => {
          const item = plan.bySlot[slot];
          const zone = plan.layout.zones[slot];
          if (!item) {
            slotStatuses[plan.idx][slot] = { status: 'absent', cacheHit: false };
            return { planIdx: plan.idx, slot, item: null, zone, rawComposite: null };
          }
          const fetched = await resolveItemRawBytes(item, {
            resolvePhoto,
            authorizePhoto,
            allowlist,
            fetchImpl,
            photoCache,
            signal: combinedSignal,
            fetchTimeoutMs: Math.min(DEFAULT_FETCH_TIMEOUT_MS, deadlineMs),
          });
          if (fetched.cacheHit) cacheHits++;
          else cacheMisses++;

          if (!fetched.ok) {
            slotStatuses[plan.idx][slot] = { status: fetched.status, cacheHit: fetched.cacheHit };
            return { planIdx: plan.idx, slot, item, zone, rawComposite: null };
          }

          try {
            const decoded = await decodeNormalizedPhotoBytes(fetched.bytes, {
              width: zone.imgW,
              height: zone.imgH,
              background: STUDIO_CARD_HEX,
              sharpImpl,
            });
            slotStatuses[plan.idx][slot] = {
              status: 'rendered',
              cacheHit: fetched.cacheHit,
              sourceFormat: decoded.sourceFormat,
            };
            return {
              planIdx: plan.idx,
              slot,
              item,
              zone,
              rawComposite: {
                input: decoded.raw,
                raw: {
                  width: decoded.info.width,
                  height: decoded.info.height,
                  channels: decoded.info.channels,
                },
                left: zone.imgX,
                top: zone.imgY,
              },
            };
          } catch (err) {
            slotStatuses[plan.idx][slot] = { status: err?.code || 'decode_failed', cacheHit: fetched.cacheHit };
            return { planIdx: plan.idx, slot, item, zone, rawComposite: null };
          }
        }),
      ),
    );

    if (combinedSignal.aborted) {
      note('deadline');
      return {
        ok: false,
        error: 'deadline',
        jpegBuffer: null,
        bytes: null,
        width: PORTFOLIO_SHEET_WIDTH,
        height: PORTFOLIO_SHEET_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        lookTokens,
        slotStatuses,
        allLooksRendered: false,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    let firstFailureStatus = null;
    for (let i = 0; i < lookPlans.length; i++) {
      for (const slot of CANONICAL_SLOTS) {
        if (lookPlans[i].bySlot[slot] && slotStatuses[i][slot]?.status !== 'rendered') {
          if (!firstFailureStatus) firstFailureStatus = slotStatuses[i][slot]?.status || 'source_missing';
        }
      }
    }
    const allLooksRendered = firstFailureStatus === null;

    if (requireAllPhotos && !allLooksRendered) {
      const statusMap = firstFailureStatus === 'host_not_allowlisted' || firstFailureStatus === 'ssrf_blocked'
        ? 'unauthorized_or_missing'
        : firstFailureStatus === 'deadline'
          ? 'deadline'
          : firstFailureStatus === 'source_missing' || firstFailureStatus === 'missing'
            ? 'source_missing'
            : 'photo_processing';
      note(statusMap);
      return {
        ok: false,
        error: statusMap,
        jpegBuffer: null,
        bytes: null,
        width: PORTFOLIO_SHEET_WIDTH,
        height: PORTFOLIO_SHEET_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        lookTokens,
        slotStatuses,
        allLooksRendered: false,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const headerTitle = anonymizeLabels
      ? 'PORTFOLIO COMPARISON SHEET'
      : escapeXml(sanitizeLabelText(title, 48).toUpperCase() || 'PORTFOLIO COMPARISON');
    const headerRight = anonymizeLabels
      ? `${looks.length} VERIFIED LOOKS · OUTERWEAR / TOP / BOTTOM / SHOES`
      : escapeXml(sanitizeLabelText(subtitle, 60).toUpperCase() || `${looks.length} LOOKS · PROPORTION-SCALED`);

    const baseShapes = [];
    const labelShapes = [];

    for (const plan of lookPlans) {
      const { layout, token, look, slotLabels } = plan;
      baseShapes.push(
        `<rect x="${layout.cardX}" y="${layout.cardY}" width="${layout.cardW}" height="${layout.cardH}" rx="10" fill="${STUDIO_CARD_HEX}" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`,
      );
      const lookHeading = anonymizeLabels
        ? escapeXml(token)
        : escapeXml(sanitizeLabelText(`${token}${look?.title ? ' · ' + look.title : ''}`, 36));
      labelShapes.push(
        `<text x="${layout.cardX + 14}" y="${layout.headerY}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="${layout.compact ? 15 : 18}" font-weight="700" letter-spacing="1.2">${lookHeading}</text>`,
      );

      for (const slot of CANONICAL_SLOTS) {
        const z = layout.zones[slot];
        const st = slotStatuses[plan.idx][slot]?.status;
        const fill = st === 'rendered' ? '#FAF9F7' : STUDIO_PLACEHOLDER_HEX;
        baseShapes.push(
          `<rect x="${z.x}" y="${z.y}" width="${z.w}" height="${z.h}" rx="6" fill="${fill}" stroke="#EAE6DF" stroke-width="1"/>`,
        );
        labelShapes.push(
          `<text x="${z.x + 8}" y="${z.y + 15}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="10" font-weight="700" letter-spacing="1.2">${escapeXml(slotLabels[slot])}</text>`,
        );
        if (st !== 'rendered') {
          const msg = st === 'absent' ? 'NONE' : 'UNAVAILABLE';
          labelShapes.push(
            `<text x="${Math.round(z.x + z.w / 2)}" y="${Math.round(z.y + z.h / 2)}" text-anchor="middle" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600">${msg}</text>`,
          );
        }
      }

      if (!layout.compact && !anonymizeLabels) {
        const score = Number.isFinite(Number(look?.editorialScore))
          ? `SCORE ${Number(look.editorialScore).toFixed(1)}/10`
          : '';
        if (score) {
          labelShapes.push(
            `<text x="${layout.cardX + layout.cardW - 14}" y="${layout.headerY}" text-anchor="end" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="13" font-weight="700">${escapeXml(score)}</text>`,
          );
        }
      }
    }

    const baseSvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${PORTFOLIO_SHEET_WIDTH}" height="${PORTFOLIO_SHEET_HEIGHT}">`
      + `<rect width="${PORTFOLIO_SHEET_WIDTH}" height="${PORTFOLIO_SHEET_HEIGHT}" fill="${STUDIO_BACKDROP_HEX}"/>`
      + `<text x="24" y="38" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="20" font-weight="700" letter-spacing="1.8">${headerTitle}</text>`
      + `<text x="1776" y="38" text-anchor="end" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="600" letter-spacing="1.2">${headerRight}</text>`
      + `<line x1="24" y1="54" x2="1776" y2="54" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`
      + baseShapes.join('')
      + `</svg>`,
    );

    const overlaySvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${PORTFOLIO_SHEET_WIDTH}" height="${PORTFOLIO_SHEET_HEIGHT}">${labelShapes.join('')}</svg>`,
    );

    const composites = [
      ...cellResults.filter((c) => c.rawComposite).map((c) => c.rawComposite),
      { input: overlaySvg, left: 0, top: 0 },
    ];

    const pipeline = sharpImpl(baseSvg).composite(composites);
    const encoded = await encodeBoundedJpeg(pipeline, maxOutputBytes, [80, 70, 58, 48]);
    if (!encoded.ok) {
      note('oversized');
      return {
        ok: false,
        error: 'oversized',
        jpegBuffer: null,
        bytes: null,
        width: PORTFOLIO_SHEET_WIDTH,
        height: PORTFOLIO_SHEET_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        lookTokens,
        slotStatuses,
        allLooksRendered,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const sha256 = createHash('sha256').update(encoded.buffer).digest('hex');
    note('ready');
    return {
      ok: true,
      jpegBuffer: encoded.buffer,
      bytes: encoded.buffer,
      width: PORTFOLIO_SHEET_WIDTH,
      height: PORTFOLIO_SHEET_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256,
      lookTokens,
      slotStatuses,
      allLooksRendered,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } catch (err) {
    note(combinedSignal.aborted ? 'deadline' : 'photo_processing');
    return {
      ok: false,
      error: combinedSignal.aborted ? 'deadline' : (err?.code || 'photo_processing'),
      jpegBuffer: null,
      bytes: null,
      width: PORTFOLIO_SHEET_WIDTH,
      height: PORTFOLIO_SHEET_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256: null,
      lookTokens,
      slotStatuses,
      allLooksRendered: false,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}

function unwrapGarmentEntry(entry) {
  if (!entry || typeof entry !== 'object') return null;
  if (entry.item && typeof entry.item === 'object') return entry.item;
  return entry;
}

const SHOWROOM_RACKS = Object.freeze([
  Object.freeze({ slot: 'top', aliases: ['top', 'tops'], prefix: 'T', label: 'TOP' }),
  Object.freeze({ slot: 'outerwear', aliases: ['outerwear', 'layer', 'layers'], prefix: 'L', label: 'OUTERWEAR' }),
  Object.freeze({ slot: 'bottom', aliases: ['bottom', 'bottoms'], prefix: 'B', label: 'BOTTOM' }),
  Object.freeze({ slot: 'shoes', aliases: ['shoes', 'footwear'], prefix: 'S', label: 'SHOES' }),
]);

/**
 * Render a 1600x1600 4-rack Visual Showroom Board (TOP T1..T8, OUTERWEAR L1..L8,
 * BOTTOM B1..B8, SHOES S1..S8 — up to 32 real photographed garments on warm
 * `#F6F5F2` studio canvas) so a Vision-Language Stylist can see real fabric,
 * drape, shade, silhouette, shoe profile, and photo cleanliness BEFORE composing.
 *
 * Security & honesty guarantees:
 *   - Reuses `resolveItemRawBytes` (SSRF allowlist + bounded LRU `photoCache`)
 *     and `decodeNormalizedPhotoBytes` (AVIF/WebP/PNG/JPEG + byte/pixel caps).
 *   - Skips unreachable/broken/blocked photos honestly and ONLY emits tokens
 *     for garments whose photographs actually decoded and rendered on the board.
 */
export async function renderVisualShowroomBoard(slotPools, options = {}) {
  const started = performance.now();
  const {
    title = 'PRIVATE VISUAL SHOWROOM',
    subtitle = 'REAL GARMENT PHOTOGRAPHS · CHOOSE EXACT TOKENS (T / L / B / S)',
    maxPerSlot = 8,
    minPerSlot = 0,
    requireMinPerSlot = false,
    anonymizeLabels = false,
    includePriceTags = true,
    maxOutputBytes = DEFAULT_SHOWROOM_MAX_BYTES,
    deadlineMs = 9000,
    signal = null,
    photoCache = null,
    resolvePhoto = null,
    authorizePhoto = null,
    allowlist = HOST_ALLOWLIST,
    fetchImpl = globalThis.fetch,
    sharpImpl = sharp,
  } = options;

  const perSlotCap = Math.max(1, Math.min(8, Math.floor(Number(maxPerSlot) || 8)));
  const poolsObj = slotPools && typeof slotPools === 'object' ? slotPools : {};

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(400, Math.min(25_000, deadlineMs)));
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;

  const rackCardW = 185;
  const rackCardH = 336;
  const rackGapX = 10;
  const rackStartX = 25;
  const rackStartY = 74;
  const rackH = 368;
  const rackGapY = 10;
  const imgPadX = 8;
  const imgOffsetY = 36;
  const imgW = 169;
  const imgH = 254;

  const requestedByRack = SHOWROOM_RACKS.map((rack) => {
    let rawList = [];
    for (const alias of rack.aliases) {
      if (Array.isArray(poolsObj[alias]) && poolsObj[alias].length > 0) {
        rawList = poolsObj[alias];
        break;
      }
    }
    const items = [];
    const seenIds = new Set();
    for (const entry of rawList) {
      const item = unwrapGarmentEntry(entry);
      if (!item) continue;
      const dedupeKey = item.id != null ? String(item.id) : null;
      if (dedupeKey && seenIds.has(dedupeKey)) continue;
      if (dedupeKey) seenIds.add(dedupeKey);
      items.push(item);
      if (items.length >= perSlotCap) break;
    }
    return { rack, items };
  });

  const totalRequested = requestedByRack.reduce((sum, r) => sum + r.items.length, 0);
  const tokens = {};
  const tokensBySlot = { top: [], outerwear: [], bottom: [], shoes: [] };
  const slotCounts = { top: 0, outerwear: 0, bottom: 0, shoes: 0 };
  const skipped = [];
  let cacheHits = 0;
  let cacheMisses = 0;

  if (totalRequested === 0) {
    clearTimeout(timer);
    controller.abort();
    return {
      ok: false,
      error: 'empty_showroom_pool',
      jpegBuffer: null,
      bytes: null,
      width: SHOWROOM_BOARD_WIDTH,
      height: SHOWROOM_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256: null,
      tokens,
      tokenMap: new Map(),
      tokensBySlot,
      slotCounts,
      skipped,
      renderedCount: 0,
      totalRequested: 0,
      renderMs: Math.round(performance.now() - started),
      cacheHits: 0,
      cacheMisses: 0,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  }

  try {
    const rackResults = await Promise.all(
      requestedByRack.map(async ({ rack, items }, rackIdx) => {
        const fetchedEntries = await Promise.all(
          items.map(async (item, itemIdx) => {
            const fetched = await resolveItemRawBytes(item, {
              resolvePhoto,
              authorizePhoto,
              allowlist,
              fetchImpl,
              photoCache,
              signal: combinedSignal,
              fetchTimeoutMs: Math.min(DEFAULT_FETCH_TIMEOUT_MS, deadlineMs),
            });
            if (fetched.cacheHit) cacheHits++;
            else cacheMisses++;

            if (!fetched.ok) {
              return { ok: false, item, itemIdx, status: fetched.status };
            }
            try {
              const decoded = await decodeNormalizedPhotoBytes(fetched.bytes, {
                width: imgW,
                height: imgH,
                background: STUDIO_CARD_HEX,
                sharpImpl,
              });
              return { ok: true, item, itemIdx, decoded };
            } catch (err) {
              return { ok: false, item, itemIdx, status: err?.code || 'decode_failed' };
            }
          }),
        );
        return { rack, rackIdx, fetchedEntries };
      }),
    );

    if (combinedSignal.aborted) {
      return {
        ok: false,
        error: 'deadline',
        jpegBuffer: null,
        bytes: null,
        width: SHOWROOM_BOARD_WIDTH,
        height: SHOWROOM_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        tokens,
        tokenMap: new Map(),
        tokensBySlot,
        slotCounts,
        skipped,
        renderedCount: 0,
        totalRequested,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const baseShapes = [];
    const overlayShapes = [];
    const rawComposites = [];
    let renderedCount = 0;

    for (const { rack, rackIdx, fetchedEntries } of rackResults) {
      const rackY = rackStartY + rackIdx * (rackH + rackGapY);
      const cardY = rackY + 26;
      const renderedInRack = [];

      for (const entry of fetchedEntries) {
        if (!entry.ok) {
          skipped.push({
            slot: rack.slot,
            id: entry.item?.id != null ? String(entry.item.id) : null,
            status: entry.status,
          });
          continue;
        }
        renderedInRack.push(entry);
      }

      const rackCount = renderedInRack.length;
      slotCounts[rack.slot] = rackCount;

      // Subtle rack shelf strip + header label
      baseShapes.push(
        `<rect x="18" y="${rackY}" width="1564" height="${rackH}" rx="10" fill="#F0EEE9" stroke="${STUDIO_BORDER_HEX}" stroke-width="1"/>`,
      );
      const rangeText = rackCount > 0 ? `${rack.prefix}1–${rack.prefix}${rackCount}` : 'NO PHOTOS';
      overlayShapes.push(
        `<text x="28" y="${rackY + 18}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="13" font-weight="800" letter-spacing="1.6">${escapeXml(rack.label)} · ${escapeXml(rangeText)}</text>`,
      );
      overlayShapes.push(
        `<text x="1572" y="${rackY + 18}" text-anchor="end" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600" letter-spacing="1.0">${rackCount} OF ${fetchedEntries.length} PHOTOGRAPHED</text>`,
      );

      for (let col = 0; col < renderedInRack.length; col++) {
        const entry = renderedInRack[col];
        const token = `${rack.prefix}${col + 1}`;
        tokens[token] = entry.item;
        tokensBySlot[rack.slot].push(token);
        renderedCount++;

        const cardX = rackStartX + col * (rackCardW + rackGapX);
        const imgX = cardX + imgPadX;
        const imgY = cardY + imgOffsetY;

        baseShapes.push(
          `<rect x="${cardX}" y="${cardY}" width="${rackCardW}" height="${rackCardH}" rx="8" fill="${STUDIO_CARD_HEX}" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.2"/>`,
        );

        rawComposites.push({
          input: entry.decoded.raw,
          raw: {
            width: entry.decoded.info.width,
            height: entry.decoded.info.height,
            channels: entry.decoded.info.channels,
          },
          left: imgX,
          top: imgY,
        });

        // Bold high-contrast token badge pill (T1..S8)
        overlayShapes.push(
          `<rect x="${cardX + 8}" y="${cardY + 6}" width="44" height="24" rx="5" fill="${STUDIO_INK_HEX}"/>`,
          `<text x="${cardX + 30}" y="${cardY + 23}" text-anchor="middle" fill="#FFFFFF" font-family="Helvetica Neue,Arial,sans-serif" font-size="14" font-weight="800" letter-spacing="0.6">${escapeXml(token)}</text>`,
        );

        if (!anonymizeLabels) {
          const subtypeTag = sanitizeLabelText(String(entry.item?.subtype || '').replace(/[_-]+/g, ' '), 17);
          if (subtypeTag && subtypeTag.toLowerCase() !== 'unknown') {
            overlayShapes.push(
              `<text x="${cardX + 58}" y="${cardY + 22}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="10" font-weight="600">${escapeXml(subtypeTag.toUpperCase())}</text>`,
            );
          }
          const brand = sanitizeLabelText(entry.item?.brand || entry.item?.subtype || '', 20);
          const price = includePriceTags ? formatPriceTag(entry.item) : '';
          if (brand) {
            overlayShapes.push(
              `<text x="${cardX + 8}" y="${cardY + 309}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="700">${escapeXml(brand)}</text>`,
            );
          }
          if (price) {
            overlayShapes.push(
              `<text x="${cardX + 8}" y="${cardY + 326}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600">${escapeXml(price)}</text>`,
            );
          }
        }
      }
    }

    const tokenMap = new Map(Object.entries(tokens));
    const minRequired = Math.max(0, Math.floor(Number(minPerSlot) || 0));
    const failedFloor = requireMinPerSlot && minRequired > 0
      && SHOWROOM_RACKS.some((r) => slotCounts[r.slot] < minRequired);

    if (renderedCount === 0 || failedFloor) {
      return {
        ok: false,
        error: renderedCount === 0 ? 'no_rendered_photos' : 'insufficient_rendered_photos',
        jpegBuffer: null,
        bytes: null,
        width: SHOWROOM_BOARD_WIDTH,
        height: SHOWROOM_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        tokens,
        tokenMap,
        tokensBySlot,
        slotCounts,
        skipped,
        renderedCount,
        totalRequested,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const headerTitle = anonymizeLabels
      ? 'PRIVATE VISUAL SHOWROOM'
      : escapeXml(sanitizeLabelText(title, 54).toUpperCase() || 'PRIVATE VISUAL SHOWROOM');
    const headerSub = anonymizeLabels
      ? 'REAL SOURCE GARMENT PHOTOGRAPHS · TOKENS T / L / B / S'
      : escapeXml(sanitizeLabelText(subtitle, 96).toUpperCase());

    const baseSvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${SHOWROOM_BOARD_WIDTH}" height="${SHOWROOM_BOARD_HEIGHT}">`
      + `<rect width="${SHOWROOM_BOARD_WIDTH}" height="${SHOWROOM_BOARD_HEIGHT}" fill="${STUDIO_BACKDROP_HEX}"/>`
      + `<text x="26" y="32" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="19" font-weight="800" letter-spacing="1.6">${headerTitle}</text>`
      + (headerSub ? `<text x="26" y="52" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600" letter-spacing="1.1">${headerSub}</text>` : '')
      + `<text x="1574" y="38" text-anchor="end" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="700" letter-spacing="1.2">${renderedCount} PHOTOGRAPHED GARMENTS · 4 RACKS</text>`
      + `<line x1="26" y1="64" x2="1574" y2="64" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`
      + baseShapes.join('')
      + `</svg>`,
    );

    overlayShapes.push(
      `<text x="26" y="1592" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="10" font-weight="600" letter-spacing="1.3">STYLIST 6 VISUAL ATELIER · REAL GARMENT PHOTOGRAPHS ONLY · ZERO FABRICATED MEDIA</text>`,
    );

    const overlaySvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${SHOWROOM_BOARD_WIDTH}" height="${SHOWROOM_BOARD_HEIGHT}">${overlayShapes.join('')}</svg>`,
    );

    const composites = [
      ...rawComposites,
      { input: overlaySvg, left: 0, top: 0 },
    ];

    const pipeline = sharpImpl(baseSvg).composite(composites);
    const encoded = await encodeBoundedJpeg(pipeline, maxOutputBytes, [80, 70, 58, 48]);
    if (!encoded.ok) {
      return {
        ok: false,
        error: 'oversized',
        jpegBuffer: null,
        bytes: null,
        width: SHOWROOM_BOARD_WIDTH,
        height: SHOWROOM_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        tokens,
        tokenMap,
        tokensBySlot,
        slotCounts,
        skipped,
        renderedCount,
        totalRequested,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const sha256 = createHash('sha256').update(encoded.buffer).digest('hex');
    return {
      ok: true,
      jpegBuffer: encoded.buffer,
      bytes: encoded.buffer,
      width: SHOWROOM_BOARD_WIDTH,
      height: SHOWROOM_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256,
      tokens,
      tokenMap,
      tokensBySlot,
      slotCounts,
      skipped,
      renderedCount,
      totalRequested,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } catch (err) {
    return {
      ok: false,
      error: combinedSignal.aborted ? 'deadline' : (err?.code || 'photo_processing'),
      jpegBuffer: null,
      bytes: null,
      width: SHOWROOM_BOARD_WIDTH,
      height: SHOWROOM_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256: null,
      tokens,
      tokenMap: new Map(Object.entries(tokens)),
      tokensBySlot,
      slotCounts,
      skipped,
      renderedCount: Object.keys(tokens).length,
      totalRequested,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}

// Proportion-aware left-half zones inside 1400x900 Visual Swap Board
const SWAP_LOCKED_ZONES = Object.freeze({
  outerwear: Object.freeze({
    cardX: 24, cardY: 108, cardW: 202, cardH: 764,
    imgX: 34, imgY: 144, imgW: 182, imgH: 668,
    labelX: 34, labelY: 129,
    footerY1: 836, footerY2: 856,
  }),
  top: Object.freeze({
    cardX: 238, cardY: 108, cardW: 176, cardH: 408,
    imgX: 248, imgY: 144, imgW: 156, imgH: 312,
    labelX: 248, labelY: 129,
    footerY1: 480, footerY2: 500,
  }),
  shoes: Object.freeze({
    cardX: 238, cardY: 528, cardW: 176, cardH: 344,
    imgX: 248, imgY: 564, imgW: 156, imgH: 248,
    labelX: 248, labelY: 549,
    footerY1: 836, footerY2: 856,
  }),
  bottom: Object.freeze({
    cardX: 426, cardY: 108, cardW: 198, cardH: 764,
    imgX: 436, imgY: 144, imgW: 178, imgH: 668,
    labelX: 436, labelY: 129,
    footerY1: 836, footerY2: 856,
  }),
});

/**
 * Render a 1400x900 Visual Swap Board for surgical single-slot replacement:
 *   - Left half: the 3 LOCKED garments rendered in true proportion with slot labels
 *     plus a warm highlight placeholder marking `targetSlot`.
 *   - Right half: up to 6 photographed replacement candidates (`R1..R6`) for
 *     `targetSlot` with bold high-contrast `R1..R6` badges + brand & price.
 */
export async function renderVisualSwapBoard(lockedItems, targetSlot, candidates, options = {}) {
  const started = performance.now();
  const {
    title = null,
    subtitle = '',
    maxCandidates = 6,
    anonymizeLabels = false,
    includePriceTags = true,
    maxOutputBytes = DEFAULT_SWAP_MAX_BYTES,
    deadlineMs = 7000,
    signal = null,
    photoCache = null,
    resolvePhoto = null,
    authorizePhoto = null,
    allowlist = HOST_ALLOWLIST,
    fetchImpl = globalThis.fetch,
    sharpImpl = sharp,
  } = options;

  const normTargetSlot = String(targetSlot || '').trim().toLowerCase();
  const targetSlotLabel = (normTargetSlot || 'SLOT').toUpperCase();
  const targetZoneKey = CANONICAL_SLOTS.includes(normTargetSlot) ? normTargetSlot : 'outerwear';
  const candidateCap = Math.max(1, Math.min(6, Math.floor(Number(maxCandidates) || 6)));

  const rawLocked = (Array.isArray(lockedItems) ? lockedItems : [])
    .map(unwrapGarmentEntry)
    .filter(Boolean);
  const { bySlot: lockedBySlot, slotLabels } = mapItemsToBoardSlots(rawLocked);
  // Ensure the target slot itself is never rendered as a locked garment
  if (CANONICAL_SLOTS.includes(normTargetSlot)) {
    lockedBySlot[normTargetSlot] = null;
  }

  const candidateList = [];
  const seenCandidateIds = new Set();
  for (const entry of Array.isArray(candidates) ? candidates : []) {
    const item = unwrapGarmentEntry(entry);
    if (!item) continue;
    const dedupeKey = item.id != null ? String(item.id) : null;
    if (dedupeKey && seenCandidateIds.has(dedupeKey)) continue;
    if (dedupeKey) seenCandidateIds.add(dedupeKey);
    candidateList.push(item);
    if (candidateList.length >= candidateCap) break;
  }

  const totalCandidatesRequested = candidateList.length;
  const candidateTokens = {};
  const lockedSlotStatus = {};
  const skippedCandidates = [];
  let cacheHits = 0;
  let cacheMisses = 0;

  if (totalCandidatesRequested === 0) {
    return {
      ok: false,
      error: 'empty_swap_candidates',
      jpegBuffer: null,
      bytes: null,
      width: SWAP_BOARD_WIDTH,
      height: SWAP_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256: null,
      candidateTokens,
      candidateTokenMap: new Map(),
      lockedSlotStatus,
      skippedCandidates,
      renderedCount: 0,
      totalCandidatesRequested: 0,
      renderMs: Math.round(performance.now() - started),
      cacheHits: 0,
      cacheMisses: 0,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(400, Math.min(20_000, deadlineMs)));
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;

  const candCardW = 230;
  const candCardH = 375;
  const candGapX = 14;
  const candGapY = 14;
  const candStartX = 657;
  const candStartY = 108;
  const candImgW = 210;
  const candImgH = 280;

  try {
    const [lockedDecoded, candidatesDecoded] = await Promise.all([
      Promise.all(
        CANONICAL_SLOTS.map(async (slot) => {
          const item = lockedBySlot[slot];
          const zone = SWAP_LOCKED_ZONES[slot];
          if (!item) {
            lockedSlotStatus[slot] = { status: slot === targetZoneKey ? 'target_slot' : 'absent', cacheHit: false };
            return { slot, item: null, zone, rawComposite: null };
          }
          const fetched = await resolveItemRawBytes(item, {
            resolvePhoto,
            authorizePhoto,
            allowlist,
            fetchImpl,
            photoCache,
            signal: combinedSignal,
            fetchTimeoutMs: Math.min(DEFAULT_FETCH_TIMEOUT_MS, deadlineMs),
          });
          if (fetched.cacheHit) cacheHits++;
          else cacheMisses++;
          if (!fetched.ok) {
            lockedSlotStatus[slot] = { status: fetched.status, cacheHit: fetched.cacheHit };
            return { slot, item, zone, rawComposite: null };
          }
          try {
            const decoded = await decodeNormalizedPhotoBytes(fetched.bytes, {
              width: zone.imgW,
              height: zone.imgH,
              background: STUDIO_CARD_HEX,
              sharpImpl,
            });
            lockedSlotStatus[slot] = {
              status: 'rendered',
              cacheHit: fetched.cacheHit,
              sourceFormat: decoded.sourceFormat,
            };
            return {
              slot,
              item,
              zone,
              rawComposite: {
                input: decoded.raw,
                raw: {
                  width: decoded.info.width,
                  height: decoded.info.height,
                  channels: decoded.info.channels,
                },
                left: zone.imgX,
                top: zone.imgY,
              },
            };
          } catch (err) {
            lockedSlotStatus[slot] = { status: err?.code || 'decode_failed', cacheHit: fetched.cacheHit };
            return { slot, item, zone, rawComposite: null };
          }
        }),
      ),
      Promise.all(
        candidateList.map(async (item) => {
          const fetched = await resolveItemRawBytes(item, {
            resolvePhoto,
            authorizePhoto,
            allowlist,
            fetchImpl,
            photoCache,
            signal: combinedSignal,
            fetchTimeoutMs: Math.min(DEFAULT_FETCH_TIMEOUT_MS, deadlineMs),
          });
          if (fetched.cacheHit) cacheHits++;
          else cacheMisses++;
          if (!fetched.ok) {
            return { ok: false, item, status: fetched.status };
          }
          try {
            const decoded = await decodeNormalizedPhotoBytes(fetched.bytes, {
              width: candImgW,
              height: candImgH,
              background: STUDIO_CARD_HEX,
              sharpImpl,
            });
            return { ok: true, item, decoded };
          } catch (err) {
            return { ok: false, item, status: err?.code || 'decode_failed' };
          }
        }),
      ),
    ]);

    if (combinedSignal.aborted) {
      return {
        ok: false,
        error: 'deadline',
        jpegBuffer: null,
        bytes: null,
        width: SWAP_BOARD_WIDTH,
        height: SWAP_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        candidateTokens,
        candidateTokenMap: new Map(),
        lockedSlotStatus,
        skippedCandidates,
        renderedCount: 0,
        totalCandidatesRequested,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const renderedCandidates = [];
    for (const c of candidatesDecoded) {
      if (!c.ok) {
        skippedCandidates.push({
          id: c.item?.id != null ? String(c.item.id) : null,
          status: c.status,
        });
        continue;
      }
      renderedCandidates.push(c);
    }

    if (renderedCandidates.length === 0) {
      return {
        ok: false,
        error: 'no_rendered_candidates',
        jpegBuffer: null,
        bytes: null,
        width: SWAP_BOARD_WIDTH,
        height: SWAP_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        candidateTokens,
        candidateTokenMap: new Map(),
        lockedSlotStatus,
        skippedCandidates,
        renderedCount: 0,
        totalCandidatesRequested,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const baseShapes = [];
    const overlayShapes = [];
    const rawComposites = [];

    // Left Half: 3 locked garments in proportion + target slot callout
    for (const entry of lockedDecoded) {
      const { slot, item, zone, rawComposite } = entry;
      const st = lockedSlotStatus[slot]?.status;
      const isTarget = slot === targetZoneKey && !item;

      if (isTarget) {
        baseShapes.push(
          `<rect x="${zone.cardX}" y="${zone.cardY}" width="${zone.cardW}" height="${zone.cardH}" rx="10" fill="#FAF4EB" stroke="#B88A44" stroke-width="2" stroke-dasharray="8 6"/>`,
        );
        const cx = Math.round(zone.cardX + zone.cardW / 2);
        const cy = Math.round(zone.cardY + zone.cardH / 2);
        overlayShapes.push(
          `<text x="${zone.labelX}" y="${zone.labelY}" fill="#9B6E2C" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="800" letter-spacing="1.4">SWAP TARGET</text>`,
          `<text x="${cx}" y="${cy - 10}" text-anchor="middle" fill="#8A5F22" font-family="Helvetica Neue,Arial,sans-serif" font-size="15" font-weight="800" letter-spacing="1.2">${escapeXml(targetSlotLabel)}</text>`,
          `<text x="${cx}" y="${cy + 14}" text-anchor="middle" fill="#8A5F22" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="700">PICK R1–R${renderedCandidates.length} →</text>`,
        );
        continue;
      }

      const fill = st === 'rendered' ? STUDIO_CARD_HEX : STUDIO_PLACEHOLDER_HEX;
      baseShapes.push(
        `<rect x="${zone.cardX}" y="${zone.cardY}" width="${zone.cardW}" height="${zone.cardH}" rx="10" fill="${fill}" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.4"/>`,
      );
      overlayShapes.push(
        `<text x="${zone.labelX}" y="${zone.labelY}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="800" letter-spacing="1.4">LOCKED · ${escapeXml(slotLabels[slot])}</text>`,
      );

      if (rawComposite) {
        rawComposites.push(rawComposite);
      } else {
        const cx = Math.round(zone.imgX + zone.imgW / 2);
        const cy = Math.round(zone.imgY + zone.imgH / 2);
        const msg = st === 'absent' ? 'NONE' : 'PHOTO UNAVAILABLE';
        overlayShapes.push(
          `<text x="${cx}" y="${cy}" text-anchor="middle" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600">${msg}</text>`,
        );
      }

      if (item && !anonymizeLabels) {
        const brand = sanitizeLabelText(item.brand || item.subtype || '', 19);
        const price = includePriceTags ? formatPriceTag(item) : '';
        if (brand) {
          overlayShapes.push(
            `<text x="${zone.labelX}" y="${zone.footerY1}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="700">${escapeXml(brand)}</text>`,
          );
        }
        if (price) {
          overlayShapes.push(
            `<text x="${zone.labelX}" y="${zone.footerY2}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600">${escapeXml(price)}</text>`,
          );
        }
      }
    }

    // Right Half: up to 6 photographed replacement candidates R1..R6
    for (let idx = 0; idx < renderedCandidates.length; idx++) {
      const entry = renderedCandidates[idx];
      const token = `R${idx + 1}`;
      candidateTokens[token] = entry.item;

      const row = Math.floor(idx / 3);
      const col = idx % 3;
      const cardX = candStartX + col * (candCardW + candGapX);
      const cardY = candStartY + row * (candCardH + candGapY);
      const imgX = cardX + 10;
      const imgY = cardY + 40;

      baseShapes.push(
        `<rect x="${cardX}" y="${cardY}" width="${candCardW}" height="${candCardH}" rx="10" fill="${STUDIO_CARD_HEX}" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.4"/>`,
      );
      rawComposites.push({
        input: entry.decoded.raw,
        raw: {
          width: entry.decoded.info.width,
          height: entry.decoded.info.height,
          channels: entry.decoded.info.channels,
        },
        left: imgX,
        top: imgY,
      });

      // Bold high-contrast candidate token badge (R1..R6)
      overlayShapes.push(
        `<rect x="${cardX + 10}" y="${cardY + 8}" width="50" height="26" rx="5" fill="${STUDIO_INK_HEX}"/>`,
        `<text x="${cardX + 35}" y="${cardY + 26}" text-anchor="middle" fill="#FFFFFF" font-family="Helvetica Neue,Arial,sans-serif" font-size="15" font-weight="800" letter-spacing="0.6">${escapeXml(token)}</text>`,
      );

      if (!anonymizeLabels) {
        const subtypeTag = sanitizeLabelText(String(entry.item?.subtype || '').replace(/[_-]+/g, ' '), 16);
        if (subtypeTag && subtypeTag.toLowerCase() !== 'unknown') {
          overlayShapes.push(
            `<text x="${cardX + 68}" y="${cardY + 25}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="10" font-weight="600">${escapeXml(subtypeTag.toUpperCase())}</text>`,
          );
        }
        const brand = sanitizeLabelText(entry.item?.brand || entry.item?.subtype || '', 22);
        const price = includePriceTags ? formatPriceTag(entry.item) : '';
        if (brand) {
          overlayShapes.push(
            `<text x="${cardX + 10}" y="${cardY + 342}" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="13" font-weight="700">${escapeXml(brand)}</text>`,
          );
        }
        if (price) {
          overlayShapes.push(
            `<text x="${cardX + 10}" y="${cardY + 361}" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="600">${escapeXml(price)}</text>`,
          );
        }
      }
    }

    const renderedCount = renderedCandidates.length;
    const candidateTokenMap = new Map(Object.entries(candidateTokens));
    const defaultTitle = `VISUAL SURGICAL SWAP · REPLACE ${targetSlotLabel}`;
    const headerTitle = anonymizeLabels
      ? escapeXml(defaultTitle)
      : escapeXml(sanitizeLabelText(title || defaultTitle, 64).toUpperCase());
    const headerSub = anonymizeLabels
      ? `LEFT: LOCKED GARMENTS · RIGHT: ${renderedCount} CANDIDATE REPLACEMENTS (R1–R${renderedCount})`
      : escapeXml(sanitizeLabelText(subtitle || `LEFT: 3 LOCKED GARMENTS · RIGHT: CHOOSE BEST ${targetSlotLabel} REPLACEMENT R1–R${renderedCount}`, 100).toUpperCase());

    const baseSvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${SWAP_BOARD_WIDTH}" height="${SWAP_BOARD_HEIGHT}">`
      + `<rect width="${SWAP_BOARD_WIDTH}" height="${SWAP_BOARD_HEIGHT}" fill="${STUDIO_BACKDROP_HEX}"/>`
      + `<text x="26" y="34" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="19" font-weight="800" letter-spacing="1.6">${headerTitle}</text>`
      + `<text x="26" y="56" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="11" font-weight="600" letter-spacing="1.1">${headerSub}</text>`
      + `<line x1="26" y1="68" x2="1374" y2="68" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`
      + `<line x1="640" y1="78" x2="640" y2="876" stroke="${STUDIO_BORDER_HEX}" stroke-width="1.5"/>`
      + `<text x="24" y="96" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="800" letter-spacing="1.5">LOCKED ENSEMBLE · KEEPING 3 PIECES</text>`
      + `<text x="657" y="96" fill="${STUDIO_INK_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-weight="800" letter-spacing="1.5">CANDIDATE REPLACEMENTS · ${escapeXml(targetSlotLabel)} (R1–R${renderedCount})</text>`
      + baseShapes.join('')
      + `</svg>`,
    );

    overlayShapes.push(
      `<text x="26" y="891" fill="${STUDIO_MUTED_HEX}" font-family="Helvetica Neue,Arial,sans-serif" font-size="10" font-weight="600" letter-spacing="1.3">STYLIST 6 VISUAL SWAP BOARD · PROPORTION-SCALED LOCKED ENSEMBLE + REAL PHOTOGRAPHED REPLACEMENTS</text>`,
    );

    const overlaySvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${SWAP_BOARD_WIDTH}" height="${SWAP_BOARD_HEIGHT}">${overlayShapes.join('')}</svg>`,
    );

    const composites = [
      ...rawComposites,
      { input: overlaySvg, left: 0, top: 0 },
    ];

    const pipeline = sharpImpl(baseSvg).composite(composites);
    const encoded = await encodeBoundedJpeg(pipeline, maxOutputBytes, [82, 72, 60, 48]);
    if (!encoded.ok) {
      return {
        ok: false,
        error: 'oversized',
        jpegBuffer: null,
        bytes: null,
        width: SWAP_BOARD_WIDTH,
        height: SWAP_BOARD_HEIGHT,
        mediaType: 'image/jpeg',
        mimeType: 'image/jpeg',
        sha256: null,
        candidateTokens,
        candidateTokenMap,
        lockedSlotStatus,
        skippedCandidates,
        renderedCount,
        totalCandidatesRequested,
        renderMs: Math.round(performance.now() - started),
        cacheHits,
        cacheMisses,
        version: EDITORIAL_BOARD_RENDERER_VERSION,
      };
    }

    const sha256 = createHash('sha256').update(encoded.buffer).digest('hex');
    return {
      ok: true,
      jpegBuffer: encoded.buffer,
      bytes: encoded.buffer,
      width: SWAP_BOARD_WIDTH,
      height: SWAP_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256,
      candidateTokens,
      candidateTokenMap,
      lockedSlotStatus,
      skippedCandidates,
      renderedCount,
      totalCandidatesRequested,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } catch (err) {
    return {
      ok: false,
      error: combinedSignal.aborted ? 'deadline' : (err?.code || 'photo_processing'),
      jpegBuffer: null,
      bytes: null,
      width: SWAP_BOARD_WIDTH,
      height: SWAP_BOARD_HEIGHT,
      mediaType: 'image/jpeg',
      mimeType: 'image/jpeg',
      sha256: null,
      candidateTokens,
      candidateTokenMap: new Map(Object.entries(candidateTokens)),
      lockedSlotStatus,
      skippedCandidates,
      renderedCount: Object.keys(candidateTokens).length,
      totalCandidatesRequested,
      renderMs: Math.round(performance.now() - started),
      cacheHits,
      cacheMisses,
      version: EDITORIAL_BOARD_RENDERER_VERSION,
    };
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}
