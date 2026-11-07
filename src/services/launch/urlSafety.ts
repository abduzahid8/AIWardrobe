/**
 * Phase 7M–7O — Storage security, image upload safety, SSRF/URL-fetch
 * safety. Local validators; server revalidates on receipt.
 */

export const PRIVATE_BUCKETS = ['wardrobe-images', 'outfit-images', 'reference-images'] as const;

// ── Image upload safety (7N) ─────────────────────────────────────

export const ALLOWED_IMAGE_MIME = ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'] as const;

export const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
export const MAX_IMAGE_DIMENSION_PX = 8192;

export interface UploadCandidate {
    mime: string;
    bytes: number;
    width: number | null;
    height: number | null;
    filename: string;
}

export function validateImageUpload(candidate: UploadCandidate): { ok: boolean; reason: string } {
    if (!(ALLOWED_IMAGE_MIME as readonly string[]).includes(candidate.mime)) return { ok: false, reason: 'unsupported_mime' };
    if (!Number.isFinite(candidate.bytes) || candidate.bytes <= 0 || candidate.bytes > MAX_UPLOAD_BYTES) {
        return { ok: false, reason: 'invalid_size' };
    }
    if (candidate.width !== null && (candidate.width <= 0 || candidate.width > MAX_IMAGE_DIMENSION_PX)) {
        return { ok: false, reason: 'invalid_dimensions' };
    }
    if (candidate.height !== null && (candidate.height <= 0 || candidate.height > MAX_IMAGE_DIMENSION_PX)) {
        return { ok: false, reason: 'invalid_dimensions' };
    }
    if (candidate.filename.includes('..') || candidate.filename.includes('/') || candidate.filename.includes('\\')) {
        return { ok: false, reason: 'unsafe_filename' };
    }
    if (/\.([ej]s|html?|svg|exe|sh|php)\b/i.test(candidate.filename)) return { ok: false, reason: 'executable_filename' };
    return { ok: true, reason: 'ok' };
}

export function sanitizeFilename(name: string): string {
    return name.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 128);
}

// ── SSRF / URL fetch safety (7O) ─────────────────────────────────

const ALLOWED_PROTOCOLS = ['https:'];

export const MAX_FETCH_BYTES = 10 * 1024 * 1024;
export const MAX_REDIRECTS = 3;
export const FETCH_TIMEOUT_MS = 15_000;

export const ALLOWED_FETCH_CONTENT_TYPES = ['text/html', 'application/json', 'image/jpeg', 'image/png', 'image/webp'] as const;

export function validateIngestUrl(raw: string): { ok: boolean; reason: string } {
    let url: URL;
    try {
        url = new URL(raw);
    } catch {
        return { ok: false, reason: 'malformed_url' };
    }
    if (!ALLOWED_PROTOCOLS.includes(url.protocol)) return { ok: false, reason: 'protocol_not_allowed' };
    const host = url.hostname.toLowerCase();
    if (host === 'localhost' || host.endsWith('.localhost')) return { ok: false, reason: 'blocked_host' };
    if (host === '127.0.0.1' || host === '::1' || host === '[::1]') return { ok: false, reason: 'blocked_host' };
    // IPv4 private/link-local ranges.
    const v4 = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (v4) {
        const [, a, b] = v4.map(Number);
        if (a === 10) return { ok: false, reason: 'private_range' };
        if (a === 172 && b >= 16 && b <= 31) return { ok: false, reason: 'private_range' };
        if (a === 192 && b === 168) return { ok: false, reason: 'private_range' };
        if (a === 127) return { ok: false, reason: 'private_range' };
        if (a === 169 && b === 254) return { ok: false, reason: 'link_local' };
        if (a === 0) return { ok: false, reason: 'private_range' };
    }
    if (host.includes(':') && !v4) {
        // IPv6: block loopback, link-local, unique-local.
        const h = host.replace(/^\[|\]$/g, '');
        if (h === '::1' || h.startsWith('fe80:') || h.startsWith('fc') || h.startsWith('fd')) {
            return { ok: false, reason: 'private_range' };
        }
    }
    if (url.username || url.password) return { ok: false, reason: 'credentials_in_url' };
    return { ok: true, reason: 'ok' };
}
