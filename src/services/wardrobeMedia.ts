import { decode } from 'base64-arraybuffer';
import { supabase } from '../../lib/supabase';
import { authEvents } from '../../store/authEvents';
import { Config } from '../config/env';
import { withRequestDeadline } from '../lib/requestDeadline';

export const WARDROBE_MEDIA_BUCKET = 'wardrobe-images';
export const WARDROBE_MEDIA_MAX_BYTES = 10 * 1024 * 1024;
const extensions: Record<string, string> = {
  'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/heic': 'heic', 'image/heif': 'heif',
};
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export class WardrobeMediaError extends Error {
  constructor(public readonly code: 'invalid_source' | 'unsupported_format' | 'invalid_size' | 'account_changed' | 'access_denied') {
    super(code); this.name = 'WardrobeMediaError';
  }
}

/** Parse only this project's wardrobe bucket, including temporary signed links. */
export function wardrobeMediaReference(uri: string): { path: string; owner: string; canonicalUrl: string; authenticated: boolean } | null {
  if (!/^https?:\/\//i.test(uri)) return null;
  try {
    const url = new URL(uri), project = new URL(Config.supabase.url);
    if (url.origin !== project.origin || url.username || url.password) return null;
    const match = url.pathname.match(/^\/storage\/v1\/object\/(authenticated|sign)\/wardrobe-images\/(.+)$/);
    if (!match) return null;
    const segments = match[2].split('/').map(decodeURIComponent);
    if (!uuid.test(segments[0]) || segments.some(part => !part || part === '.' || part === '..' || /[\\/]/.test(part))) return null;
    const path = segments.join('/');
    return { path, owner: segments[0], authenticated: match[1] === 'authenticated',
      canonicalUrl: `${project.origin}/storage/v1/object/authenticated/${WARDROBE_MEDIA_BUCKET}/${segments.map(encodeURIComponent).join('/')}` };
  } catch { return null; }
}

/** Credentials go only to the current owner's authenticated object endpoint. */
export function wardrobeMediaHeaders(uri: string, userId?: string, accessToken?: string): Record<string, string> | undefined {
  const reference = wardrobeMediaReference(uri);
  return reference?.authenticated && reference.owner === userId && accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined;
}

function accountGuard(userId: string) {
  const revision = authEvents.getRevision();
  const check = () => {
    if (revision !== authEvents.getRevision() || !authEvents.canApplyToUser(userId)) throw new WardrobeMediaError('account_changed');
  };
  check(); return check;
}

function matchesEncoding(buffer: ArrayBuffer, mime: string): boolean {
  const bytes = new Uint8Array(buffer);
  const ascii = (start: number, end: number) => String.fromCharCode(...bytes.slice(start, end));
  if (mime === 'image/jpeg') return bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
  if (mime === 'image/png') return [137,80,78,71,13,10,26,10].every((byte, index) => bytes[index] === byte);
  if (mime === 'image/webp') return ascii(0,4) === 'RIFF' && ascii(8,12) === 'WEBP';
  return ascii(4,8) === 'ftyp' && ['heic','heix','hevc','hevx','mif1','msf1','heif'].includes(ascii(8,12));
}

/** Read and validate input bytes without uploading or changing their encoding. */
export async function readWardrobeImageInput(uri: string, mimeType?: string | null, check: () => void = () => undefined) {
  let bytes: ArrayBuffer, type: string;
  const data = uri.match(/^data:([^;,]+);base64,([\s\S]*)$/i);
  if (data) {
    type = data[1].toLowerCase();
    if (!extensions[type]) throw new WardrobeMediaError('unsupported_format');
    const encoded = data[2].replace(/\s/g, '');
    if (encoded.length > Math.ceil(WARDROBE_MEDIA_MAX_BYTES / 3) * 4) throw new WardrobeMediaError('invalid_size');
    if (!encoded || encoded.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) throw new WardrobeMediaError('invalid_source');
    bytes = decode(encoded);
  } else {
    if (!/^(file|content|blob):/i.test(uri)) throw new WardrobeMediaError('invalid_source');
    const suffix = uri.split(/[?#]/)[0].split('.').pop()?.toLowerCase();
    const input = await withRequestDeadline(async signal => {
      const response = await fetch(uri, { signal });
      check();
      // Error/redirect bodies must not become images just because their bytes
      // match a supported signature. Retain status 0 for local read adapters.
      if (response.status >= 300) throw new WardrobeMediaError('invalid_source');
      const responseType = response.headers?.get('content-type')?.split(';')[0]?.toLowerCase();
      const type = (mimeType || Object.entries(extensions).find(([, extension]) => extension === suffix)?.[0]
        || (responseType && extensions[responseType] ? responseType : '') || 'image/jpeg').toLowerCase().replace('image/jpg', 'image/jpeg');
      if (!extensions[type]) throw new WardrobeMediaError('unsupported_format');
      return { type, bytes: await response.arrayBuffer() };
    });
    type = input.type;
    bytes = input.bytes;
  }
  check();
  if (!bytes.byteLength || bytes.byteLength > WARDROBE_MEDIA_MAX_BYTES) throw new WardrobeMediaError('invalid_size');
  if (!matchesEncoding(bytes, type)) throw new WardrobeMediaError('invalid_source');
  return { bytes, mimeType: type, extension: extensions[type] };
}

/** Upload unchanged bytes under a stable caller-owned ID; return no expiring token. */
export async function uploadWardrobeImage(userId: string, objectId: string, uri: string, mimeType?: string | null, variant: 'original' | 'thumbnail' = 'original'): Promise<string> {
  const check = accountGuard(userId);
  if (!uuid.test(userId) || !uuid.test(objectId)) throw new WardrobeMediaError('invalid_source');
  const existing = wardrobeMediaReference(uri);
  if (existing) {
    if (existing.owner !== userId) throw new WardrobeMediaError('access_denied');
    return existing.canonicalUrl;
  }
  if (/^https?:\/\//i.test(uri)) {
    try { new URL(uri); } catch { throw new WardrobeMediaError('invalid_source'); }
    return uri; // Existing remote catalog assets do not need a duplicate upload.
  }
  const { bytes, mimeType: type, extension } = await readWardrobeImageInput(uri, mimeType, check);
  check();
  const path = `${userId}/${objectId}/${variant}.${extension}`;
  const { error } = await supabase.storage.from(WARDROBE_MEDIA_BUCKET).upload(path, bytes, { contentType: type, upsert: true });
  check();
  if (error) throw error;
  return `${new URL(Config.supabase.url).origin}/storage/v1/object/authenticated/${WARDROBE_MEDIA_BUCKET}/${path}`;
}

/** Explicit bearer export, never routine rendering or persistence. Token TTL does
 * not revoke already cached responses; CDN deletion invalidation can take time. */
export async function createWardrobeImageAccessUrl(userId: string, uri: string, expiresIn = 300): Promise<string> {
  const check = accountGuard(userId);
  const reference = wardrobeMediaReference(uri);
  if (!reference || reference.owner !== userId) throw new WardrobeMediaError('access_denied');
  if (!Number.isInteger(expiresIn) || expiresIn < 60 || expiresIn > 900) throw new WardrobeMediaError('invalid_source');
  const { data, error } = await supabase.storage.from(WARDROBE_MEDIA_BUCKET).createSignedUrl(reference.path, expiresIn);
  check();
  if (error || !data?.signedUrl) throw error || new Error('Image access was not confirmed');
  return data.signedUrl;
}
