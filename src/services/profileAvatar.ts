import { supabase } from '../../lib/supabase';
import { Config } from '../config/env';

export const AVATAR_MAX_BYTES = 5 * 1024 * 1024;
const extensions: Record<string, string> = {
  'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp',
  'image/heic': 'heic', 'image/heif': 'heif',
};
export class AvatarUploadError extends Error {
  constructor(public readonly code: 'unsupported_format' | 'invalid_size') {
    super(code);
    this.name = 'AvatarUploadError';
  }
}

/** Never send a session token to external images or another project's host. */
export function avatarRequestHeaders(uri: string | undefined, accessToken?: string): Record<string, string> | undefined {
  if (!uri || !accessToken) return undefined;
  try {
    const target = new URL(uri), project = new URL(Config.supabase.url);
    if (target.origin !== project.origin || !target.pathname.startsWith('/storage/v1/object/authenticated/avatars/')) return undefined;
    return { Authorization: `Bearer ${accessToken}` };
  } catch { return undefined; }
}

export async function uploadProfileAvatar(userId: string, uri: string, mimeType?: string | null) {
  const extension = uri.split(/[?#]/)[0].split('.').pop()?.toLowerCase();
  const inferredType = Object.entries(extensions).find(([, ext]) => ext === extension)?.[0];
  const type = (mimeType || inferredType || 'image/jpeg').toLowerCase().replace('image/jpg', 'image/jpeg');
  if (!extensions[type]) throw new AvatarUploadError('unsupported_format');
  // React Native uploads use ArrayBuffer; Blob/FormData can corrupt file data.
  const bytes = await (await fetch(uri)).arrayBuffer();
  if (!bytes.byteLength || bytes.byteLength > AVATAR_MAX_BYTES) throw new AvatarUploadError('invalid_size');
  const path = `${userId}/avatar_${Date.now()}_${Math.random().toString(36).slice(2, 10)}.${extensions[type]}`;
  const { error } = await supabase.storage.from('avatars').upload(path, bytes, {
    contentType: type, upsert: false,
  });
  if (error) throw error;
  const { data } = supabase.storage.from('avatars').getPublicUrl(path);
  // getPublicUrl only constructs a URL. Use the authenticated endpoint for
  // this private bucket; never persist a signed URL/token in the profile.
  return { path, url: data.publicUrl.replace('/object/public/', '/object/authenticated/') };
}
