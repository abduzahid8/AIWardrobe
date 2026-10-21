import { wardrobeMediaReference } from '../services/wardrobeMedia';
import { pendingWardrobeImageReference, resolvePendingWardrobeImage } from '../services/pendingWardrobeMedia';

export interface ImageIdentity { userId?: string; accessToken?: string }

/** Private sources must not reach an image loader/cache after an identity change. */
export function mediaImageSource(source: unknown, identity: ImageIdentity): unknown {
  if (Array.isArray(source)) {
    const sources = source.map(item => mediaImageSource(item, identity)).filter(item => item !== undefined);
    return sources.length ? sources : undefined;
  }
  const value = typeof source === 'string' ? { uri: source } : source;
  if (!value || typeof value !== 'object' || !('uri' in value) || typeof value.uri !== 'string') return source;
  const pending = pendingWardrobeImageReference(value.uri);
  if (pending) {
    if (pending.owner !== identity.userId || !identity.accessToken) return undefined;
    const uri = resolvePendingWardrobeImage(value.uri, pending.owner);
    if (!uri) return undefined;
    const result: Record<string, unknown> = { ...value, uri, cacheKey: `${pending.owner}:pending:${pending.id}.${pending.extension}` };
    delete result.headers; // A local file never needs a bearer credential.
    return result;
  }
  const reference = wardrobeMediaReference(value.uri);
  if (!reference) return source;
  if (reference.owner !== identity.userId || !identity.accessToken) return undefined;
  const current = value as Record<string, unknown>;
  const headers = typeof current.headers === 'object' && current.headers !== null ? current.headers : {};
  return { ...current, uri: reference.canonicalUrl,
    cacheKey: `${reference.owner}:${reference.path}`,
    headers: { ...headers, Authorization: `Bearer ${identity.accessToken}` } };
}

export interface MediaPrefetchOptions { cachePolicy?: 'memory' | 'disk' | 'memory-disk'; headers?: Record<string, string> }
export async function prefetchMediaImages(
  urls: string | string[], options: MediaPrefetchOptions | MediaPrefetchOptions['cachePolicy'], identity: ImageIdentity,
  prefetch: (url: string, options?: MediaPrefetchOptions | MediaPrefetchOptions['cachePolicy']) => Promise<boolean>,
): Promise<boolean> {
  const results = await Promise.all((Array.isArray(urls) ? urls : [urls]).map(uri => {
    if (!wardrobeMediaReference(uri)) return prefetch(uri, options);
    const settings = typeof options === 'string' ? { cachePolicy: options } : options;
    const source = mediaImageSource({ uri, headers: settings?.headers }, identity) as { uri: string; headers: Record<string, string> } | undefined;
    return source ? prefetch(source.uri, { ...settings, headers: source.headers }) : Promise.resolve(false);
  }));
  return results.every(Boolean);
}


/** The installed Android RN image implementation forwards headers only from
 * array-form sources. Preserve its object-form width/height fallback too. */
export function nativeMediaImageSource(source: unknown, platform: string, dimensions: {width?: unknown; height?: unknown} = {}): unknown {
  if(platform!=='android'||!source||typeof source!=='object')return source;
  if(Array.isArray(source)){
    // Android applies the first candidate's headers to the chosen candidate.
    // Mixed credential scopes cannot safely share that native request.
    const first=source[0]?.headers??{};
    if(source.some(item=>{const headers=item?.headers??{},keys=Object.keys(first);return keys.length!==Object.keys(headers).length||keys.some(key=>first[key]!==headers[key]);}))return undefined;
    return source;
  }
  const current=source as Record<string,unknown>;
  if(typeof current.uri!=='string'||!current.headers||typeof current.headers!=='object')return source;
  return [{...current,width:current.width??dimensions.width,height:current.height??dimensions.height}];
}
