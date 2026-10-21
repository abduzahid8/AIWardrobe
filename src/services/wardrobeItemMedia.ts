import type { ClothingItem } from '../types/domain';
import { uploadWardrobeImage, wardrobeMediaReference, WardrobeMediaError } from './wardrobeMedia';
import { resolvePendingWardrobeImage } from './pendingWardrobeMedia';

/** Resolve managed container paths and replace device-only bytes with durable cloud references. */
export async function prepareWardrobeItemMedia<T extends Pick<ClothingItem, 'id' | 'imageUrl' | 'thumbnailUrl'>>(owner: string, item: T): Promise<T> {
  async function prepare(uri: string, variant: 'original' | 'thumbnail') {
    const cloud = wardrobeMediaReference(uri);
    if (cloud) {
      if (cloud.owner !== owner) throw new WardrobeMediaError('access_denied');
      return cloud.canonicalUrl;
    }
    if (/^https?:\/\//i.test(uri)) {
      try { new URL(uri); } catch { throw new WardrobeMediaError('invalid_source'); }
      return uri;
    }
    const source = resolvePendingWardrobeImage(uri, owner);
    if (!source) throw new WardrobeMediaError('access_denied');
    return uploadWardrobeImage(owner, item.id, source, undefined, variant);
  }
  const imageUrl = await prepare(item.imageUrl, 'original');
  const thumbnailUrl = item.thumbnailUrl
    ? item.thumbnailUrl === item.imageUrl ? imageUrl : await prepare(item.thumbnailUrl, 'thumbnail')
    : item.thumbnailUrl;
  return { ...item, imageUrl, thumbnailUrl };
}
