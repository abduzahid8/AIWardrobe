import { Image } from '../components/MediaImage';
import { ImageManipulator, SaveFormat, type ImageRef } from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system/legacy';
import { authEvents } from '../../store/authEvents';
import { pendingWardrobeImageReference, resolvePendingWardrobeImage, stageNativeWardrobeImage } from './pendingWardrobeMedia';
import { WardrobeMediaError } from './wardrobeMedia';

export const WARDROBE_THUMBNAIL_EDGE = 512;
export const WARDROBE_JPEG_THUMBNAIL_QUALITY = 0.95;
export const WARDROBE_THUMBNAIL_MAX_INPUT_PIXELS = 50_000_000;

function isCacheFile(uri: string) {
  if (!FileSystem.cacheDirectory) return false;
  try {
    const file = new URL(uri), root = new URL(FileSystem.cacheDirectory);
    const rootPath = root.pathname.endsWith('/') ? root.pathname : root.pathname + '/';
    // deleteAsync can recursively remove directories. Codec output must name a
    // strict descendant file, never the cache root or an encoded path escape.
    const relative = file.pathname.slice(rootPath.length);
    const segments = relative.split('/').map(segment => decodeURIComponent(segment));
    return file.protocol === 'file:' && file.host === root.host && !file.search && !file.hash
      && file.pathname.startsWith(rootPath) && segments.every(segment =>
        segment.length > 0 && segment !== '.' && segment !== '..' && !/[\\/\0]/.test(segment));
  } catch { return false; }
}

let thumbnailWork: Promise<void> = Promise.resolve();

/** Limit native decoding to one photo at a time, including overlapping batch saves. */
export function createNativeWardrobeThumbnail(owner: string, imageId: string, original: string): Promise<string> {
  const revision = authEvents.getRevision();
  const operation = thumbnailWork.then(() => renderThumbnail(owner, imageId, original, revision));
  thumbnailWork = operation.then(() => undefined, () => undefined);
  return operation;
}
export const flushNativeWardrobeThumbnails = () => thumbnailWork;

/** JPEG inputs are opaque; retain PNG for formats that may carry transparency. */
async function renderThumbnail(owner: string, imageId: string, original: string, revision: number): Promise<string> {
  const check = () => {
    if (revision !== authEvents.getRevision() || !authEvents.canApplyToUser(owner)) throw new WardrobeMediaError('account_changed');
  };
  check();
  const reference = pendingWardrobeImageReference(original);
  if (!reference || reference.owner !== owner) throw new WardrobeMediaError('access_denied');
  const source = resolvePendingWardrobeImage(original, owner);
  if (!source) throw new WardrobeMediaError('invalid_source');
  const { width, height } = await Image.getSize(source);
  check();
  if (![width, height].every(value => Number.isSafeInteger(value) && value > 0)
      || width * height > WARDROBE_THUMBNAIL_MAX_INPUT_PIXELS) throw new WardrobeMediaError('invalid_size');
  const context = ImageManipulator.manipulate(source);
  let rendered: ImageRef | undefined;
  let temporary: string | undefined;
  try {
    if (Math.max(width, height) > WARDROBE_THUMBNAIL_EDGE) {
      context.resize(width >= height ? { width: WARDROBE_THUMBNAIL_EDGE } : { height: WARDROBE_THUMBNAIL_EDGE });
    }
    rendered = await context.renderAsync();
    check();
    const opaqueJpeg = reference.extension.toLowerCase() === 'jpg';
    const result = await rendered.saveAsync({
      format: opaqueJpeg ? SaveFormat.JPEG : SaveFormat.PNG,
      compress: opaqueJpeg ? WARDROBE_JPEG_THUMBNAIL_QUALITY : 1,
    });
    temporary = result.uri;
    check();
    const scale = Math.min(1, WARDROBE_THUMBNAIL_EDGE / Math.max(width, height));
    const expectedWidth = Math.max(1, Math.round(width * scale));
    const expectedHeight = Math.max(1, Math.round(height * scale));
    if (!isCacheFile(temporary) || temporary === source
        || ![result.width, result.height].every(value => Number.isSafeInteger(value) && value > 0 && value <= WARDROBE_THUMBNAIL_EDGE)
        || result.width > width || result.height > height
        || Math.abs(result.width - expectedWidth) > 1 || Math.abs(result.height - expectedHeight) > 1) {
      throw new WardrobeMediaError('invalid_source');
    }
    return await stageNativeWardrobeImage(owner, imageId, temporary, opaqueJpeg ? 'image/jpeg' : 'image/png');
  } finally {
    // Cache output is disposable only after stageNativeWardrobeImage has copied
    // it (or rejected). Never delete the caller's durable original.
    if (temporary && isCacheFile(temporary) && temporary !== source) {
      await FileSystem.deleteAsync(temporary, { idempotent: true }).catch(() => undefined);
    }
    rendered?.release();
    context.release();
  }
}
