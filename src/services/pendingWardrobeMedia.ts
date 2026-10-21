import * as FileSystem from 'expo-file-system/legacy';
import { encode } from 'base64-arraybuffer';
import { authEvents } from '../../store/authEvents';
import { readWardrobeImageInput, wardrobeMediaReference, WardrobeMediaError, WARDROBE_MEDIA_MAX_BYTES } from './wardrobeMedia';

const ROOT = 'aiwardrobe-pending-media';
const UUID = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}';
const validId = new RegExp(`^${UUID}$`, 'i');
const managedPath = new RegExp(`/${ROOT}/(${UUID})/(${UUID})\\.(jpg|png|webp|heic|heif)$`, 'i');
let fileWork: Promise<void> = Promise.resolve();
function serialize<T>(action: () => Promise<T>): Promise<T> {
  const operation = fileWork.then(action);
  fileWork = operation.then(() => undefined, () => undefined);
  return operation;
}
export function pendingWardrobeImageReference(uri: string) {
  if (!uri.startsWith('file:')) return null;
  try {
    const parsed = new URL(uri);
    if (parsed.search || parsed.hash) return null;
    const match = parsed.pathname.match(managedPath);
    return match ? { owner: match[1], id: match[2], extension: match[3] } : null;
  } catch { return null; }
}
function ownerDirectory(owner: string) {
  if (!validId.test(owner) || !FileSystem.documentDirectory) throw new WardrobeMediaError('invalid_source');
  return `${FileSystem.documentDirectory}${ROOT}/${owner}/`;
}

/** Rebase managed files after iOS changes the application's container path. */
export function resolvePendingWardrobeImage(uri: string, owner: string): string | undefined {
  const stored = pendingWardrobeImageReference(uri);
  if (!stored) return uri;
  if (stored.owner !== owner || !FileSystem.documentDirectory) return undefined;
  return `${ownerDirectory(owner)}${stored.id}.${stored.extension}`;
}

/** Native-only: commit validated original bytes in Documents before queue persistence. */
export async function stageNativeWardrobeImage(owner: string, imageId: string, uri: string, mimeType?: string | null): Promise<string> {
  const revision = authEvents.getRevision();
  const check = () => {
    if (revision !== authEvents.getRevision() || !authEvents.canApplyToUser(owner)) throw new WardrobeMediaError('account_changed');
  };
  check();
  if (!validId.test(owner) || !validId.test(imageId)) throw new WardrobeMediaError('invalid_source');
  const cloud = wardrobeMediaReference(uri);
  if (cloud) {
    if (cloud.owner !== owner) throw new WardrobeMediaError('access_denied');
    return cloud.canonicalUrl;
  }
  if (/^https?:\/\//i.test(uri)) return uri;
  // Queue the read as well as the commit: native fetch and base64 encoding
  // otherwise allocate complete copies for every concurrently submitted photo.
  return serialize(async () => {
    check();
    const source = resolvePendingWardrobeImage(uri, owner);
    if (!source) throw new WardrobeMediaError('access_denied');
    const directory = ownerDirectory(owner);
    if (/^file:/i.test(source)) {
      // Native fetch buffers the complete file. Reject excessive inputs before
      // allocating that buffer (and the subsequent base64 staging copy).
      const info = await FileSystem.getInfoAsync(source);
      check();
      if (!info.exists || info.isDirectory) throw new WardrobeMediaError('invalid_source');
      if (!Number.isSafeInteger(info.size) || info.size <= 0 || info.size > WARDROBE_MEDIA_MAX_BYTES) {
        throw new WardrobeMediaError('invalid_size');
      }
    }
    const { bytes, extension } = await readWardrobeImageInput(source, mimeType, check);
    const encoded = encode(bytes);
    const destination = `${directory}${imageId}.${extension}`;
    const temporary = `${directory}.${imageId}.${extension}.pending`;
    check();
    await FileSystem.makeDirectoryAsync(directory, { intermediates: true });
    check();
    const existing = await FileSystem.getInfoAsync(destination);
    if (existing.exists) {
      if (existing.isDirectory) throw new WardrobeMediaError('invalid_source');
      const saved = await FileSystem.readAsStringAsync(destination, { encoding: FileSystem.EncodingType.Base64 });
      check();
      if (saved !== encoded) throw new WardrobeMediaError('invalid_source');
      return destination; // Idempotent retry cannot silently replace another photo.
    }
    let moved = false;
    try {
      await FileSystem.writeAsStringAsync(temporary, encoded, { encoding: FileSystem.EncodingType.Base64 });
      check();
      await FileSystem.moveAsync({ from: temporary, to: destination }); moved = true;
      check();
      const committed = await FileSystem.getInfoAsync(destination);
      check();
      if (!committed.exists || committed.isDirectory || committed.size !== bytes.byteLength) throw new WardrobeMediaError('invalid_source');
      return destination;
    } catch (error) {
      await FileSystem.deleteAsync(temporary, { idempotent: true }).catch(() => undefined);
      if (moved) await FileSystem.deleteAsync(destination, { idempotent: true }).catch(() => undefined);
      throw error;
    }
  });
}

/** Delete only managed files belonging to the explicitly supplied former owner. */
export function removePendingWardrobeImage(owner: string, uri: string): Promise<void> {
  const stored = pendingWardrobeImageReference(uri);
  if (!stored || stored.owner !== owner) return Promise.resolve();
  const path = resolvePendingWardrobeImage(uri, owner);
  return path ? serialize(() => FileSystem.deleteAsync(path, { idempotent: true })) : Promise.resolve();
}
export function clearPendingWardrobeImages(owner: string): Promise<void> {
  const directory = ownerDirectory(owner);
  return serialize(() => FileSystem.deleteAsync(directory, { idempotent: true }));
}
/** Account-boundary cleanup also removes orphaned files without persisted rows.
 * Share the staging queue so a previously started commit cannot recreate files
 * after deletion. A superseded cleanup cannot delete a newer account's files.
 */
export function clearAllPendingWardrobeImages(expectedRevision: number): Promise<void> {
  return serialize(async () => {
    const check = () => {
      if (expectedRevision !== authEvents.getRevision()) throw new WardrobeMediaError('account_changed');
    };
    check();
    if (!FileSystem.documentDirectory) throw new WardrobeMediaError('invalid_source');
    await FileSystem.deleteAsync(`${FileSystem.documentDirectory}${ROOT}/`, { idempotent: true });
    check();
  });
}
export const flushPendingWardrobeImageWork = () => fileWork;
