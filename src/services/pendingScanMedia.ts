import * as FileSystem from 'expo-file-system/legacy';
import { authEvents } from '../../store/authEvents';

const ROOT = 'aiwardrobe-scan-queue';
const UUID = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}';
const validOwner = new RegExp(`^${UUID}$`, 'i');
const managed = new RegExp(`/${ROOT}/(${UUID})/(${UUID})\\.(jpg|jpeg|png|webp|heic|heif|mp4|mov|m4v)$`, 'i');
const activeRemovals = new Set<string>();
let work: Promise<void> = Promise.resolve();
function serialize<T>(action: () => Promise<T>): Promise<T> {
  const result = work.then(action); work = result.then(() => undefined, () => undefined); return result;
}
function directory(owner: string): string {
  if (!validOwner.test(owner) || !FileSystem.documentDirectory) throw new Error('scan_media_invalid_owner');
  return `${FileSystem.documentDirectory}${ROOT}/${owner}/`;
}
function reference(uri: string) {
  try {
    const url = new URL(uri);
    if (url.protocol !== 'file:' || url.search || url.hash) return null;
    return url.pathname.match(managed);
  } catch { return null; }
}
export function resolvePendingScanMedia(uri: string, owner: string): string {
  const match = reference(uri);
  if (!match) return uri;
  if (match[1] !== owner) throw new Error('scan_media_wrong_owner');
  return `${directory(owner)}${match[2]}.${match[3]}`;
}
/** Commit an OS file copy before the queue record is acknowledged. */
export function stagePendingScanMedia(owner: string, uri: string, type: 'image' | 'video'): Promise<string> {
  const revision = authEvents.getRevision();
  const check = () => {
    if (revision !== authEvents.getRevision() || owner !== authEvents.getUserId()) throw new Error('scan_media_account_changed');
  };
  return serialize(async () => {
    check();
    const folder = directory(owner);
    const source = resolvePendingScanMedia(uri, owner);
    if (!/^(file|content):\/\//i.test(source)) throw new Error('scan_media_invalid_source');
    const info = await FileSystem.getInfoAsync(source); check();
    const limit = type === 'video' ? 100 * 1024 * 1024 : 14 * 1024 * 1024;
    if (!info.exists || info.isDirectory || !Number.isSafeInteger(info.size) || info.size <= 0 || info.size > limit) throw new Error('scan_media_invalid_size');
    if (reference(source)) return source;
    const id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.floor(Math.random() * 16); return (c === 'x' ? r : (r & 3) | 8).toString(16); });
    const extension = source.match(/\.(jpg|jpeg|png|webp|heic|heif|mp4|mov|m4v)$/i)?.[1].toLowerCase() || (type === 'video' ? 'mp4' : 'jpg');
    const destination = `${folder}${id}.${extension}`, temporary = `${folder}.${id}.pending`;
    const existing = await FileSystem.getInfoAsync(destination); check();
    if (existing.exists) throw new Error('scan_media_id_collision');
    let moved = false;
    try {
      await FileSystem.makeDirectoryAsync(folder, { intermediates: true }); check();
      await FileSystem.copyAsync({ from: source, to: temporary }); check();
      const copied = await FileSystem.getInfoAsync(temporary); check();
      if (!copied.exists || copied.isDirectory || copied.size !== info.size) throw new Error('scan_media_copy_incomplete');
      await FileSystem.moveAsync({ from: temporary, to: destination }); moved = true; check();
      const committed = await FileSystem.getInfoAsync(destination); check();
      if (!committed.exists || committed.isDirectory || committed.size !== info.size) throw new Error('scan_media_commit_incomplete');
      return destination;
    } catch (error) {
      await FileSystem.deleteAsync(temporary, { idempotent: true }).catch(() => undefined);
      if (moved) await FileSystem.deleteAsync(destination, { idempotent: true }).catch(() => undefined);
      throw error;
    }
  });
}
function removalPaths(owner: string, uri: string): { path: string; marker: string } | null {
  const match = reference(uri);
  if (!match || match[1] !== owner) return null;
  const path = resolvePendingScanMedia(uri, owner);
  return { path, marker: `${directory(owner)}.${match[2]}.${match[3]}.remove` };
}
/** Persist deletion intent before a queue record can disappear. */
export function preparePendingScanRemoval(owner: string, uri: string): Promise<void> {
  const paths = removalPaths(owner, uri);
  if (!paths) return Promise.resolve();
  const revision = authEvents.getRevision();
  return serialize(async () => {
    if (owner !== authEvents.getUserId() || revision !== authEvents.getRevision()) throw new Error('scan_media_account_changed');
    const folder = await FileSystem.getInfoAsync(directory(owner));
    if (owner !== authEvents.getUserId() || revision !== authEvents.getRevision()) throw new Error('scan_media_account_changed');
    if (!folder.exists) return;
    activeRemovals.add(paths.path);
    try { await FileSystem.writeAsStringAsync(paths.marker, 'remove'); }
    catch (error) { activeRemovals.delete(paths.path); throw error; }
  });
}
export function cancelPendingScanRemoval(owner: string, uri: string): Promise<void> {
  const paths = removalPaths(owner, uri);
  if (!paths) return Promise.resolve();
  return serialize(async () => {
    try { await FileSystem.deleteAsync(paths.marker, { idempotent: true }); }
    finally { activeRemovals.delete(paths.path); }
  });
}
/** Never delete a picker/camera source or a different owner's managed file. */
export function removePendingScanMedia(owner: string, uri: string, references?: () => string[]): Promise<void> {
  const paths = removalPaths(owner, uri);
  if (!paths) return Promise.resolve();
  return serialize(async () => {
    try {
      const folder = await FileSystem.getInfoAsync(directory(owner));
      if (!folder.exists) return;
      const retained = references?.().some(referenceUri => {
        try { return resolvePendingScanMedia(referenceUri, owner) === paths.path; } catch { return false; }
      });
      if (retained) { await FileSystem.deleteAsync(paths.marker, { idempotent: true }); return; }
      await FileSystem.writeAsStringAsync(paths.marker, 'remove');
      await FileSystem.deleteAsync(paths.path, { idempotent: true });
      await FileSystem.deleteAsync(paths.marker, { idempotent: true });
    } finally { activeRemovals.delete(paths.path); }
  });
}
/** Reconcile only explicit deletion intents and interrupted temporary copies.
 * Unmarked committed files are retained: they may be an unacknowledged capture.
 */
export function reconcilePendingScanMedia(owner: string, references: () => string[]): Promise<void> {
  const revision = authEvents.getRevision();
  const current = () => owner === authEvents.getUserId() && revision === authEvents.getRevision();
  return serialize(async () => {
    if (!current() || !validOwner.test(owner) || !FileSystem.documentDirectory) return;
    const folder = directory(owner);
    const info = await FileSystem.getInfoAsync(folder);
    if (!current() || !info.exists) return;
    const names = await FileSystem.readDirectoryAsync(folder);
    for (const name of names) {
      if (!current()) return;
      const temporary = new RegExp(`^\\.${UUID}\\.pending$`, 'i').test(name);
      const marker = name.match(new RegExp(`^\\.(${UUID}\\.(?:jpg|jpeg|png|webp|heic|heif|mp4|mov|m4v))\\.remove$`, 'i'));
      if (!temporary && !marker) continue;
      const path = folder + name;
      const entry = await FileSystem.getInfoAsync(path);
      if (!current()) return;
      if (!entry.exists || entry.isDirectory) continue;
      if (temporary) { await FileSystem.deleteAsync(path, { idempotent: true }); continue; }
      const mediaPath = folder + marker![1];
      if (activeRemovals.has(mediaPath)) continue;
      const retained = references().some(uri => {
        try { return resolvePendingScanMedia(uri, owner) === mediaPath; } catch { return false; }
      });
      if (!current()) return;
      if (!retained) await FileSystem.deleteAsync(mediaPath, { idempotent: true });
      if (!current()) return;
      await FileSystem.deleteAsync(path, { idempotent: true });
    }
  });
}
export function clearAllPendingScanMedia(expectedRevision: number): Promise<void> {
  return serialize(async () => {
    if (expectedRevision !== authEvents.getRevision()) return;
    if (!FileSystem.documentDirectory) return;
    await FileSystem.deleteAsync(`${FileSystem.documentDirectory}${ROOT}/`, { idempotent: true });
    activeRemovals.clear();
  });
}
