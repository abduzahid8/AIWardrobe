import AsyncStorage from '@react-native-async-storage/async-storage';
import type { OutfitLog } from '../../features/calendar/types';
import { authEvents } from '../../store/authEvents';

export const CALENDAR_STORAGE_PREFIX = 'outfitLogs::';
const storageKey = (owner: string) => CALENDAR_STORAGE_PREFIX + encodeURIComponent(owner);
type Logs = Record<string, OutfitLog>;
let writes: Promise<unknown> = Promise.resolve();

function assertSession(revision: number, owner: string | null): asserts owner is string {
  if (!owner || authEvents.getRevision() !== revision || authEvents.getUserId() !== owner) throw new Error('Calendar account changed');
}

async function readStored(owner: string): Promise<Logs> {
  // The unowned legacy outfitLogs key is deliberately not migrated or overwritten.
  const raw = await AsyncStorage.getItem(storageKey(owner));
  if (!raw) return {};
  const value: unknown = JSON.parse(raw);
  // A corrupt cache must not be silently replaced with an empty calendar.
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid calendar storage');
  return value as Logs;
}

export async function readCalendarLogs(): Promise<Logs> {
  const revision = authEvents.getRevision();
  const owner = authEvents.getUserId();
  if (!owner) return {};
  await writes;
  assertSession(revision, owner);
  const logs = await readStored(owner);
  assertSession(revision, owner);
  return logs;
}

/** All calendar writers share one read/modify/write queue to preserve other dates. */
export function updateCalendarLogs(change: (logs: Logs) => void, revision = authEvents.getRevision()): Promise<Logs> {
  const owner = authEvents.getUserId();
  const operation = writes.then(async () => {
    assertSession(revision, owner);
    const logs = await readStored(owner);
    assertSession(revision, owner);
    change(logs);
    assertSession(revision, owner);
    await AsyncStorage.setItem(storageKey(owner), JSON.stringify(logs));
    assertSession(revision, owner);
    return logs;
  });
  // One failed operation must not poison later saves.
  writes = operation.catch(() => undefined);
  return operation;
}

/** Logout waits for a storage write already in flight before removing its key. */
export async function drainCalendarWrites(): Promise<void> { await writes; }
