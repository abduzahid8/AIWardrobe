import { processPendingActions, type PendingAction } from '../../store/wardrobeSyncService';
import { authEvents } from '../../store/authEvents';
import type { ClothingItem } from '../../src/types/domain';

const mockUpload = jest.fn(), mockFrom = jest.fn();
const owner = '11111111-1111-4111-8111-111111111111';
const id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
jest.mock('../../lib/supabase', () => ({ supabase: {
  auth: { getSession: async () => ({ data: { session: { user: { id: '11111111-1111-4111-8111-111111111111' } } } }) },
  storage: { from: () => ({ upload: (...args: unknown[]) => mockUpload(...args) }) },
  from: (...args: unknown[]) => mockFrom(...args),
} }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
jest.mock('expo-file-system/legacy', () => ({ documentDirectory: 'file:///Documents/' }));

// Real sync, preparation, validation and canonicalization; only the remote
// boundary is simulated. Commit the bytes/row BEFORE losing its response.
it.each(['error response', 'thrown transport error'])('retry preserves one record and two exact objects after %s', async mode => {
  authEvents.emitLogin(owner);
  const objects = new Map<string, number[]>();
  const rows = new Map<string, Record<string, unknown>>();
  let loseUpload = true, loseRow = true;
  const loss = () => {
    if (mode === 'thrown transport error') throw new Error('connection lost after commit');
    return { error: new Error('response lost after commit') };
  };
  mockUpload.mockImplementation(async (path: string, bytes: ArrayBuffer, options: { upsert: boolean }) => {
    expect(options.upsert).toBe(true);
    objects.set(path, [...new Uint8Array(bytes)]);
    if (loseUpload) { loseUpload = false; return loss(); }
    return { error: null };
  });
  const upsert = jest.fn(async (records: Record<string, unknown>[]) => {
    for (const row of records) rows.set(String(row.id), { ...row });
    if (loseRow) { loseRow = false; return loss(); }
    return { error: null };
  });
  mockFrom.mockImplementation((table: string) => {
    expect(table).toBe('clothing_items');
    return {
      select: () => ({ in: async () => ({ data: [...rows.values()].map(row => ({ id: row.id, updated_at: row.updated_at })), error: null }) }),
      upsert,
    };
  });
  const item = { id, imageUrl: 'data:image/jpeg;base64,/9j/2Q==',
    thumbnailUrl: 'data:image/png;base64,iVBORw0KGgo=', updatedAt: '2026-10-17T00:00:00Z' } as ClothingItem;
  const action: PendingAction = { id: 'stable-action', type: 'add_item', payload: item as unknown as Record<string, unknown>, createdAt: item.updatedAt! };
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    const first = await processPendingActions([action], [item], owner);
    expect(first).toEqual({ processedIds: [], updatedItems: [item] });
    expect(objects.size).toBe(1); expect(rows.size).toBe(0);
    const second = await processPendingActions([action], [item], owner);
    expect(second).toEqual({ processedIds: [], updatedItems: [item] });
    expect(objects.size).toBe(2); expect(rows.size).toBe(1);
    const third = await processPendingActions([action], [item], owner);
    expect(third.processedIds).toEqual(['stable-action']);
    expect(rows.size).toBe(1); expect(objects.size).toBe(2);
    expect(objects.get(`${owner}/${id}/original.jpg`)).toEqual([255,216,255,217]);
    expect(objects.get(`${owner}/${id}/thumbnail.png`)).toEqual([137,80,78,71,13,10,26,10]);
    expect(third.updatedItems[0]).toMatchObject({
      id, imageUrl: `https://storage.test/storage/v1/object/authenticated/wardrobe-images/${owner}/${id}/original.jpg`,
      thumbnailUrl: `https://storage.test/storage/v1/object/authenticated/wardrobe-images/${owner}/${id}/thumbnail.png`,
    });
    expect(rows.get(id)).toMatchObject({ id, user_id: owner, image_url: third.updatedItems[0].imageUrl, thumbnail_url: third.updatedItems[0].thumbnailUrl });
    expect(action.payload.imageUrl).toBe(item.imageUrl);
  } finally { error.mockRestore(); jest.clearAllMocks(); }
});

it('retains a queued garment after a local read timeout and can sync it on retry', async () => {
  authEvents.emitLogin(owner);
  jest.useFakeTimers();
  const previousFetch = global.fetch;
  global.fetch = jest.fn().mockImplementationOnce(() => new Promise(() => undefined))
    .mockResolvedValue({ arrayBuffer: async () => new Uint8Array([255,216,255,217]).buffer });
  const upsert = jest.fn().mockResolvedValue({ error: null });
  mockUpload.mockResolvedValue({ error: null });
  mockFrom.mockReturnValue({ select: () => ({ in: async () => ({ data: [], error: null }) }), upsert });
  const item = { id, imageUrl: 'file:///cache/photo.jpg', updatedAt: '2026-10-21T00:00:00Z' } as ClothingItem;
  const action: PendingAction = { id: 'read-timeout', type: 'add_item', payload: item as unknown as Record<string, unknown>, createdAt: item.updatedAt! };
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    const first = processPendingActions([action], [item], owner);
    await jest.advanceTimersByTimeAsync(15_000);
    expect(await first).toEqual({ processedIds: [], updatedItems: [item] });
    expect(mockUpload).not.toHaveBeenCalled();
    expect(upsert).not.toHaveBeenCalled();
    const retried = await processPendingActions([action], [item], owner);
    expect(retried.processedIds).toEqual([action.id]);
    expect(retried.updatedItems[0].imageUrl).toContain('/object/authenticated/wardrobe-images/');
    expect(upsert).toHaveBeenCalledTimes(1);
  } finally {
    global.fetch = previousFetch;
    error.mockRestore(); jest.useRealTimers(); jest.clearAllMocks();
  }
});
