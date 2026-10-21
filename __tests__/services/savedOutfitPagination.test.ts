import { listCloudOutfits } from '../../src/services/savedOutfitService';
import { authEvents } from '../../store/authEvents';
const mockRange = jest.fn(), mockEq = jest.fn(), mockOrder = jest.fn(), mockLt = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: {
  auth: { getSession: async () => ({ data: { session: { user: { id: 'owner' } } } }) },
  from: () => ({ select: () => ({ eq: mockEq }) }),
} }));
beforeEach(() => {
  jest.resetAllMocks(); authEvents.emitLogin('owner');
  const chain = { order: mockOrder, range: mockRange, limit: mockRange, lt: mockLt };
  mockEq.mockReturnValue(chain); mockOrder.mockReturnValue(chain); mockLt.mockReturnValue(chain);
});

test('all 1,001 outfits are returned across five owner-scoped pages', async () => {
  const rows = Array.from({ length: 1001 }, (_, i) => ({ id: `outfit-${String(1001-i).padStart(4,'0')}` }));
  for (let offset = 0; offset < rows.length; offset += 250) mockRange.mockResolvedValueOnce({ data: rows.slice(offset, offset + 250), error: null });
  expect((await listCloudOutfits()).map(row => row.id)).toEqual(rows.map(row => row.id));
  expect(mockRange.mock.calls).toEqual(Array.from({length:5}, () => [250]));
  expect(mockLt.mock.calls).toEqual([249,499,749,999].map(index => ['id', rows[index].id]));
  expect(mockEq).toHaveBeenCalledWith('user_id', 'owner');
  expect(mockOrder).toHaveBeenCalledWith('id', { ascending: false });
});

test('page failure rejects the whole refresh rather than returning a partial collection', async () => {
  mockRange.mockResolvedValueOnce({ data: Array.from({ length: 250 }, (_, i) => ({ id: `${i}` })), error: null })
    .mockResolvedValueOnce({ data: null, error: new Error('offline') });
  await expect(listCloudOutfits()).rejects.toThrow('offline');
});

test('account transition during a page read rejects the old collection', async () => {
  let finish!: (value: unknown) => void;
  mockRange.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  const pending = listCloudOutfits();
  await Promise.resolve();
  authEvents.emitLogout(); authEvents.emitLogin('next-owner');
  finish({ data: [{ id: 'old-private-outfit' }], error: null });
  await expect(pending).rejects.toThrow('Account changed');
});

test('concurrent deletion on an earlier page cannot skip an unseen saved outfit', async () => {
  const original = Array.from({ length: 501 }, (_, index) => ({ id: `outfit-${String(501-index).padStart(4,'0')}`, created_at: '2026-10-16' }));
  let live = [...original], cursor: string | undefined, requests = 0;
  mockLt.mockImplementation((_column, value) => { cursor = value; return { limit: mockRange }; });
  mockRange.mockImplementation(async (start: number, end?: number) => {
    const available = cursor ? live.filter(row => row.id < cursor!) : live;
    const rows = end === undefined ? available.slice(0, start) : available.slice(start, end + 1);
    if (requests++ === 0) live = live.slice(1);
    return { data: rows, error: null };
  });
  expect((await listCloudOutfits()).map(row => row.id)).toEqual(original.map(row => row.id));
});

test('a non-advancing collection cursor rejects rather than looping', async () => {
  const rows = Array.from({ length: 250 }, (_, index) => ({ id: `outfit-${String(250-index).padStart(4,'0')}` }));
  mockRange.mockResolvedValue({ data: rows, error: null });
  await expect(listCloudOutfits()).rejects.toThrow('cursor did not advance');
  expect(mockRange).toHaveBeenCalledTimes(2);
});

test('collection results retain chronological display order after ID traversal', async () => {
  mockRange.mockResolvedValue({ data: [
    { id: 'c', created_at: '2026-10-15T12:00:00Z' },
    { id: 'b', created_at: '2026-10-17T12:00:00Z' },
    { id: 'a', created_at: '2026-10-16T12:00:00Z' },
  ], error: null });
  expect((await listCloudOutfits()).map(row => row.id)).toEqual(['b', 'a', 'c']);
});
