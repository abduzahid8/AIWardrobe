import { saveCloudOutfit, removeCloudOutfit } from '../../src/services/savedOutfitService';
import { authEvents } from '../../store/authEvents';
const mockSession = jest.fn(), mockUpsert = jest.fn(), mockDelete = jest.fn(), mockEq = jest.fn(), mockSingle = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { auth: { getSession: () => mockSession() }, from: () => ({ upsert: mockUpsert, delete: mockDelete }) } }));
const input = { id: 'outfit', date: '2026-10-16', occasion: 'casual', items: [{ id: 'top',name:'Cotton shirt',macroCategory:'top', imageUrl: 'https://example.invalid/top.png' },{id:'layer',name:'Fine cardigan',macroCategory:'outerwear'},{id:'bottom',name:'Chino trousers',macroCategory:'bottom'},{id:'shoes',name:'Leather loafers',macroCategory:'shoes'}] };
beforeEach(() => {
  jest.clearAllMocks(); authEvents.emitLogin('owner');
  mockSession.mockResolvedValue({ data: { session: { user: { id: 'owner' } } }, error: null });
  mockUpsert.mockReturnValue({ select: () => ({ single: mockSingle }) });
  mockSingle.mockResolvedValue({ data: { id: 'outfit' }, error: null });
  mockDelete.mockReturnValue({ eq: mockEq }); mockEq.mockReturnValue({ eq: mockEq, then: (resolve: (v: unknown) => void) => Promise.resolve({ error: null }).then(resolve) });
});

test('repeated saves use the same local ID and authenticated owner', async () => {
  await saveCloudOutfit(input); await saveCloudOutfit(input);
  expect(mockUpsert).toHaveBeenCalledTimes(2);
  for (const [row] of mockUpsert.mock.calls) expect(row).toMatchObject({ id: 'outfit', user_id: 'owner', items: input.items });
});

test('returned error or missing confirmation rejects rather than reporting saved', async () => {
  mockSingle.mockResolvedValueOnce({ data: null, error: new Error('offline') });
  await expect(saveCloudOutfit(input)).rejects.toThrow('offline');
  mockSingle.mockResolvedValueOnce({ data: null, error: null });
  await expect(saveCloudOutfit(input)).rejects.toThrow('not confirmed');
});

test('unsave filters both record and authenticated owner', async () => {
  await removeCloudOutfit('outfit');
  expect(mockEq).toHaveBeenCalledWith('id', 'outfit');
  expect(mockEq).toHaveBeenCalledWith('user_id', 'owner');
});

test('invalid date and missing session cannot save', async () => {
  await expect(saveCloudOutfit({ ...input, date: '2026-02-29' })).rejects.toThrow('Invalid');
  mockSession.mockResolvedValueOnce({ data: { session: null }, error: null });
  await expect(saveCloudOutfit(input)).rejects.toThrow('Signed-in');
  expect(mockUpsert).not.toHaveBeenCalled();
});

test('rejects incomplete or duplicated-slot saves before auth or writes',async()=>{
 await expect(saveCloudOutfit({...input,items:input.items.slice(1)})).rejects.toThrow('Invalid saved outfit');
 await expect(saveCloudOutfit({...input,items:[input.items[0],input.items[0],...input.items.slice(2)]})).rejects.toThrow('Invalid saved outfit');
 expect(mockSession).not.toHaveBeenCalled();expect(mockUpsert).not.toHaveBeenCalled();
});
