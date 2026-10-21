import { saveClothingEdits } from '../../src/services/clothingEditorService';
import { authEvents } from '../../store/authEvents';
const mockSession = jest.fn();
const mockUpdate = jest.fn();
const mockEq = jest.fn();
const mockSingle = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: {
  auth: { getSession: () => mockSession() },
  from: () => ({ update: mockUpdate }),
} }));
beforeEach(() => {
  jest.clearAllMocks();
  authEvents.emitLogin('owner');
  mockSession.mockResolvedValue({ data: { session: { user: { id: 'owner' } } }, error: null });
  mockUpdate.mockReturnValue({ eq: mockEq });
  mockEq.mockReturnValue({ eq: mockEq, select: () => ({ single: mockSingle }) });
  mockSingle.mockResolvedValue({ data: { id: 'item' }, error: null });
});

test('persists only editable fields, maps DB columns and requires owner/id confirmation', async () => {
  await expect(saveClothingEdits('item', { category: 'bottom', subCategory: 'bottoms', primaryColor: 'Black',
    colorHex: '#000000', seasons: ['winter'], occasions: ['work'], userId: 'forged-owner', wearCount: 999 })).resolves.toEqual({ id: 'item' });
  expect(mockUpdate).toHaveBeenCalledWith(expect.objectContaining({ category: 'bottom', sub_category: 'bottoms', type: 'bottoms', primary_color: 'Black', color_hex: '#000000', seasons: ['winter'], occasions: ['work'] }));
  expect(mockUpdate.mock.calls[0][0]).not.toHaveProperty('user_id');
  expect(mockUpdate.mock.calls[0][0]).not.toHaveProperty('wear_count');
  expect(mockEq).toHaveBeenCalledWith('user_id', 'owner');
  expect(mockEq).toHaveBeenCalledWith('id', 'item');
});

test('returned Supabase errors reject and zero-row results never count as success', async () => {
  mockSingle.mockResolvedValueOnce({ data: null, error: new Error('denied') });
  await expect(saveClothingEdits('item', {})).rejects.toThrow('denied');
  mockSingle.mockResolvedValueOnce({ data: null, error: null });
  await expect(saveClothingEdits('item', {})).rejects.toThrow('not confirmed');
});

test('missing session does not issue a table update', async () => {
  mockSession.mockResolvedValueOnce({ data: { session: null }, error: null });
  await expect(saveClothingEdits('item', {})).rejects.toThrow('Signed-in');
  expect(mockUpdate).not.toHaveBeenCalled();
});
