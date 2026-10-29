import { recordShownOutfits } from '../../src/services/recentOutfitIds';
jest.mock('../../src/services/recentOutfitIds', () => ({ recordShownOutfits: jest.fn() }));
import { act, renderHook, waitFor } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useDailyAIOutfit } from '../../hooks/useDailyAIOutfit';
import { generateOutfitsFromDB } from '../../src/services/outfitGenerationService';
import { dailyOutfitCacheKey } from '../../src/services/dailyOutfitCache';

let mockUserId = 'user-a';
const mockPreferences = {};
jest.mock('../../store/auth', () => ({ __esModule: true, default: (select: any) => select({ user: { id: mockUserId } }) }));
jest.mock('../../store/stylePreferenceStore', () => ({ useStylePreferenceStore: (select: any) => select({ preferences: mockPreferences }) }));
jest.mock('expo-image', () => ({ Image: { prefetch: jest.fn().mockResolvedValue(undefined) } }));
jest.mock('../../src/services/outfitGenerationService', () => ({ generateOutfitsFromDB: jest.fn(), isShopOnlyDeck: (style: string) => style === 'business_casual' }));
const generate = generateOutfitsFromDB as jest.Mock;
const deck = (id: string) => [{ id, description: 'A look', style: 'casual', occasion: 'everyday', items: [
  { id: `${id}-top`, name: 'Cotton shirt', macroCategory: 'top' },
  { id: `${id}-bottom`, name: 'Navy trousers', macroCategory: 'bottom' },
  { id: `${id}-shoes`, name: 'Brown loafers', macroCategory: 'shoes' },
] }];
const options = { style: 'casual', occasion: 'everyday', variants: 15 };
function deferred() {
  let resolve!: (value: any) => void;
  const promise = new Promise<any>(r => { resolve = r; });
  return { promise, resolve };
}
beforeEach(() => {
  jest.clearAllMocks();
  mockUserId = 'user-a';
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  generate.mockResolvedValue({ success: true, outfits: deck('first') });
});
it('keeps exact products, caches a smaller valid deck and does not fabricate weather', async () => {
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.outfits).toEqual(deck('first')));
  expect(generate).toHaveBeenCalledWith(expect.objectContaining({ weather: undefined, limit: 15 }));
  expect(AsyncStorage.setItem).toHaveBeenCalledWith(dailyOutfitCacheKey('user-a', 'casual', 'everyday'), expect.any(String));
});
it('serves a smaller valid cache without generating replacement products', async () => {
  const date = new Date();
  const dateKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify({ ...options, dateKey, outfits: deck('cached') }));
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.outfits).toEqual(deck('cached')));
  expect(generate).not.toHaveBeenCalled();
});
it('shows a generation failure without constructing a static deck', async () => {
  generate.mockResolvedValue({ success: false, error: 'no_valid_outfits', outfits: [] });
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.error).toBe('no_valid_outfits'));
  expect(result.current.outfits).toEqual([]);
  expect(AsyncStorage.setItem).not.toHaveBeenCalled();
});
it('ignores a previous account response after switching accounts', async () => {
  const old = deferred();
  generate.mockReturnValueOnce(old.promise).mockResolvedValueOnce({ success: true, outfits: deck('new-user') });
  const { result, rerender } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(generate).toHaveBeenCalledTimes(1));
  mockUserId = 'user-b';
  rerender(undefined);
  await waitFor(() => expect(result.current.outfits).toEqual(deck('new-user')));
  await act(async () => old.resolve({ success: true, outfits: deck('old-user') }));
  expect(result.current.outfits).toEqual(deck('new-user'));
  expect(recordShownOutfits).not.toHaveBeenCalledWith(deck('old-user'), 'user-a');
  expect(AsyncStorage.setItem).not.toHaveBeenCalledWith(dailyOutfitCacheKey('user-a', 'casual', 'everyday'), expect.anything());
});
it('coalesces rapid regenerate taps and keeps the displayed deck while pending', async () => {
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.loading).toBe(false));
  await waitFor(() => expect(result.current.outfits).toHaveLength(1));
  const next = deferred();
  generate.mockReturnValueOnce(next.promise);
  act(() => { void result.current.regenerate(); void result.current.regenerate(); });
  expect(generate).toHaveBeenCalledTimes(2);
  expect(result.current.outfits).toEqual(deck('first'));
  await act(async () => next.resolve({ success: true, outfits: deck('next') }));
  expect(result.current.outfits).toEqual(deck('next'));
});

it('rejects owned clothing for shop-only sections even in a small deck', async () => {
  const { result } = renderHook(() => useDailyAIOutfit({ ...options, style: 'business_casual', variants: 3 }));
  await waitFor(() => expect(result.current.error).toBe('no_valid_outfits_for_request'));
  expect(result.current.outfits).toEqual([]);
  expect(recordShownOutfits).not.toHaveBeenCalled();
});
it('separates caches for two occasions with the same style', () => {
  expect(dailyOutfitCacheKey('user', 'classic', 'work')).not.toBe(dailyOutfitCacheKey('user', 'classic', 'date'));
});

it('keeps the localized budget explanation when generation fails', async () => {
  generate.mockResolvedValue({ success: false, error: 'stylist_generation_failed', errorTranslationKey: 'outfitMaker.budgetTooLow', outfits: [] });
  const { result } = renderHook(() => useDailyAIOutfit({ style: 'casual', occasion: 'Everyday' }));
  await waitFor(() => expect(result.current.error).toBe('outfitMaker.budgetTooLow'));
  expect(result.current.loading).toBe(false);
  expect(result.current.outfits).toEqual([]);
});
it('free users without a daily cache see the subscription requirement without an AI request', async () => {
  const { result } = renderHook(() => useDailyAIOutfit({ ...options, allowGeneration: false }));
  await waitFor(() => expect(result.current.error).toBe('outfitMaker.subscriptionRequired'));
  expect(generate).not.toHaveBeenCalled();
  await act(async () => { await result.current.regenerate(); });
  expect(generate).not.toHaveBeenCalled();
});
it('generation-locked users can still view their valid daily cache', async () => {
  const date = new Date();
  const dateKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify({ ...options, dateKey, outfits: deck('cached') }));
  const { result } = renderHook(() => useDailyAIOutfit({ ...options, allowGeneration: false }));
  await waitFor(() => expect(result.current.outfits).toEqual(deck('cached')));
  expect(generate).not.toHaveBeenCalled();
});
it('upgrading unlocks generation without leaving Home', async () => {
  const { result, rerender } = renderHook<ReturnType<typeof useDailyAIOutfit>, { allowed: boolean }>(({ allowed }) => useDailyAIOutfit({ ...options, allowGeneration: allowed }), { initialProps: { allowed: false } });
  await waitFor(() => expect(result.current.error).toBe('outfitMaker.subscriptionRequired'));
  rerender({ allowed: true });
  await waitFor(() => expect(result.current.outfits).toEqual(deck('first')));
  expect(generate).toHaveBeenCalledTimes(1);
});

it('retains the same deck without reloading when Home loses and regains focus', async () => {
  const { result, rerender } = renderHook<ReturnType<typeof useDailyAIOutfit>, { enabled: boolean }>(
    ({ enabled }) => useDailyAIOutfit({ ...options, enabled }),
    { initialProps: { enabled: true } },
  );
  await waitFor(() => expect(result.current.outfits).toEqual(deck('first')));
  const originalDeck = result.current.outfits;
  rerender({ enabled: false });
  expect(result.current.outfits).toBe(originalDeck);
  rerender({ enabled: true });
  expect(result.current.outfits).toBe(originalDeck);
  expect(generate).toHaveBeenCalledTimes(1);
  expect(AsyncStorage.getItem).toHaveBeenCalledTimes(1);
});
