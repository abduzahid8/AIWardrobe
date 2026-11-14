jest.mock('../../hooks/useNetworkRecovery', () => ({ useNetworkRecovery: jest.fn() }));
import { recordShownOutfits } from '../../src/services/recentOutfitIds';
jest.mock('../../src/services/recentOutfitIds', () => ({ recordShownOutfits: jest.fn() }));
import { act, renderHook, waitFor } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useDailyAIOutfit, clearDailyOutfitSessionCache } from '../../hooks/useDailyAIOutfit';
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
  { id: `${id}-outer`, name: 'Navy jacket', macroCategory: 'outerwear' },
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
  clearDailyOutfitSessionCache();
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

it('keeps the loaded deck when Home loses and regains focus', async () => {
  const { result, rerender } = renderHook<ReturnType<typeof useDailyAIOutfit>, { enabled: boolean }>(
    ({ enabled }) => useDailyAIOutfit({ ...options, enabled }),
    { initialProps: { enabled: true } },
  );
  await waitFor(() => expect(result.current.outfits).toEqual(deck('first')));
  const originalDeck = result.current.outfits;
  rerender({ enabled: false });
  rerender({ enabled: true });
  expect(result.current.outfits).toBe(originalDeck);
  expect(generate).toHaveBeenCalledTimes(1);
  expect(AsyncStorage.getItem).toHaveBeenCalledTimes(1);
});

it('reuses an in-flight initial load when Home regains focus', async () => {
  const pending = deferred();
  generate.mockReturnValueOnce(pending.promise);
  const { result, rerender } = renderHook<ReturnType<typeof useDailyAIOutfit>, { enabled: boolean }>(
    ({ enabled }) => useDailyAIOutfit({ ...options, enabled }),
    { initialProps: { enabled: true } },
  );
  await waitFor(() => expect(generate).toHaveBeenCalledTimes(1));
  rerender({ enabled: false });
  rerender({ enabled: true });
  expect(generate).toHaveBeenCalledTimes(1);
  await act(async () => pending.resolve({ success: true, outfits: deck('resumed') }));
  expect(result.current.outfits).toEqual(deck('resumed'));
});


it('replaces an old three-piece cache with a valid four-piece deck', async () => {
  const date = new Date();
  const dateKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  const incomplete = deck('old').map(outfit => ({ ...outfit, items: outfit.items.slice(1) }));
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify({ ...options, dateKey, outfits: incomplete }));
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.outfits).toEqual(deck('first')));
  expect(generate).toHaveBeenCalledTimes(1);
  expect(result.current.outfits[0].items).toHaveLength(4);
});

it('rejects a generated three-piece outfit instead of displaying it', async () => {
  const incomplete = deck('bad').map(outfit => ({ ...outfit, items: outfit.items.slice(1) }));
  generate.mockResolvedValue({ success: true, outfits: incomplete });
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.error).toBe('no_valid_outfits_for_request'));
  expect(result.current.outfits).toEqual([]);
});


it('shows the validated deck immediately after the Home section remounts', async () => {
  const first = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(first.result.current.outfits).toEqual(deck('first')));
  const loaded = first.result.current.outfits;
  first.unmount();
  const second = renderHook(() => useDailyAIOutfit(options));
  expect(second.result.current.outfits).toBe(loaded);
  expect(second.result.current.loading).toBe(false);
  expect(generate).toHaveBeenCalledTimes(1);
  expect(AsyncStorage.getItem).toHaveBeenCalledTimes(1);
});

it('revalidates a session deck against changed weather before showing it on remount', async () => {
  const first = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(first.result.current.outfits).toEqual(deck('first')));
  first.unmount();
  const second = renderHook(() => useDailyAIOutfit({ ...options, weather: { temp: -5, condition: 'snow' } }));
  expect(second.result.current.outfits).toEqual([]);
  await waitFor(() => expect(generate).toHaveBeenCalledTimes(2));
});

it('hides a warm-weather deck when weather turns freezing without remounting Home', async () => {
  const { result, rerender } = renderHook<{ outfits: any[]; loading: boolean; error: string | null; regenerate: () => Promise<void> }, { weather?: { temp: number; condition: string } }>(
    ({ weather }) => useDailyAIOutfit({ ...options, weather }),
    { initialProps: { weather: undefined } },
  );
  await waitFor(() => expect(result.current.outfits).toEqual(deck('first')));
  rerender({ weather: { temp: -5, condition: 'snow' } });
  expect(result.current.outfits).toEqual([]);
  await waitFor(() => expect(generate).toHaveBeenCalledTimes(2));
});

it('does not reuse a session deck for a different account', async () => {
  const first = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(first.result.current.outfits).toEqual(deck('first')));
  first.unmount();
  mockUserId = 'user-b';
  generate.mockResolvedValue({ success: true, outfits: deck('second') });
  const second = renderHook(() => useDailyAIOutfit(options));
  expect(second.result.current.outfits).toEqual([]);
  await waitFor(() => expect(second.result.current.outfits).toEqual(deck('second')));
});

it('shows Team Collaboration only when all four pieces are shop clothing', async () => {
  const shopDeck = deck('shop-look').map(outfit => ({ ...outfit, items: outfit.items.map(item => ({ ...item, isShopItem: true })) }));
  const mixedDeck = deck('mixed-look').map(outfit => ({ ...outfit, items: outfit.items.map((item, index) => ({ ...item, isShopItem: index !== 0 })) }));
  generate.mockResolvedValue({ success: true, outfits: [...mixedDeck, ...shopDeck] });
  const { result } = renderHook(() => useDailyAIOutfit({ style: 'business_casual', occasion: 'Team Collaboration', variants: 6 }));
  await waitFor(() => expect(result.current.outfits).toEqual(shopDeck));
  expect(result.current.outfits[0].items).toHaveLength(4);
  expect(result.current.outfits[0].items.every(item => item.isShopItem)).toBe(true);
});

it('retries a failed first deck after connectivity returns', async () => {
  const recovery = require('../../hooks/useNetworkRecovery').useNetworkRecovery as jest.Mock;
  generate.mockResolvedValueOnce({ success: false, error: 'network_error' })
    .mockResolvedValueOnce({ success: true, outfits: deck('recovered') });
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(result.current.error).toBe('network_error'));
  act(() => recovery.mock.calls[recovery.mock.calls.length - 1][0]());
  await waitFor(() => expect(result.current.outfits).toEqual(deck('recovered')));
  expect(generate).toHaveBeenCalledTimes(2);
});


it('retries if connectivity recovers while the first request is still failing', async () => {
  const recovery = require('../../hooks/useNetworkRecovery').useNetworkRecovery as jest.Mock;
  const pending = deferred();
  generate.mockReturnValueOnce(pending.promise).mockResolvedValueOnce({success:true,outfits:deck('recovered-late')});
  const { result } = renderHook(() => useDailyAIOutfit(options));
  await waitFor(() => expect(generate).toHaveBeenCalledTimes(1));
  act(() => recovery.mock.calls[recovery.mock.calls.length - 1][0]());
  await act(async () => pending.resolve({success:false,error:'network_error'}));
  await waitFor(() => expect(result.current.outfits).toEqual(deck('recovered-late')));
  expect(generate).toHaveBeenCalledTimes(2);
});
test('quality-rule updates cannot reuse an old daily deck cache',()=>{
 expect(dailyOutfitCacheKey('user','casual','Weekend')).toContain('v21_stylist_quality');
});
