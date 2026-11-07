import AsyncStorage from '@react-native-async-storage/async-storage';
const mockInvoke = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: { functions: { invoke: (...args: unknown[]) => mockInvoke(...args) } },
}));
import { analyticsService } from '../../src/services/analyticsService';
import { sanitizeAnalyticsProperties } from '../../src/services/launch/telemetry';
beforeEach(async () => {
  jest.useFakeTimers();
  await AsyncStorage.clear();
  await analyticsService.setEnabled(false);
  await analyticsService.setEnabled(true);
  mockInvoke.mockReset();
});
afterEach(() => {
  analyticsService.destroy();
  jest.useRealTimers();
});
test('runtime payloads include schema versions and exclude prohibited data', async () => {
  expect(
    sanitizeAnalyticsProperties({
      images: ['raw'],
      bodyProfile: { waist: 70 },
      message: 'token',
      source: 'local',
      requestId: 'https://private',
      schemaVersion: 99,
    })
  ).toEqual({ schemaVersion: 1, source: 'local' });
  analyticsService.trackEvent('outfit_saved', { source: 'ai-maker', bodyData: 'private' });
  const queue = await analyticsService.getQueue();
  expect(queue).toHaveLength(1);
  expect(queue[0].schemaVersion).toBe(1);
  expect(queue[0].properties).toEqual({ schemaVersion: 1, source: 'ai-maker' });
});
test('Edge error responses retain events; successful retry preserves concurrent new events', async () => {
  analyticsService.trackEvent('buycheck_completed', { source: 'local' });
  mockInvoke.mockResolvedValueOnce({ data: null, error: new Error('offline') });
  await analyticsService.flush();
  expect(await analyticsService.getQueue()).toHaveLength(1);
  mockInvoke.mockImplementationOnce(async () => {
    analyticsService.trackEvent('outfit_saved');
    return { data: { success: true }, error: null };
  });
  await analyticsService.flush();
  expect((await analyticsService.getQueue()).map((event) => event.name)).toEqual(['outfit_saved']);
  expect(mockInvoke.mock.calls[0][1].body.events[0].eventId).toBe(
    mockInvoke.mock.calls[1][1].body.events[0].eventId
  );
});
test('shared Edge sanitizer handles malformed objects and retains dedupe-safe metadata', () => {
  expect(sanitizeAnalyticsProperties(null as any)).toEqual({ schemaVersion: 1 });
  expect(
    sanitizeAnalyticsProperties({
      requestId: 'request-1',
      helpful: true,
      candidateCount: 7,
      rawImage: 'sensitive',
      waist: 80,
    })
  ).toEqual({ schemaVersion: 1, requestId: 'request-1', helpful: true, candidateCount: 7 });
});
