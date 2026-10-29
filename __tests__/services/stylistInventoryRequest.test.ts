/** @jest-environment node */
import { requestWithInventoryRotation } from '../../src/services/stylistInventoryRequest';

test('exhausted rotation retries once against real inventory', async () => {
  const result = { data: { success: true, outfits: [{ id: 'real-look' }] }, error: null };
  const invoke = jest.fn().mockResolvedValueOnce({ data: null, error: {
    context: new Response(JSON.stringify({ code: 'no_matching_clothing', missingSlots: ['shoes'] }), { status: 422 }),
  } }).mockResolvedValueOnce(result);
  expect(await requestWithInventoryRotation(invoke, ['shop_seen-shoes'])).toBe(result);
  expect(invoke.mock.calls).toEqual([[['shop_seen-shoes']], [[]]]);
});

test.each(['no_outfit_within_budget', 'invalid_generation_request', 'ai_generation_unavailable'])('%s does not relax rotation or retry', async code => {
  const invoke = jest.fn().mockResolvedValue({ data: { success: false, code }, error: null });
  await requestWithInventoryRotation(invoke, ['seen']);
  expect(invoke).toHaveBeenCalledTimes(1);
});

test('a genuinely empty catalog gets no retry loop', async () => {
  const invoke = jest.fn().mockResolvedValue({ data: { success: false, code: 'no_matching_clothing' }, error: null });
  await requestWithInventoryRotation(invoke, ['seen']);
  expect(invoke).toHaveBeenCalledTimes(2);
  invoke.mockClear();
  await requestWithInventoryRotation(invoke, []);
  expect(invoke).toHaveBeenCalledTimes(1);
});

test('cancellation during the first request prevents rotation retry', async () => {
  const controller = new AbortController();
  const invoke = jest.fn(async () => {
    controller.abort();
    return { data: { success: false, code: 'no_matching_clothing' }, error: null };
  });
  await requestWithInventoryRotation(invoke, ['seen'], controller.signal);
  expect(invoke).toHaveBeenCalledTimes(1);
});
