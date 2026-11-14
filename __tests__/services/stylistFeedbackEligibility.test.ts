/**
 * The screen must not offer a feedback control the stylist_look_feedback
 * constraints would reject: look_id ~ '^outfit_[1-3]$', concept_id slug,
 * bounded engine_version, and a request id.
 *
 * The write payload additionally carries the gate-D linking columns
 * (item_ids, look_signature) added by migration 20261113120000, bounded and
 * degraded client-side so a malformed look can never fail the upsert.
 */
import { canRecordLookFeedback, recordLookFeedback, lookFeedbackItemIds, buildLookFeedbackSignature } from '../../src/services/stylistFeedback';

jest.mock('../../lib/supabase', () => {
  const upsert = jest.fn();
  const from = jest.fn(() => ({ upsert }));
  return { supabase: { from }, __upsert: upsert };
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { supabase } = require('../../lib/supabase') as typeof import('../../lib/supabase');
const upsertMock = (supabase.from('stylist_look_feedback') as unknown as { upsert: jest.Mock }).upsert;

const valid = {
  id: 'outfit_1',
  concept: { id: 'clean-lines' },
  generationRequestId: '3f1d0f14-6d3e-4f7a-9e0d-2d8f7c1b5a44',
  engineVersion: 'stylist-5-integrated',
};

describe('look feedback eligibility', () => {
  it('accepts exactly the server-issued look shape', () => {
    expect(canRecordLookFeedback(valid)).toBe(true);
    expect(canRecordLookFeedback({ ...valid, id: 'outfit_3' })).toBe(true);
  });

  it('rejects recovered, prefixed, and locally renumbered looks', () => {
    expect(canRecordLookFeedback({ ...valid, id: 'recovered_1' })).toBe(false);
    expect(canRecordLookFeedback({ ...valid, id: 'look_0' })).toBe(false);
    expect(canRecordLookFeedback({ ...valid, id: 'outfit_4' })).toBe(false);
  });

  it('rejects missing provenance and out-of-contract concept ids', () => {
    expect(canRecordLookFeedback({ ...valid, generationRequestId: undefined })).toBe(false);
    expect(canRecordLookFeedback({ ...valid, engineVersion: '' })).toBe(false);
    expect(canRecordLookFeedback({ ...valid, concept: undefined })).toBe(false);
    expect(canRecordLookFeedback({ ...valid, concept: { id: 'Clean Lines!' } })).toBe(false);
  });

  it('never writes an ineligible look', async () => {
    await expect(recordLookFeedback({ id: 'recovered_1' } as any, 1)).resolves.toBeUndefined();
    await expect(recordLookFeedback({ ...valid, generationRequestId: undefined }, -1)).resolves.toBeUndefined();
    expect(upsertMock).not.toHaveBeenCalled();
  });
});

describe('look feedback write payload (gate D linking columns)', () => {
  beforeEach(() => {
    upsertMock.mockReset();
    upsertMock.mockResolvedValue({ error: null });
  });

  it('writes item ids and the styling-feature signature with the rating', async () => {
    const look = {
      ...valid,
      direction: { id: 'signature', qualified: true },
      items: [
        { id: 'shop-sku-1', macroCategory: 'top', color: 'white', name: 'white cotton shirt' },
        { id: 'shop-sku-2', macroCategory: 'bottom', color: 'grey', name: 'grey tailored trousers' },
        { id: 'shop-sku-3', macroCategory: 'shoes', color: 'brown', name: 'brown leather derbies' },
      ],
    };
    await recordLookFeedback(look, 1);
    expect(upsertMock).toHaveBeenCalledTimes(1);
    const [payload, options] = upsertMock.mock.calls[0];
    expect(payload).toMatchObject({
      request_id: valid.generationRequestId,
      look_id: 'outfit_1',
      concept_id: 'clean-lines',
      engine_version: 'stylist-5-integrated',
      rating: 1,
      reason: 'like',
      item_ids: ['shop-sku-1', 'shop-sku-2', 'shop-sku-3'],
    });
    expect(payload.look_signature).toMatchObject({
      version: 1,
      direction: 'signature',
    });
    expect(String(payload.look_signature.signature)).toMatch(/^sq1:/);
    expect(payload.look_signature.features).toHaveLength(4);
    expect(options).toEqual({ onConflict: 'user_id,request_id,look_id' });
  });

  it('bounds and de-dupes item ids, dropping non-string entries', () => {
    const ids = lookFeedbackItemIds({
      ...valid,
      items: [
        { id: ' a ' }, { id: 'a' }, { id: 42 }, { id: null }, {},
        ...Array.from({ length: 20 }, (_, index) => ({ id: `sku-${index}` })),
      ],
    });
    expect(ids[0]).toBe('a');
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.length).toBeLessThanOrEqual(8);
    expect(ids.every((id) => typeof id === 'string' && id.length > 0)).toBe(true);
  });

  it('degrades to null signature pieces for a look without usable items', async () => {
    const signature = buildLookFeedbackSignature(valid);
    expect(signature).toEqual({ version: 1, direction: null, signature: null, features: [] });
    await recordLookFeedback(valid, -1);
    const [payload] = upsertMock.mock.calls[0];
    expect(payload.item_ids).toBeUndefined();
    expect(payload.look_signature).toEqual({ version: 1, direction: null, signature: null, features: [] });
    expect(payload.rating).toBe(-1);
    expect(payload.reason).toBe('coherence');
  });

  it('keeps the write working for hostile item shapes', async () => {
    const hostile = {
      ...valid,
      items: [{ id: 'shop-sku-1', get name(): string { throw new Error('boom'); } }, 42, null],
      direction: { id: 'refresh' },
    } as any;
    const signature = buildLookFeedbackSignature(hostile);
    expect(signature.version).toBe(1);
    expect(signature.direction).toBe('refresh');
    await expect(recordLookFeedback(hostile as any, 1)).resolves.toBeUndefined();
    expect(upsertMock).toHaveBeenCalledTimes(1);
    const [payload] = upsertMock.mock.calls[0];
    expect(payload.look_signature.version).toBe(1);
    expect(payload.item_ids).toEqual(['shop-sku-1']);
  });

  it('surfaces storage failures as feedback_unavailable', async () => {
    upsertMock.mockResolvedValue({ error: { message: 'rls denied' } });
    await expect(recordLookFeedback(valid, 1)).rejects.toThrow('feedback_unavailable');
  });
});
