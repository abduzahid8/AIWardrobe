/**
 * The screen must not offer a feedback control the stylist_look_feedback
 * constraints would reject: look_id ~ '^outfit_[1-3]$', concept_id slug,
 * bounded engine_version, and a request id.
 */
import { canRecordLookFeedback, recordLookFeedback } from '../../src/services/stylistFeedback';

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
  });
});
