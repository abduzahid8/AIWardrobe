import {
  createBetaEmitter,
  feedbackEvent,
  makeBetaFeedback,
  revertBetaCorrection,
  sanitizeResultDiagnostics,
} from '../../src/services/launch/validationLoop';
const record = () =>
  makeBetaFeedback({
    appVersion: '3.0.2',
    feature: 'buycheck',
    helpful: false,
    issueType: 'wrong_fit',
    comment: 'private comment',
    diagnostics: { requestId: 'req-1', confidence: 'low', candidateCount: 4 },
    correction: { field: 'fit', before: 'snug', after: 'relaxed' },
  });
test('result exports use an allowlist and reject raw images, body data and unknown diagnostic keys at runtime', () => {
  const raw = {
    requestId: 'https://user:secret@host/image',
    traceId: 'someone@example.com',
    confidence: 'high',
    candidateCount: -1,
    images: ['base64-secret'],
    bodyProfile: { waist: 75 },
    chainOfThought: 'private',
    featureFlags: { visual_parser: false, secret: 'key' },
  };
  expect(sanitizeResultDiagnostics(raw as any)).toEqual({
    confidence: 'high',
    featureFlags: { visual_parser: false },
  });
  const event = feedbackEvent(record());
  expect(event.schemaVersion).toBe(1);
  expect(JSON.stringify(event)).not.toMatch(/private comment|relaxed|snug|bodyProfile|images/);
});
test('correction undo is auditable and preserves the original export', () => {
  const original = record(),
    undone = revertBetaCorrection(original);
  expect(original.correction?.revertedAt).toBeUndefined();
  expect(undone.correction).toMatchObject({
    field: 'fit',
    before: 'snug',
    after: 'relaxed',
    revertedAt: expect.any(String),
  });
  expect(feedbackEvent(undone, true).name).toBe('correction_reverted');
});
test('failing telemetry is harmless, retryable, deduplicated and reconstructed without extra fields', async () => {
  const sink = jest.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(undefined);
  const emit = createBetaEmitter(sink),
    event = { ...feedbackEvent(record()), rawImages: 'forbidden' };
  expect(await emit(event)).toBe(false);
  expect(await emit(event)).toBe(true);
  expect(await emit(event)).toBe(false);
  expect(sink).toHaveBeenCalledTimes(2);
  expect(sink.mock.calls[1][0].rawImages).toBeUndefined();
  expect(sink.mock.calls[1][0].schemaVersion).toBe(1);
});
test('concurrent duplicate outcomes are delivered once', async () => {
  const sink = jest.fn(async () => {}),
    emit = createBetaEmitter(sink),
    event = feedbackEvent(record());
  await Promise.all([emit(event), emit(event)]);
  expect(sink).toHaveBeenCalledTimes(1);
});
