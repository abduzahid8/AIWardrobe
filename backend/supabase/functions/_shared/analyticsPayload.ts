/** Analytics payload allowlist. No free text, raw images, body data or URLs. */
const fields = new Set([
  'method',
  'category',
  'source',
  'tier',
  'screen',
  'errorType',
  'feature',
  'issueType',
  'requestId',
  'traceId',
  'outcome',
  'confidence',
  'fallback',
  'ruleVersion',
  'correctionField',
]);
const numeric = new Set(['price', 'latencyMs', 'candidateCount', 'schemaVersion']);
export function sanitizeAnalyticsProperties(
  input: Record<string, unknown> = {}
): Record<string, unknown> {
  const clean: Record<string, unknown> = { schemaVersion: 1 };
  for (const [key, value] of Object.entries(
    input && typeof input === 'object' && !Array.isArray(input) ? input : {}
  )) {
    if (fields.has(key) && typeof value === 'string' && /^[a-zA-Z0-9_. -]{1,96}$/.test(value))
      clean[key] = value;
    if (
      numeric.has(key) &&
      key !== 'schemaVersion' &&
      typeof value === 'number' &&
      Number.isFinite(value) &&
      value >= 0
    )
      clean[key] = value;
    if (
      ['helpful', 'fallbackUsed', 'hardViolation', 'duplicate'].includes(key) &&
      typeof value === 'boolean'
    )
      clean[key] = value;
  }
  return clean;
}
