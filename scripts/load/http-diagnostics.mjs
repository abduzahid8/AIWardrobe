/** Allow-listed response metadata only; never record request headers, URLs or bodies. */
export function responseDiagnostics(response) {
  const result = {};
  for (const [header, field, pattern] of [
    ['sb-request-id', 'supabaseRequestId', /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i],
    ['cf-ray', 'cloudflareRay', /^[a-f0-9]{8,32}-[a-z]{3}$/i],
  ]) {
    const value = response.headers.get(header);
    if (value && pattern.test(value)) result[field] = value;
  }
  const raw = response.headers.get('x-envoy-upstream-service-time');
  if (raw !== null && /^\d+(\.\d+)?$/.test(raw) && Number(raw) <= 120000) result.upstreamMs = Number(raw);
  return result;
}
export function timingSummary(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const percentile = p => ordered.length ? ordered[Math.min(ordered.length - 1, Math.ceil(ordered.length * p) - 1)] : null;
  return { samples: ordered.length, p50: percentile(.5), p95: percentile(.95), p99: percentile(.99) };
}
