/** One supported JSON-mode call. This is a ranker, never a garment generator.
 * NVIDIA Nemotron 3.5 supports json_object and enable_thinking:false; no
 * speculative json_schema/tool contract or legacy outfit parser is involved.
 */
export interface RankingStatus {
  provider: 'nvidia';
  model: string;
  status: string;
  expected?: number;
  received?: number;
  duplicates?: number;
  unknown?: number;
  /** Set only when a contract-violating response was safely repaired. */
  repaired?: boolean;
  droppedUnknown?: number;
  filledMissing?: number;
  finishReason?: string;
}
/** Extract the first complete JSON object from a response that may carry
 * surrounding prose or code fences. Returns null when no object is found. */
function extractJsonObject(raw: string): string | null {
  const text = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/, '');
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return null;
  return text.slice(start, end + 1);
}
export function validateLookRanking(raw: string, allowedIds: readonly string[]) {
  let parsed: unknown;
  try {
    const body = extractJsonObject(raw);
    if (!body) return { ids: null, status: 'invalid_json' };
    parsed = JSON.parse(body);
  } catch {
    return { ids: null, status: 'invalid_json' };
  }
  if (
    !parsed ||
    typeof parsed !== 'object' ||
    Array.isArray(parsed) ||
    Object.keys(parsed).length !== 1 ||
    !('rankedLookIds' in parsed) ||
    !Array.isArray(parsed.rankedLookIds) ||
    parsed.rankedLookIds.some((id) => typeof id !== 'string')
  )
    return { ids: null, status: 'invalid_schema' };
  const ids = parsed.rankedLookIds as string[],
    allowed = new Set(allowedIds);
  const duplicates = ids.length - new Set(ids).size,
    unknown = ids.filter((id) => !allowed.has(id)).length;
  const details = { expected: allowed.size, received: ids.length, duplicates, unknown };
  if (ids.length !== allowed.size || duplicates || unknown)
    return { ids: null, status: 'rank_contract_mismatch', ...details };
  return { ids, status: 'ok', ...details };
}
/**
 * Safe deterministic recovery for contract-violating ranking responses.
 *
 * - Preserves every valid unique ID in the provider's own best-first order.
 * - Never accepts unknown IDs or invented products: they are dropped, not stored.
 * - Fills missing positions only from genuinely eligible shortlist members
 *   (the `allowedIds` order, which is the deterministic scorer's order), so
 *   the repaired tail is exactly what the fallback would have produced.
 * - Refuses repair when the response carries too little provider signal
 *   (fewer than half the shortlist as valid unique IDs) or no usable IDs at
 *   all — those remain a full deterministic fallback.
 *
 * Returns the repaired full-length ordering with status `repaired_hybrid`,
 * or `{ ids: null, status }` with the original failure status when the
 * response cannot be safely repaired.
 */
export function repairLookRanking(raw: string, allowedIds: readonly string[]) {
  const validation = validateLookRanking(raw, allowedIds);
  if (validation.ids) return { ...validation, repaired: false as const };
  if (validation.status !== 'rank_contract_mismatch') return { ...validation, repaired: false as const };
  let parsed: { rankedLookIds: string[] };
  try {
    const body = extractJsonObject(raw);
    if (!body) return { ...validation, repaired: false as const };
    parsed = JSON.parse(body) as { rankedLookIds: string[] };
  } catch {
    return { ...validation, repaired: false as const };
  }
  if (!parsed || !Array.isArray(parsed.rankedLookIds)) return { ...validation, repaired: false as const };
  const allowed = new Set(allowedIds);
  const seen = new Set<string>();
  const ordered: string[] = [];
  let droppedUnknown = 0;
  for (const id of parsed.rankedLookIds) {
    if (typeof id !== 'string' || !allowed.has(id)) {
      droppedUnknown += 1;
      continue;
    }
    if (seen.has(id)) continue;
    seen.add(id);
    ordered.push(id);
  }
  // Too little provider signal to preserve: fall back fully rather than
  // amplifying one or two noisy positions into a full ordering.
  if (ordered.length === 0 || ordered.length < Math.ceil(allowed.size / 2)) {
    return { ...validation, repaired: false as const };
  }
  const filled: string[] = [];
  for (const id of allowedIds) {
    if (!seen.has(id)) {
      seen.add(id);
      filled.push(id);
    }
  }
  const ids = [...ordered, ...filled];
  if (ids.length !== allowed.size) return { ...validation, repaired: false as const };
  return {
    ids,
    status: 'repaired_hybrid',
    expected: allowed.size,
    received: parsed.rankedLookIds.length,
    duplicates: parsed.rankedLookIds.length - new Set(parsed.rankedLookIds).size,
    unknown: droppedUnknown,
    repaired: true as const,
    droppedUnknown,
    filledMissing: filled.length,
  };
}
export function nvidiaRankingBody(
  model: string,
  system: string,
  payload: unknown,
  lookIds: readonly string[]
) {
  return {
    model,
    messages: [
      {
        role: 'system',
        content:
          system +
          '\nReturn exactly one JSON object with only rankedLookIds, an array of ALL ' +
          lookIds.length +
          ' supplied IDs in best-first order. Each supplied ID must occur exactly once: never repeat, omit, or invent an ID. Copy IDs literally. No product IDs. Required ID set: ' +
          JSON.stringify(lookIds),
      },
      { role: 'user', content: JSON.stringify(payload) },
    ],
    temperature: 0.1,
    max_tokens: 700,
    stream: false,
    response_format: { type: 'json_object' },
    ...(model.includes('nemotron') ? { chat_template_kwargs: { enable_thinking: false } } : {}),
  };
}
export async function rankNvidiaLooks(
  key: string,
  model: string,
  system: string,
  payload: unknown,
  lookIds: readonly string[],
  fetcher: typeof fetch = fetch
): Promise<{ ids: string[] | null; attempt: RankingStatus }> {
  const controller = new AbortController();
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
      reject(Error('ranking_timeout'));
    }, 12000);
  });
  const bounded = <T>(p: Promise<T>) => Promise.race([p, deadline]);
  try {
    const response = await bounded(
      fetcher('https://integrate.api.nvidia.com/v1/chat/completions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(nvidiaRankingBody(model, system, payload, lookIds)),
        signal: controller.signal,
      })
    );
    if (!response.ok) {
      await response.body?.cancel();
      return { ids: null, attempt: { provider: 'nvidia', model, status: String(response.status) } };
    }
    const data = await bounded(response.json());
    const choice = data.choices?.[0];
    const content = typeof choice?.message?.content === 'string' ? choice.message.content : '';
    const result = repairLookRanking(content, lookIds);
    return {
      ids: result.ids,
      attempt: {
        provider: 'nvidia',
        model,
        status: result.status,
        ...('expected' in result
          ? {
              expected: result.expected,
              received: result.received,
              duplicates: result.duplicates,
              unknown: result.unknown,
            }
          : {}),
        ...('repaired' in result && result.repaired
          ? {
              repaired: true as const,
              droppedUnknown: result.droppedUnknown,
              filledMissing: result.filledMissing,
            }
          : {}),
        finishReason: ['stop', 'length', 'content_filter'].includes(choice?.finish_reason)
          ? choice.finish_reason
          : 'unknown',
      },
    };
  } catch {
    return {
      ids: null,
      attempt: {
        provider: 'nvidia',
        model,
        status: timedOut ? 'timeout' : 'network_or_response_error',
      },
    };
  } finally {
    clearTimeout(timer!);
  }
}
