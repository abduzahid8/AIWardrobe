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
  finishReason?: string;
}
export function validateLookRanking(raw: string, allowedIds: readonly string[]) {
  let parsed: unknown;
  try {
    parsed = JSON.parse(
      raw
        .trim()
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/, '')
    );
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
          ' supplied IDs in best-first order. Each supplied ID must occur exactly once. Copy IDs literally. No product IDs. Required ID set: ' +
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
    const result = validateLookRanking(
      typeof choice?.message?.content === 'string' ? choice.message.content : '',
      lookIds
    );
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
