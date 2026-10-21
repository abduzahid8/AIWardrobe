// Shared per-user rate limiter for Edge Functions.
//
// Uses the project's own Postgres as the backing store (no Redis dep).
// Callers pass the request, the feature key, and the per-minute cap.
// Returns a Response on deny, or null on allow.
//
// Prereqs: the calling function must have access to SUPABASE_URL and
// SUPABASE_SERVICE_ROLE_KEY env vars (standard for deployed functions).

// deno-lint-ignore-file no-explicit-any
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';

export interface RateLimitOptions {
  /** Unique key per logical endpoint, e.g. 'ai-process'. */
  feature: string;
  /** Max calls per window per user. */
  limit: number;
  /** Window in seconds. Defaults to 60. */
  windowSec?: number;
  /** CORS headers from the calling endpoint, including on denied requests. */
  headers?: Record<string, string>;
  /** Authentication-only endpoints (e.g. receipt sync) do not require a paid tier. */
  meterPaidWork?: boolean;
}

/**
 * Returns a 429 Response if the caller has exceeded the cap; otherwise null.
 * Always return early on null === allowed.
 */
export async function enforceRateLimit(
  req: Request,
  opts: RateLimitOptions,
): Promise<Response | null> {
  const windowSec = opts.windowSec ?? 60;
  const reply = (error: string, status: number, retryAfter?: number) =>
    new Response(JSON.stringify({ error, ...(retryAfter ? { retry_after_sec: retryAfter } : {}) }), {
      status,
      headers: {
        ...opts.headers,
        'content-type': 'application/json',
        ...(retryAfter ? { 'retry-after': String(retryAfter) } : {}),
      },
    });
  if (!Number.isSafeInteger(opts.limit) || opts.limit < 1 ||
      !Number.isSafeInteger(windowSec) || windowSec < 1 || !opts.feature) {
    return reply('rate_limit_unavailable', 503);
  }

  const authHeader = req.headers.get('authorization') ?? '';
  const match = /^Bearer\s+(\S+)$/i.exec(authHeader.trim());
  if (!match) return reply('unauthorized', 401);

  const url = Deno.env.get('SUPABASE_URL') || Deno.env.get('SB_URL');
  const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('SB_SERVICE_KEY');
  if (!url || !serviceKey) return reply('rate_limit_unavailable', 503);

  try {
    const admin = createClient(url, serviceKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data: userRes, error: authError } = await admin.auth.getUser(match[1]);
    if (authError) {
      // Auth outages are temporary failures; invalid credentials remain 401.
      return reply('unauthorized', Number(authError.status) >= 500 ? 503 : 401);
    }
    const userId = userRes?.user?.id;
    if (!userId) return reply('unauthorized', 401);

    const nowMs = Date.now();
    const bucketMs = Math.floor(nowMs / (windowSec * 1000)) * windowSec * 1000;
    const { data, error } = await admin.rpc('increment_rate_bucket', {
      p_user: userId,
      p_feature: `rl:${opts.feature}:${bucketMs}`,
    });
    // Paid work must never bypass admission control when storage is unavailable.
    if (error || !Number.isSafeInteger(data) || data < 1) {
      return reply('rate_limit_unavailable', 503);
    }
    if (data > opts.limit) {
      const retryAfter = Math.max(1, Math.ceil((bucketMs + windowSec * 1000 - nowMs) / 1000));
      return reply('rate_limited', 429, retryAfter);
    }
    if (opts.meterPaidWork === false) return null;
    const { data: admitted, error: quotaError } = await admin.rpc('consume_edge_quota', {
      p_user: userId,
      p_endpoint: opts.feature,
    });
    if (quotaError || typeof admitted !== 'boolean') return reply('rate_limit_unavailable', 503);
    if (!admitted) return reply('feature_limit_reached', 403);
    return null;
  } catch {
    return reply('rate_limit_unavailable', 503);
  }
}
