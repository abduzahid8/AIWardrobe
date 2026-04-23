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

  const authHeader = req.headers.get('authorization') ?? '';
  const jwt = authHeader.replace(/^Bearer\s+/i, '').trim();
  if (!jwt) {
    return new Response(JSON.stringify({ error: 'unauthorized' }), {
      status: 401,
      headers: { 'content-type': 'application/json' },
    });
  }

  const url = Deno.env.get('SUPABASE_URL');
  const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (!url || !serviceKey) {
    // Fail open so a misconfigured env doesn't DoS production, but log it.
    console.warn('[rateLimit] missing SUPABASE env — allowing');
    return null;
  }

  // Resolve user id from the JWT via the anon-compatible endpoint.
  const anon = createClient(url, serviceKey);
  const { data: userRes } = await anon.auth.getUser(jwt);
  const userId = userRes?.user?.id;
  if (!userId) {
    return new Response(JSON.stringify({ error: 'unauthorized' }), {
      status: 401,
      headers: { 'content-type': 'application/json' },
    });
  }

  // Fixed-window counter keyed by minute bucket encoded into `feature`.
  const now = new Date();
  const bucketMs = Math.floor(now.getTime() / (windowSec * 1000)) * windowSec * 1000;
  const bucketKey = `rl:${opts.feature}:${bucketMs}`;

  const { data, error } = await anon.rpc('increment_rate_bucket', {
    p_user: userId,
    p_feature: bucketKey,
  });

  if (error) {
    console.warn('[rateLimit] db error — failing open', error.message);
    return null;
  }

  const used: number = typeof data === 'number' ? data : 1;

  if (used > opts.limit) {
    return new Response(
      JSON.stringify({ error: 'rate_limited', retry_after_sec: windowSec }),
      {
        status: 429,
        headers: {
          'content-type': 'application/json',
          'retry-after': String(windowSec),
        },
      },
    );
  }

  return null;
}
