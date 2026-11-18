/**
 * Isolated Stylist 6 cloud staging route (Vercel Node.js).
 *
 * This is NOT the Stylist 5 compute route: it runs the Stylist 6 fashion
 * engine server-side and reports a truthful `stylist-6-*` generation
 * version. Access requires a short-lived HMAC signed by the isolated
 * Stylist 6 staging proxy, in addition to the original user's JWT.
 *
 * Nothing is metered here before the Stylist 6 handler runs; JWT
 * verification, rate limiting and consume_edge_quota happen inside the
 * handler exactly once per admitted request.
 */
import {
  verifyStylistTransport,
  STYLIST_TRANSPORT_MAX_BODY_BYTES,
} from '../transport/signature.mjs';
import * as supabaseJs from '@supabase/supabase-js';
// Version duplicated (not imported) so the route file has zero Stylist 6
// module dependencies at runtime: the handler arrives only via the built
// bundle. cloud_handler.test.mjs asserts the two stay identical.
const STYLIST6_CLOUD_VERSION = 'stylist-6-cloud-staging-1';

type CloudHandler = {
  handleStylist6CloudGenerate: (input: {
    method: string;
    headers: Record<string, string>;
    bodyText: string;
    env: Record<string, string>;
    io: unknown;
    timeoutMs?: number;
  }) => Promise<{ status: number; body: unknown }>;
  createCloudIo: (input: {
    url: string; anonKey: string; serviceKey: string; supabaseJs: unknown;
  }) => unknown;
};

let handlerPromise: Promise<CloudHandler> | undefined;

function cloudHandler(): Promise<CloudHandler> {
  if (!handlerPromise) {
    handlerPromise = (async () => {
      // Generated during npm run build from the isolated Stylist 6 entry.
      // A single artifact: no per-file module resolution at runtime.
      const handler = await import('../dist/stylist6-bundle.mjs') as CloudHandler;
      if (typeof handler.handleStylist6CloudGenerate !== 'function') {
        throw new Error('stylist6_handler_unavailable');
      }
      return handler;
    })().catch((error) => {
      handlerPromise = undefined;
      throw error;
    });
  }
  return handlerPromise;
}

function errorResponse(code: string, status: number): Response {
  return Response.json(
    { success: false, outfits: [], code, error: code, generationVersion: STYLIST6_CLOUD_VERSION },
    { status },
  );
}

function envOf(name: string): string {
  return process.env[name] || '';
}

/** Server-only AI fashion configuration, never controlled by client data.
 * Both gates must reach cloud_handler; otherwise the whole-look Fashion
 * Director remains NOT_ENABLED even when the hosting environment has '1'.
 * Keep this narrow allowlist: never pass all process.env to an LLM.
 */
export function stylist6CreativeServerConfig() {
  return {
    STYLIST6_CREATIVE_DIRECTOR_ENABLED: envOf('STYLIST6_CREATIVE_DIRECTOR_ENABLED'),
    STYLIST6_AI_SLATE_ENABLED: envOf('STYLIST6_AI_SLATE_ENABLED'),
    // Server-controlled only: makes real catalog cohort stable for a
    // controlled SAME-SKU artistic A/B, not a customer-controlled fetch flag.
    STYLIST6_STABLE_COHORT_ENABLED: envOf('STYLIST6_STABLE_COHORT_ENABLED'),
  };
}

/** Render-only S6 staging deadline. The signed Supabase Render proxy has an
 * 85-second compute deadline; reserve 13 seconds for proxy/cold-start/return
 * JSON. Vercel staging and ALL Stylist 5 routes keep the existing 50s bound.
 * This flag is passed ONLY by the server's in-process Render entrypoint,
 * never read from user headers, body, or environment supplied by a client.
 */
export const STYLIST6_RENDER_CANARY_BUDGET_MS = 72_000;
export function stylist6ComputeBudgetMs(runtime?: { renderCanary?: boolean }): number | undefined {
  return runtime?.renderCanary === true ? STYLIST6_RENDER_CANARY_BUDGET_MS : undefined;
}

export async function POST(request: Request, runtime?: { renderCanary?: boolean }): Promise<Response> {
  // Dedicated Stylist 6 staging secret, independent of the Stylist 5
  // shared secret. Never fall back to the S5 value.
  const secret = envOf('STYLIST6_NODE_SHARED_SECRET');
  if (!secret || secret.length < 32) return errorResponse('stylist_compute_unconfigured', 503);

  const authorization = request.headers.get('authorization') || '';
  if (Number(request.headers.get('content-length') || 0) > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return errorResponse('invalid_generation_request', 413);
  }
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return errorResponse('invalid_generation_request', 413);
  }

  const previousSecret = envOf('STYLIST6_NODE_SHARED_SECRET_PREVIOUS');
  const authorized = await verifyStylistTransport({
    secret,
    previousSecret,
    body,
    authorization,
    timestamp: request.headers.get('x-aiw-timestamp'),
    nonce: request.headers.get('x-aiw-nonce'),
    signature: request.headers.get('x-aiw-signature'),
  });
  if (!authorized) return errorResponse('unauthorized', 401);

  const env = {
    STYLIST6_NODE_SHARED_SECRET: secret,
    STYLIST6_NODE_SHARED_SECRET_PREVIOUS: previousSecret,
    SUPABASE_URL: envOf('SUPABASE_URL'),
    SUPABASE_ANON_KEY: envOf('SUPABASE_ANON_KEY'),
    SUPABASE_SERVICE_ROLE_KEY: envOf('SUPABASE_SERVICE_ROLE_KEY'),
    NVIDIA_API_KEY: envOf('NVIDIA_API_KEY'),
    // Forward the server-owned switch: without this, Creative Director is
    // silently disabled despite being enabled in the compute environment.
    ...stylist6CreativeServerConfig(),
    STYLIST6_REFERENCE_BUCKET: envOf('STYLIST6_REFERENCE_BUCKET'),
    // Owner kill-switch: any truthy value ('1'/'true') forces an honest
    // NVIDIA_NOT_RUN for the hosted visual review, whatever the client asks.
    STYLIST6_VISUAL_DISABLED: envOf('STYLIST6_VISUAL_DISABLED'),
    STYLIST6_DEFAULT_VISUAL_EVAL: envOf('STYLIST6_DEFAULT_VISUAL_EVAL'),
  };
  if (!env.SUPABASE_URL || !env.SUPABASE_ANON_KEY || !env.SUPABASE_SERVICE_ROLE_KEY) {
    // Owner-only runtime diagnostics: presence flags, never credentials.
    console.error('stylist6_configuration_missing', {
      supabaseUrl: !env.SUPABASE_URL,
      supabaseAnon: !env.SUPABASE_ANON_KEY,
      supabaseService: !env.SUPABASE_SERVICE_ROLE_KEY,
    });
    return errorResponse('stylist_compute_unconfigured', 503);
  }

  const headers: Record<string, string> = { authorization };
  for (const name of ['x-aiw-timestamp', 'x-aiw-nonce', 'x-aiw-signature']) {
    const value = request.headers.get(name);
    if (value) headers[name] = value;
  }
  // Include cold-start module loading AND the Stylist 6 handler in timing.
  const cpuStart = process.cpuUsage();
  const wallStart = performance.now();
  try {
    const handler = await cloudHandler();
    const io = handler.createCloudIo({
      url: env.SUPABASE_URL,
      anonKey: env.SUPABASE_ANON_KEY,
      serviceKey: env.SUPABASE_SERVICE_ROLE_KEY,
      supabaseJs,
    });
    const { status, body: responseBody } = await handler.handleStylist6CloudGenerate({
      method: 'POST', headers, bodyText: body, env, io,
      ...(stylist6ComputeBudgetMs(runtime) === undefined
        ? {} : { timeoutMs: stylist6ComputeBudgetMs(runtime) }),
    });
    const cpu = process.cpuUsage(cpuStart);
    const cpuMs = (cpu.user + cpu.system) / 1000;
    const wallMs = performance.now() - wallStart;
    return Response.json(responseBody, {
      status,
      headers: {
        'server-timing': `aiw_node_cpu;dur=${cpuMs.toFixed(1)}, aiw_node_wall;dur=${wallMs.toFixed(1)}`,
      },
    });
  } catch {
    // Never leak backend internals, JWTs, product data or service credentials.
    return errorResponse('stylist_compute_unavailable', 503);
  }
}

export function GET(): Response {
  return errorResponse('method_not_allowed', 405);
}
