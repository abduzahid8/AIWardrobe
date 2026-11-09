# AIWardrobe Stylist 5 — Node Compute Candidate (NOT LIVE)

This is a **separate, protected compute runtime** for the existing, authoritative
`backend/supabase/functions/generate-outfits/index.ts` source. It is NOT a new
stylist algorithm or replacement client API.

## Why

Supabase Edge has a 2s active CPU/request limit. Production v206 experienced a
546 CPUTime shutdown after a previous 200 response in the same execution ID.
The built-in planner can do 256–512 quality evaluations, graph construction,
rotation exploration and reselects; a local 41ms microbenchmark for one graph
does **not** measure the entire request. Moving the full existing handler to
Node removes the Supabase Edge CPU ceiling from the heavy stages without
changing style gates, auth, quota or NVIDIA ranking.

## Source identity

Run `npm install`, `npm run build`, `npm test` from `backend/stylist-node`.

The build uses esbuild to transform ONLY the Deno HTTP listener and two
`esm.sh` Supabase imports. The full original backend handler, including
`enforceRateLimit`, `consume_edge_quota`, catalog RPCs, budget, fashion rules,
planning and provider ranking, is bundled from the current repository source.

The build prints the original Edge source SHA-256 (not a deployment fingerprint).

## Security

- Vercel exposes `/api/generate-outfits` only as an HMAC-authenticated endpoint.
- The original client continues calling `/functions/v1/generate-outfits`
  on Supabase. The **optional**, not-yet-deployed Supabase proxy signs requests.
- HMAC binds an expiring timestamp, nonce, bearer token and complete body;
  `crypto.subtle.verify` prevents timing-based signature comparison.
- The ORIGINAL server verifies user JWTs, charges quotas, enforces rate limits,
  validates item IDs and authoritatively selects outfits. Never meter in proxy.
- Set a **new cryptographically random >=32-character**
  `STYLIST_NODE_SHARED_SECRET` in BOTH Vercel and Supabase project secrets.
  **Do not** paste it into source, chat, CI logs or a public env variable.
- Set `SUPABASE_URL`, `SUPABASE_ANON_KEY`,
  `SUPABASE_SERVICE_ROLE_KEY`, `NVIDIA_API_KEY` in Vercel's **encrypted**
  server-only environment. Preserve any original paid-work kill-switch env
  settings. Do not expose service credentials to the app or browser.
- Configure `STYLIST_NODE_URL` in Supabase as the exact HTTPS URL ending in
  `/api/generate-outfits`. SSO/deployment protection must permit legitimate
  server-to-server calls without exposing an unsigned public endpoint.
- Only after signed, authenticated staging calls confirm behavior may the
  proxy replace the existing named `generate-outfits` Edge Function.

## Deployment plan (DO NOT skip these gates)

1. Deploy the isolated Node project from `backend/stylist-node`; enable
   "Include source files outside the Root Directory" for the monorepo.
2. Keep Vercel SSO/protection behavior understood; verify request signature
   rejection on 401 and valid signed calls with an authorized paid test user.
3. Verify identical response schema, generation version, quotas, budgets,
   valid catalogue IDs, image URLs, partial results, rotation and NVIDIA
   provenance against the existing v206; verify no double charge.
4. Record cold/warm CPU, wall time, success rate and 5 sequential requests
   including Old Money + Work and budget rejection. Check real visual quality.
5. Preserve exact v206 source in protected rollback storage; obtain explicit
   production authorization for the new proxy deployment.
6. Deploy the small proxy as `generate-outfits` only (not as a new public
   mobile endpoint). Confirm old clients still work; monitor 5xx/546, quotas.
7. If Node is unreachable, proxy returns 503. It must **never** silently
   replay generation on old Edge (which could double charge).
8. Revert to v206 from the verified backup only for critical incidents.

**Production remains unchanged until all gates pass.** The new route does not
create a publicly accessible keyless generation API.

## Deployment protection and secret separation

Vercel preview deployments can require Vercel Authentication even for
server-to-server traffic. Configure a Vercel **Protection Bypass for
Automation** token for the compute project and store it only as a Supabase
secret named `STYLIST_VERCEL_BYPASS_SECRET`. The edge proxy forwards it through
`x-vercel-protection-bypass` only to an HTTPS `*.vercel.app` endpoint, and
rejects redirects. This bypass does **not** authenticate generation: the
separate `STYLIST_NODE_SHARED_SECRET` HMAC and the user's original JWT are both
still required.

Do not place a bypass token, HMAC shared secret, Supabase service role key,
provider API key, or paid test user's JWT in GitHub, client-side Expo config
or any public environment. Do not connect the proxy to production until
an authenticated preview response has been verified without double billing.

## Verified isolated runtime (November 9, 2026)

- GitHub Actions validated Node typecheck, HMAC and authorization gates, and
  bundling the existing Stylist source without a parallel fashion algorithm.
- In a protected Vercel preview, `GET /api/health` returned HTTP 200.
- A temporary diagnostic route confirmed `verifyStylistTransport` and the
  original generated Stylist request handler both load successfully on Vercel.
- A GET of `/api/generate-outfits` returned the expected HTTP 405, rather
  than the previous `FUNCTION_INVOCATION_FAILED` due to missing ESM files.
- Temporary diagnostic API files were removed from this release candidate.
- An authenticated paid generation, subscription quota consumption, runtime
  CPU benchmark, and five sequential scenario requests are **not yet proven**.
- No environment secrets were inserted into source. Server-only preview
  secrets are configured; run bounded positive authenticated smoke tests with
  an authorized test user before proposing cutover.
- **Production remains on existing Supabase v207** until explicitly approved.

Node timing response header: `server-timing` includes process CPU and wall
duration for loading and running the original backend. Do not treat this as
isolated per-request CPU if the Node process handles concurrent requests.

## Verified protected staging transport — 2026-11-09

**Branch:** codex/stylist-node-compute-20261109 (draft PR #14)
**Production:** Supabase generate-outfits v207 ACTIVE, same original
stylist-5-editorial-v2-rc-20261109-r2 source fingerprint as v206;
**no Node traffic cutover**.

**Deployment status**
- Vercel Preview for Node compute reports /api/health status 200 with
  computeConfigured:true. Five server-only preview variables are configured
  without placing values in Git, PR comments, or mobile app code.
- Separate Supabase Edge Function aiw-stylist-node-staging v1 ACTIVE; the
  existing generate-outfits service was NOT replaced.
- Staging points at a pinned protected Vercel preview. The HMAC-only health
  request is non-billable and does NOT invoke the Stylist generator.
- Three consecutive signed staging transport probes returned HTTP 200 with
  hmacVerified:true; three invalid-user-JWT negative admission probes returned
  HTTP 200 from their diagnostic harness, while the original Stylist handler
  returned HTTP 401 unauthorized as expected. OriginalHandlerReached:true.
- Those six probes were issued without using real user credentials, reading
  products, calling paid NVIDIA inference or consuming generation quotas.
- The Vercel stage-connectivity and stage-auth-probe diagnostics used for
  verification were removed from the current branch once checks succeeded.
  The HMAC-gated transport-health route remains for future operational checks.
- CI on an earlier commit passed. The last diagnostic commit produced a
  GitHub Actions failed workflow with no jobs reported; re-run via GitHub App
  was disallowed. A fresh commit here triggers a new full verification.
- Original protected Node API continues to reject GET with HTTP 405.
  Full paid positive end-to-end inference is NOT yet established.

**Still required before production cutover**
1. Bounded authorized paid test-user POST against staging (not via public GET
   diagnostic), verifying response schema, no double-charge, exact products,
   budget guarantees and NVIDIA provenance.
2. Measure cold/warm CPU and wall, five sequential successful user requests,
   Old Money+Work and two other style+occasion scenarios.
3. Independently check quota/rate-limit effects and restore tests. Confirm
   production v207 backup before modifying its public route.
4. Explicit owner authorization for any production cutover. Roll back on
   5xx/546 or quota anomalies.

Do NOT merge this draft or deploy over production just because the negative
staging tests passed. Do not put JWTs, provider keys, Supabase service-role
keys, HMAC values or Vercel automation bypass values into GitHub or chat.
