# AIWardrobe Stylist 5 — Node Compute Candidate (NOT LIVE)

This is a **separate, protected compute runtime** for the existing, authoritative
\`backend/supabase/functions/generate-outfits/index.ts\` source. It is NOT a new
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

Run \`npm install\`, \`npm run build\`, \`npm test\` from \`backend/stylist-node\`.

The build uses esbuild to transform ONLY the Deno HTTP listener and two
\`esm.sh\` Supabase imports. The full original backend handler, including
\`enforceRateLimit\`, \`consume_edge_quota\`, catalog RPCs, budget, fashion rules,
planning and provider ranking, is bundled from the current repository source.

The build prints the original Edge source SHA-256 (not a deployment fingerprint).

## Security

- Vercel exposes \`/api/generate-outfits\` only as an HMAC-authenticated endpoint.
- The original client continues calling \`/functions/v1/generate-outfits\`
  on Supabase. The **optional**, not-yet-deployed Supabase proxy signs requests.
- HMAC binds an expiring timestamp, nonce, bearer token and complete body;
  \`crypto.subtle.verify\` prevents timing-based signature comparison.
- The ORIGINAL server verifies user JWTs, charges quotas, enforces rate limits,
  validates item IDs and authoritatively selects outfits. Never meter in proxy.
- Set a **new cryptographically random >=32-character**
  \`STYLIST_NODE_SHARED_SECRET\` in BOTH Vercel and Supabase project secrets.
  **Do not** paste it into source, chat, CI logs or a public env variable.
- Set \`SUPABASE_URL\`, \`SUPABASE_ANON_KEY\`,
  \`SUPABASE_SERVICE_ROLE_KEY\`, \`NVIDIA_API_KEY\` in Vercel's **encrypted**
  server-only environment. Preserve any original paid-work kill-switch env
  settings. Do not expose service credentials to the app or browser.
- Configure \`STYLIST_NODE_URL\` in Supabase as the exact HTTPS URL ending in
  \`/api/generate-outfits\`. SSO/deployment protection must permit legitimate
  server-to-server calls without exposing an unsigned public endpoint.
- Only after signed, authenticated staging calls confirm behavior may the
  proxy replace the existing named \`generate-outfits\` Edge Function.

## Deployment plan (DO NOT skip these gates)

1. Deploy the isolated Node project from \`backend/stylist-node\`; enable
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
6. Deploy the small proxy as \`generate-outfits\` only (not as a new public
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
