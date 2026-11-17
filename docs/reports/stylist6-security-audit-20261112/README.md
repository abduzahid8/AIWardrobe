# Stylist 6 cloud path — security and reliability audit (2026-11-12)

Scope: the isolated staging path
`app → Supabase Edge Function aiw-stylist6-node-staging (v9) → HMAC-signed
Vercel Node compute (/api/stylist6-generate-outfits) → Stylist 6 handler`.
Reviewed at commit 9928e0e with the deployed configuration read back live.

## Findings (all verified, none outstanding)

### Identity and isolation
- **JWT verification before any state touch.** The handler verifies the user
  JWT via Supabase Auth (`getUser`) before inventory, claims, or quota; the
  Supabase gateway also enforces `verify_jwt` (live probe: unauthenticated
  POST → 401).
- **Cross-user wardrobe isolation.** Owned rows are read from
  `clothing_items` filtered by the verified JWT user id, with the
  service-role client; there is no client-supplied row path. Client
  `catalogRows` are ignored and reported (`clientInventoryIgnored`).
- **Claim isolation.** `stylist6_request_claims` has RLS enabled, no anon/
  authenticated policies, grants only to `service_role`; the claim/finish
  RPCs are `security definer` with empty `search_path` and are executable
  by `service_role` only. Keys are `(user_id, request_id)`; a fingerprint
  mismatch under one logical id returns `request_content_conflict` (409)
  without a second debit (tested in the handler suite).

### Transport
- **HMAC.** Dedicated `STYLIST6_NODE_SHARED_SECRET` (never the S5 value, no
  fallback), 5-minute timestamp window + nonce, constant-time comparison,
  request body bound into the signature. Size is checked twice (declared
  and actual, 413). Live probe through the whole chain returned
  application-JSON 400 `invalid_stylist6_request` with the exact S6 version
  — proving gateway → proxy → protection-bypass → HMAC → handler without
  metering.
- **Redirect discipline.** The proxy fetch uses `redirect:'error'`, the
  origin is a pinned `*.vercel.app` host, and the bypass header is only
  attached when the secret matches a strict charset. Vercel Deployment
  Protection verified enabled (direct GET → 302 Vercel Authentication).
- **Credentials.** Server-only secrets (`SUPABASE_SERVICE_ROLE_KEY`,
  NVIDIA key, HMAC secret) exist only as Vercel Preview-branch envs bound
  to the S6 branch and as Supabase function secrets; the client app holds
  only the public anon key. No secret values were read or printed in this
  audit; names were read via `vercel env ls`.

### Admission and quota
- **Order:** transport → JWT → cheap rate bucket (`increment_rate_bucket`,
  no debit) → server inventory → anchor validation → durable claim →
  single quota debit (`consume_edge_quota`) → planning. Invalid anchors and
  duplicate/conflicting requests fail **before** the debit.
- **Idempotency.** Durable claim via `claim_stylist6_request` (atomic insert
  wins; concurrent duplicates serialize on the primary key). Redelivery
  replays the recorded outcome as 409 without re-planning or re-debiting.
  Missing claim storage fails closed before debit. Claims and debits remain
  **separate transactions**: this is at-most-once admission, NOT atomic
  exactly-once billing or response delivery. A crash can strand a claim
  without a debit or preserve a debit without a recorded outcome; ambiguous
  requests must never be retried with a new id. This round added terminal
  recording for the post-claim "no feasible looks" outcome so replays carry
  the real prior code.
- **Fingerprint.** Canonical (key-order-insensitive) SHA-256 over the
  request; changed content under one logical id conflicts instead of
  double-billing.

### Failure handling
- **Timeouts.** Layered: client 120 s deadline (abort propagates through
  supabase-js), proxy 55 s abort, compute handler 45 s internal deadline
  (surfaces 504 `stylist6_ranking_failed`), Vercel maxDuration 60 s, NVIDIA
  provider 8 s / critic 18 s, product fetch 8 s.
- **Bounded work.** ≤4 slots × 100 discovery rows, ≤500-row wardrobe pages,
  visual profiles in 100-id chunks, ≤512 KB visual images, ≤4 MB product
  downloads, 64 KB provider response cap, response size caps, per-minute
  rate 10.
- **Errors.** Every failure returns a stable application-JSON code with the
  truthful engine version; internals, JWTs, credentials and product data
  are never leaked (the route's catch returns a bare 503; presence-flag
  diagnostics log only booleans). The app now recovers the real server code
  from `FunctionsHttpError` so users see the server's verdict.
- **Recency history.** Fixed this round: rotation-history reads no longer
  await an unbounded persistence write chain (a hung AsyncStorage write
  could previously stall a read); reads now settle writes with a 1.5 s
  bound, keeping "recency is best-effort, never break generation".

### Media and NVIDIA
- **Inference permission separation.** Network reachability is not
  authorization: product images are fetched only from the committed
  `HOST_ALLOWLIST`; every outfit transmitted to the provider must pass
  `isInferenceAuthorized` per item, otherwise the look completes as
  `NVIDIA_NOT_RUN` with full deterministic ranking. Reference bytes come
  only from the owner-staged private bucket (`STYLIST6_REFERENCE_BUCKET`,
  unset today → reference-guided judging honestly `NOT_RUN`).
- **Model contract.** `nvidia/nemotron-3.5-super-vl-120b-a12b` unchanged;
  structured verdicts validated (scores in range, 3–6 evidence lines,
  candidate-id binding); refusal/truncation/content-filter decode to
  honest abstentions; provider body streamed with a hard byte cap; ranking
  is never relabeled as provider judgment.

## Residual risks (accepted, documented)
1. Claims and quota are separate transactions (above) — at-most-once
   semantics, no change without a migration that would risk double debits.
2. `catalog_discovery_candidates`' JSON projection omits two fields
   (`is_active`, `fashion_subtype`); the S6 server boundary reconstructs
   both from projected data. Production S5 uses different RPCs
   (`catalog_stylist_candidates*`) and is unaffected.
3. The pinned compute deployment is immutable; the proxy pin must be
   updated on every redeploy (done this round) or staging fails closed
   (503), never open.
4. `STYLIST6_REFERENCE_BUCKET` unset → reference-guided judging unavailable
   in cloud until the owner stages licensed reference bytes.
5. The Vercel bypass secret is project-wide (not deployment-scoped) —
   accepted by the owner previously; S5 HMAC remains independent.
