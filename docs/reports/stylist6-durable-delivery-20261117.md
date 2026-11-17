# Stylist 6 — paid result delivered to Supabase but lost on iPhone

**P0 production blocker** — November 17, 2026, [GitHub issue #18](https://github.com/abduzahid8/AIWardrobe/issues/18).

## Actual independent evidence (not inferred)

Pinned staging Render `723703d`, native signed-in iPhone simulator,
New clothing / Classic / Work, Supabase JWT Edge v61.

- Render at 17:59:02 UTC: 50,056ms, six AI generated and source-admitted,
  nine independent full-garment photo attempts, four approved,
  final PHOTO fashion VLM selected two real four-SKU looks.
- Supabase `function_edge_logs` for the same call: real
  `aiw-stylist6-node-staging` v61, **HTTP 200**, response length
  **7,847 bytes**, **52,619 ms** execution. Thus server success is proven.
- The iPhone app nevertheless showed **ZERO outfits**, red
  `outfitMaker.generateFailed`; Metro logged a transport failure.
  First earlier cold Render run likewise had a mobile Edge request failure.
- New photo-informed *second* final editorial creative loop was
  `not_run` in this particular source request (insufficient safe time).
  510 automated tests alone did NOT prove user-visible success.

## New delivery architecture

The first paid POST generation is never retried automatically.
The already-existing durable `stylist6_request_claims` and transactional
debit prevent second paid requests with the same logical ID.

For a **successfully settled** cloud generation only, the server stores
the **EXACT original successful validated JSON** in new server-only-writable
`public.stylist6_completed_results`, keyed by
`(user_id,generationRequestId)`, bound by FK to the existing immutable
idempotency claim. A 64KiB payload cap applies, TTL 45 minutes and an
independent `pg_cron` expired-data deletion every 15 minutes.

RLS on; `anon` has no SELECT or INSERT, `authenticated` has SELECT
only and row-level access **only when** `auth.uid() = user_id` and row
has not expired. `service_role` alone INSERTs (plus SELECT/DELETE).
Synthetic owner-vs-other-user transactional SQL privacy tests passed;
all fixtures rolled back without persisting a user result. No privileged
key or request secret enters mobile code, and production Stylist5
`generate-outfits` is untouched.

Native client can, after a LOST body/timeout/SDK fetch failure, make a
bounded 6.5s **read-only** owner-scoped Supabase Data API query for the
SAME request ID. It never POSTs NVIDIA, settles quota, or selects other
users' rows. The recovered deck must match request ID, exact S6
generation version, original verified-partial/full signed-off receipt,
all independent PHOTO verdicts and source/merchant/image/price/4-slot
contracts before being shown.

On an absent/expired/foreign/unverified result, client keeps the original
actionable failure; no padding, no local fake looks. A genuine HTTP 422/
NVIDIA photo rejection is never reinterpreted as recoverable success.
Timeouts now have localized `requestTimeout` instead of generic
`generateFailed` copy.

## Acceptance still open

1. New pinned LIVE server emits `stylist6_result_delivery:
   {outcome:'completed_paid_deck', recoverable:true}`.
2. Native iPhone, same paid request ID, deliberately lost initial
   HTTP200 body, after 120s/SDK failure: displays exactly the same
   server photo-approved 1/2/3 completed looks, without any extra
   generation/quota consumption.
3. Public/other-user role cannot read recovery rows, rows deleted when
   expired; verify iOS sign-out and cancellation.
4. Work & Date night real photos improve and human >=70% blind preference
   in >=30 matched scenarios, 3 judges, before Stylist6 production go.

Until these are verified, **production NO-GO**; server HTTP200 alone is
not an end-to-end product.
