# Server paid-work stop controls — local preparation only

Existing entitlement admission was already present: `enforceRateLimit()` authenticates the caller and calls the service-only `consume_edge_quota` RPC, whose authoritative tier lookup denies free/lite VTON and enforces applicable limits. No duplicate quota charge was added.

The shared server boundary now also supports operator-only environment controls:

| Variable | Meaning | Unset behavior |
|---|---|---|
| AIWARDROBE_PAID_WORK_DISABLED | Nonempty values other than `false` (case/whitespace normalized) stop all metered paid Edge work. Use `true` to stop; malformed nonempty values fail closed. | Existing behavior retained |
| AIWARDROBE_DISABLED_PAID_FEATURES | Comma-separated exact endpoint names to stop selectively, e.g. `try-on,ai-chat` | No selective stop |

The explicit current coverage is `ai-chat`, `ai-process`, `analyze-outfit`, `create-trip-plan`, `generate-outfits`, and `try-on`. Each uses and returns the shared admission result with server-owned paid-metering options. This is not a claim about arbitrary future endpoints, legacy Node API routes, background jobs, or client-only flag helpers. Inventory and reconcile any such routes before beta. `mannequin-tryon` already returns 410 and does not render.

Authentication remains required while paused. For a valid caller, stop denial happens before rate-counter/quota RPC writes and before the caller reads provider keys, composes images or invokes providers. Denial is HTTP 503 with `{ "error": "feature_disabled" }` and the endpoint's CORS headers. Requests cannot override these controls through body, headers or a client premium claim. Authenticated `sync-subscription` and `budget-rates` are rate-only paths (`meterPaidWork:false`) and remain available with their existing auth/rate checks; they do not consume paid quota.

The client outfit error decoder maps this code to the existing localized `outfitMaker.aiUnavailable` path, which permits validated closest-outfit alternatives when candidates exist; otherwise it shows unavailability. Core stored wardrobe/calendar data is not killed. Old clients may show their generic unavailable error; installed old/new compatibility is still manual.

`backend/supabase/functions/.env.example` is a server-only template with the safest preparation posture: global stop true and VTON selectively disabled. It is NOT loaded automatically and no real runtime variable was changed here. Do not put these under EXPO_PUBLIC_. Code defaults preserve existing deployments until an operator explicitly applies settings.

Local proof: actual shared helper tests cover stop, invalid global values, selective matching, client override rejection, auth denial, paid-quota write avoidance, subscription recovery and resume with exactly one admission. A configured VTON handler executes the real helper under a stop and performs zero config/composition/provider calls. The six-entry AST contract guards constant server feature/metering options and the early return. The installed SDK-style HTTP error-body decoder and deterministic alternative tests verify the client-side classification and existing alternative rules. These do not replace a deployed runtime drill.

Future approved drill (NOT EXECUTED):

1. Pin reviewed source/tree hashes and an approved non-production target. Capture current control values, provider caps, owner and recovery route. Hold new cohort traffic. Existing code without this guard is not protected by merely setting the variable.
2. With explicit approval, prepare stop=true before deploying the reviewed six guarded entries. Verify their deployed artifacts first. Confirm valid requests return feature_disabled, invalid users remain denied, no new quota/provider work occurs, and billing recovery still works.
3. Keep VTON and other unapproved advanced endpoints on the selective stop list. Only after separately approved budgets/provider and fallback checks, clear the global stop for the approved core scope; verify selective stop still holds.
4. On incident, stop new admissions, account for already admitted/in-flight work, and use provider-side hard caps/credential denial if immediate cancellation is required. This guard does NOT cancel requests already executing. Observe the bounded request drain before claiming calls ceased.
5. Restore only the captured approved control state after investigation. Verify the next allowed request consumes one normal admission; do not re-enable jobs/notifications/experiments implicitly.

Hosted control propagation and deployed coverage remain unverified (RC-002/RC-007). Supabase documents environment access and production secret management in [Environment variables](https://supabase.com/docs/guides/functions/secrets). No secret-setting, deployment or paid-provider command ran in this task.
