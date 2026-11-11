# Stylist 6 cloud completion — verified evidence, 2026-11-11

Staging is NO-GO until credential pairing, protected server-to-server access,
authorized positive generation and actual iOS rendering are verified.

## Applied additive database change

Applied only `stylist6_idempotency` through the authenticated Supabase migration
connector. Actual database migration version: **20261111154741**. The local file
was renamed from the unapplied proposed timestamp to match actual cloud history.
No other pending migration was pushed. Numerous historical local migration
versions differ from the live history; a blanket database push is unsafe.

Created only `public.stylist6_request_claims`,
`claim_stylist6_request(uuid,text,text)` and
`finish_stylist6_request(uuid,text,integer,text)`.
The claim signature includes a required canonical SHA-256 request fingerprint;
the prior two-argument proposed RPC was never deployed. Old compute fails closed
until the updated Preview is admitted.

Verified live: RLS enabled, zero policies, anon/authenticated table privileges
absent, service_role table access present; both RPCs deny anon/authenticated
execution and permit service_role. Two overlapping database sessions produced
one first claim and one duplicate. Changed-fingerprint conflicts and terminal
outcome replay passed live SQL assertions. Probe rows were removed; no quota RPC
was invoked by these database checks.

Claims and quota debit remain separate transactions. A crash can strand an
unbilled claim or leave a debit without a terminal outcome. Claims are never
reclaimed automatically, preventing a second debit at the expense of automatic
recovery. This is at-most-once admission, not atomic exactly-once billing or
response delivery. A timeout must not trigger a fresh logical request.

## Edge deployment

`aiw-stylist6-node-staging` version **1**, ACTIVE, JWT gateway verification true.
ID: `d3314cdf-550d-4142-9a31-2f84f26eb321`.
Bundle SHA-256: `66d662f0673e7a14aab6fc575f37e863215b651babebfb47ccb3e48225d3f664`.
Entrypoint imports the canonical proxy; the deployment contains its HMAC module.
Generate the exact secret-free connector descriptor with:

```
node scripts/stylist6/edge_package.mjs /tmp/aiw-s6-edge.json
```

Real staging gateway results: no Authorization → 401
`UNAUTHORIZED_NO_AUTH_HEADER`; invalid JWT → 401
`UNAUTHORIZED_INVALID_JWT_FORMAT`. These prove gateway rejection, not a positive
function-to-compute request. Compute separately validates the actual user through
Supabase Auth. No anonymous or invalid-JWT test was sent for generation.

## Validation

- 127 Node tests passed: Stylist 6, cloud handler, transport and existing Node route
  regressions. 25 cloud-handler cases include malformed budgets rejected pre-debit,
  canonical fingerprints and changed-content conflicts without a second debit.
- Four client Jest suites passed: 37 tests, budgets, currencies, cloud version gates.
- TypeScript passed. Targeted ESLint: zero errors, 18 existing warnings.
- Worktree secret scan: zero findings across 3254 text files; ignored files,
  binaries and history are outside that scan.
- Web export succeeded to `/tmp/aiw-s6-web` after granting dependency-cache access.
- Existing bundling solution retained; compute build passed.
- Compute deadline 45 seconds, proxy 55 seconds, Vercel limit 60 seconds.
  Deadline checks are cooperative, not a hard cancellation of all underlying IO.

## Access and iOS limitations

Owner-authenticated `vercel curl` on the original immutable Preview returned
405 with `stylist-6-cloud-staging-1`. This is owner access evidence only.
Existing Preview environment names and branch scope were confirmed; readable
Supabase URL configuration matches the intended project. Secret values cannot
be pulled from Vercel Preview.

Supabase CLI identity was available, but secret-management operations stalled.
The attempted S6 secret update timed out; pairing is not asserted. Temporary
secret input was deleted. S5 secret was not changed. No Vercel protection setting
was disabled. No successful protected Supabase-to-Vercel call is asserted.

The user approved **one** metered generation request, but secure dedicated
account access was not provided and no metered generation was executed.
Actual quota delta for a positive test is therefore unmeasured, not assumed.
Zero NVIDIA inference calls were made; provider readiness is unverified.

Actual booted Simulator inspection showed:
`[runtime not ready]: Error: Cannot find native module 'ExpoAsset'`.
Two native builds failed due to disk exhaustion. Temporary build data created by
this run was removed. ProfileScreen:3273 could not be reached or reproduced;
that reported crash is unresolved. No cloud outfit screenshot exists from this
run. No native success or app release readiness is asserted.

## Rollback and production preservation

Leave the isolated staging function fail-closed or delete only that function;
remove only S6 branch-scoped Preview configuration/deployments and revert this
commit if needed. Preserve production `generate-outfits` v207 and the S5 proxy
v4, the S5 HMAC key, production routes, entitlements and subscriptions.

The safest database rollback is to retain the inert RLS-protected claims objects,
so request history cannot be lost and inadvertently re-admitted. A schema-removal
rollback must first stop S6 traffic, archive/reconcile all claims and quota
outcomes, then remove only the three S6 objects through a reviewed migration.
Never reset or restore the shared database to undo this staging change.
SQL probes were transactionally rolled back (or cleaned after concurrency tests).
A destructive schema rollback was not executed against the live shared database.


## Updated Preview admission

Compute commit `312fdfa005325a231ff19e78f21378aba4bbfdac` is READY at
`https://aiwardrobe-stylist-compute-acrvx5ao2-abduzahid8s-projects.vercel.app`.
Deployment ID: `dpl_Dgc2rn9QS94zGo3UmsVd8xRQ8xqD`, Preview, region hnd1.
Owner-authenticated GET returned 405 with the expected S6 engine identity.
The proxy now pins that tested immutable deployment to match the applied
three-argument fingerprint RPC. A redundant explicitly requested deployment
was canceled after the Git-triggered deployment was confirmed READY at the same
SHA. That canceled deployment is not passing evidence.

The deployed proxy preflight returned 200 `ok`; a public project token passed
the gateway and reached proxy code, which returned 503
`staging_transport_unconfigured`. No runtime import failure was observed.
This confirms that Supabase HMAC configuration remains missing, rather than
asserting successful credential pairing.
