> The current executable gate is `npm run release:check`; it writes the pinned RC identity and evidence into `docs/launch/evidence/`. See [the current packet](launch/README.md). Readiness is BLOCKED until required manual and remote gates are proven.

# RELEASE.md (Phase 7CZ, 7AO–7AS, 7AT–7AV, 7CI, 7CM–7CN)

## Environments

`development` (local, debug, mocks) → `preview` (staging, info) →
`production` (warn, no mocks). Production must never silently point at dev
resources — verified by `productionPointsAtDev()` in release tooling.

## Versioning

Semver app version + integer build (Android versionCode / iOS build).
Migration compatibility window documented per release. Minimum supported
client: `3.0.0` (see `clientSupported()`); hard update only for critical
incompatibility. Old client/new backend and new client/old backend degrade
with fallback, never crash (additive migrations, flag-guarded RPC callers).

## Cutover (ordered, NOT executed)

1. Apply additive DB migrations → 2. validate schema/RLS → 3. deploy Edge
Functions → 4. enable read paths → 5. client with flags off → 6. internal
cohort → 7. observe → 8. gradual activation (canary: internal, beta, 1%,
5%, 25%, 50%, 100%, each with entry criteria, abort threshold, rollback).

## Launch-safe flag defaults

Experimental/high-cost: `off` or `limited`. Stable core (`wardrobe_core`):
`on`. Every flag documents owner, dependency, kill switch.

## Release candidate

Run `node scripts/release-candidate.mjs`. Output per check: PASS / FAIL /
MANUAL (manual QA and remote activation are always MANUAL until performed).
Release manifest (app version, build, commit, migrations, Edge versions,
flag defaults, schema/rule versions, known limitations) is produced alongside.
