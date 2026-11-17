# Local migration evidence

Command: `npm run test:migrations:local` (also in release:check). PostgreSQL is initialized in a fresh temporary directory, listens on a Unix socket only, uses synthetic roles/users/catalog data and is stopped/removed in finally. No .env, hosted URL or linked Supabase project is read. `AIWARDROBE_PG_BIN` may select installed PostgreSQL binaries; it cannot choose a database connection.

The six files listed in activation-config.json execute in order. Existing bounded catalog fixtures model baseline shop columns plus two synthetic auth users. Checks cover RPC filters, RLS, full eligible coverage, limits/pagination/rotation, private fit evidence isolation, service-only offer writes, price history coalescing, identity immutability, and data-preserving purchase disable. Added private tables have explicit authenticated grants and no anonymous access; referral/savings client writes are denied. The release isolation fixture checks owner SELECT, user-B SELECT/INSERT denial, owner-ID reassignment denial, RLS/grant inventory and representative indexes. Reverse phase 6→3 disable scripts revoke access and preserve evidence. Discovery inverse removes its function and checks that it is absent.

Scope limitations: this is NOT a fresh replay of all historical migrations, a hosted schema copy, a complete Supabase runtime, or exhaustive per-table CRUD/FK isolation proof. Existing historical migration numbering includes aliases; do not infer hosted pending status from filenames. `CREATE TABLE IF NOT EXISTS` cannot reconcile incompatible existing columns. Composite owner FKs now prevent cross-user visual asset/evidence/reference/try-on links; four representative cross-owner insert denials, cross-owner reference update/supersession denial, an owner reference, and owner-preserving reference cleanup on asset deletion are tested. Private-table RLS/grant inventory includes phase2 fit/watch and phase3 purchase tables plus referral participants. Remaining combinations need target-like probes before activation. PostgreSQL 15+ is required for column-specific SET NULL. Phase3–6 table rollback is access disable, not destructive schema reversal. Query plans/time in the log describe this synthetic local Postgres only.

The release report's migration_tests entry is authoritative for the current run; inspect its exit code/log. Hosted grants, Data API visibility, PostgREST schema cache, extension/version differences and storage policies require separate approved verification. These limitations block remote readiness even after local SQL passes.

| Installed compatibility combination | Local evidence | Status / missing proof |
|---|---|---|
| A old client + old backend | Existing contract tests and baseline synthetic schema | MANUAL REQUIRED: pinned previous binary/backend unavailable |
| B new client + old backend | catalogDiscovery tests force PGRST202/42883 and bounded fallback; IAP mocks | PARTIAL: missing-RPC behavior PASS, full installed Tier A/billing not run |
| C old client + new backend | Additive pending SQL leaves baseline shop reads; no old-column drops | PARTIAL: SQL structure only; old binary suite pending |
| D new client + new backend | RPC present-path mocks and disposable SQL behavior | PARTIAL: client/server pieces tested, full deployed installed journey pending |

CompatibilityExpectation() is a policy helper, not evidence that these four combinations ran end to end. Do not count its return value as a successful compatibility drill.
