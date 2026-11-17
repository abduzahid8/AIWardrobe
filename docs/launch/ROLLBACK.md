# Recovery plan

Rollback is a stop-exposure operation that preserves user rows. It is not a promise that an unknown hosted before-state has been reconstructed. Local reverse disable scripts are tested; hosted restore, deployment rollback and native downgrade remain untested blockers.

| Forward change | Local inverse / verification |
|---|---|
| Catalog discovery | backend/supabase/rollback/shop_catalog_discovery.sql; RPC absent; bounded read fallback works |
| Purchase intelligence | backend/supabase/rollback/purchase_intelligence_disable.sql; personal fit rows retained; new writes/public offers denied; price trigger disabled |
| Style | backend/supabase/rollback/phase3_disable.sql; client grants revoked; original synthetic evidence retained |
| Visual | backend/supabase/rollback/phase4_disable.sql; revoke client access; retain images/references/history metadata |
| Platform | backend/supabase/rollback/phase5_disable.sql; revoke client access; retain audit evidence |
| Growth | backend/supabase/rollback/phase6_disable.sql; revoke client access; retain purchases/reward/audit history |

For an approved cutover abort: (1) stop cohort expansion and paid routes with a verified kill; (2) halt new client rollout; (3) restore exact prior Edge/config artifacts; (4) disable new SQL surfaces in reverse dependency order: growth, platform, visual, style, purchase, discovery; (5) run old-client smoke and owner/user-B probes; (6) verify retained row counts and no increased anonymous permissions. If any prior deployment/schema/backup artifact is missing, stop and escalate to the named incident owner instead of guessing.

These additive tables may remain after rollback. Do not drop them to satisfy a schema diff. Future re-enable requires reviewed grant restoration and trigger recreation from the pinned forward definitions; do not blindly replay migrations with non-idempotent CREATE POLICY. Restoring production data from backup is a separate destructive operation requiring explicit authorization and a tested recovery point. Local isolated DBs are disposable and shut down after each run.

Before approval capture: schema/grants/policies dump checksum, backup restore receipt, previous function trees/config, previous offering/webhook mapping, provider limits, previous signed clients, cache version compatibility evidence, named owner, max acceptable recovery window. None of these remote receipts is fabricated here.
