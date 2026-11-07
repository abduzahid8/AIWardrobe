# AIWardrobe — Visual QA v192 and Model Routing (2026-11-07)

## State and provenance
- Owner allows active changes and fast iteration as of 2026-11-07, while preserving regression checks, accurate deployed-source identification, and rollback for high-risk operations.
- Source commit: 52a97e3f3595, branch codex/outfit-selection-quality-20261107, not main (main remains 5784fc5cff24 as checked).
- Live Supabase generate-outfits ACTIVE v192; deployment metadata updated at 2026-11-07 12:31:15 UTC = 17:31:15 Tashkent. Screenshots time 17:33–17:37, chronologically AFTER deployment; request-level attribution to v192 not confirmed without logs.
- Screenshot client identifier 088432be532b suggests a separate frontend build lineage from the source of the deployed server function. Prevent accidental regressions when syncing branches.
- User reports 75 passing targeted checks and v191 rollback. The report was not independently rerun in this review.

## Observed visual quality and UX
- Casual / Travel and Semi-Classic / Date night: visible outfits all contain top + layer + bottom + shoes. Multiple physical-looking shoe styles appear; this does not by itself prove fair account-wide rotation.
- Casual / Travel: black/olive/grey and denim outfits generally coherent. Weather is explicitly unavailable, so no confident climate layering verdict.
- Semi-Classic / Date night: some coherent combinations; important misses include: checked shirt + busy jacquard/paisley cardigan + formal black Derby shoes; light washed jeans + denim shirt + tailored navy blazer + brown Oxford shoes; possible linen-and-heavy-tailoring conflict when weather is unknown.
- Visible input text '(1 500 000 UZS)' is example placeholder, NOT proven accepted budget. The 12.7M UZS jacket highlights a UX/value problem, but not an independently verified budget-enforcement bug. Confirm request payload and budget status before claims.
- UI language mixed (English and Uzbek), likely Uzbek typo QAYERA vs QAYERGA, and visible Client/build/backend identifier debugging line in screenshots. Confirm dev-only context before modifying public UI.
- Ranking up to 9 pre-approved NVIDIA candidates cannot compensate for a weak or badly calibrated pre-approval quality floor. A score/test success is not a visual fashion-quality guarantee.

## Current engineering priority — GPT-6.1 Sol High
1. Compare commit 52a97e3, current main and live deployed generate-outfits v192. Never unknowingly deploy older main over active v192.
2. Reproduce suspect looks as fixture-based whole-outfit compatibility tests. Include positive controls preserving good Casual/Travel, denim smart-casual when appropriate and valid mixed looks.
3. Diagnose qualityFloor / occasion rules / material-weight conflicts / footwear-formality. Prefer calibrated pairwise penalties or narrow quality constraints over blanket bans.
4. Check whether weather-unavailable condition appropriately degrades confidence; verify structured maxShopBudget and prompt budget separately (placeholder is not a contract).
5. Preserve slot completeness, style, budget, physical identity, rotation, quotas, latency and authentic seller/source records. Test before and after; preserve rollback.

## Model choice for future AIWardrobe tasks
| Task | Model |
|---|---|
| Whole-outfit quality regressions, complex styling logic, multi-file bug | GPT-6.1 Sol High |
| Supabase schema/Edge, auth, quotas, migrations, security | GPT-6.1 Sol High |
| Branch synchronization / deploy safety / rollback | GPT-6.1 Sol High |
| Product architecture, catalog retrieval, dedup and variant design | GPT-6.1 Sol High |
| Uzbek translations, microcopy, isolated simple UI fixes | GPT-6 Luna High |
| Simple test scaffolding, README and limited documentation | GPT-6 Luna High |
| Screenshot QA triage | GPT-6 Luna High; escalate root cause to Sol High |

At the start of future AIWardrobe tasks, recommend explicitly: 'Модель: [name] — почему'. The assistant cannot change the user's model picker itself.
Current next task: use GPT-6.1 Sol High to resolve semi-classic/date-night stylistic conflicts in a tightly scoped tested patch; do not first expand nine candidates arbitrarily.

## Important constraints
No unreviewed destructive database change. Changes are allowed, but production safety still requires measured benefit, tests and rollback. These notes do not themselves implement or deploy anything.
Sources: user-provided simulator screenshots, GitHub commit 52a97e3, GitHub branch listing, Supabase production function metadata, earlier AIWardrobe direction-audit document.