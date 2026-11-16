# Stylist 6 — Creative Director canary, 16 Nov 2026

**Owner intent:** excellent, personalized, scalable outfits; not more product-specific exclusion rules. This work implements a foundational creative-planning seam and does NOT certify public-release fashion quality. See `docs/STYLIST6_NORTH_STAR.md` and root `AGENTS.md`.

## Implemented in PR #15

- `backend/stylist6/creative_director_ai.mjs`: one **bounded 7.5s text-only NVIDIA planning call** generates three intended outfit architectures BEFORE real-SKU retrieval; the API receives aggregate catalog subtype counts, weather, occasion, explicit color/fit preferences, and a short brief. It receives NO user IDs, wardrobe SKU IDs/names, photographs, merchant URLs, stock, prices, or auth credentials.
- Every model-proposed role must exist in the server-admitted inventory of the same slot; exactly three distinct role architectures are required. Invalid model outputs, missing key, unavailable provider and timeouts safely fall back to existing house concepts. The model cannot bypass weather, budget, anchors, provenance, style/occasion rules, provider image-authorization checks or the final independent NVIDIA fashion review.
- Deterministic inventory/context hash cache: 10 minutes, max 64 entries (in-process only); no separate database, no new provider key, no unbounded per-user state.
- `backend/stylist6/fashion_deck.mjs`: accepted AI direction briefs lead the bounded creative candidate list; current house/reference briefs remain candidate backups. `diagnostics.aiCreativeDirectorBriefs` identifies whether planning actually influenced the candidate pool.
- `backend/stylist6/cloud_handler.mjs`: Creative Director is **explicitly gated by server-only `STYLIST6_CREATIVE_DIRECTOR_ENABLED=1`**, skipped for workout and insufficient remaining deadline, and reported in a `creativeDirector` response receipt with status, call count, accepted brief count and version. It is not a new consumer API and cannot be turned on by a client payload.
- Unit and integration tests cover anonymous inputs, strict SKU/type validation, three unique concept architectures, cold-weather layer admission, caching, deterministic fallback, real-SKU retrieval, and hard-budget preservation.
- Existing isolated GitHub workflow includes these tests.

## Verification truth table (update after evidence)

| Gate | Status |
| --- | --- |
| Documentation / anti-drift instructions committed | DONE — `AGENTS.md` + `docs/STYLIST6_NORTH_STAR.md` |
| Creative Director code and offline tests committed | DONE — branch only |
| Latest exact code CI | Pending at report creation; do not claim pass without run conclusion |
| Vercel **new creative build** | BLOCKED by Vercel 402 `api-deployments-free-per-day` (over 100 deployments/day) |
| Vercel branch-only preview opt-in | SET: `STYLIST6_CREATIVE_DIRECTOR_ENABLED=1`; does not affect the old READY build |
| Supabase staging repin | NOT DONE: can only repin after new compute is READY and verified |
| Authenticated live iOS sample and photo sheet on **new creative build** | NOT DONE |
| Independent human blind >=70% in >=30 S5/S6 comparisons | NOT DONE |
| Verified merchant stock/size/delivery/image rights | NOT DONE |
| Production Stylist 5 | NOT CHANGED by this work |

## Reproduce and stage safely

1. Pull the latest PR #15 branch. Run `node --check backend/stylist6/creative_director_ai.mjs`, `node --test backend/stylist6/creative_director_ai.test.mjs`, `node --import tsx --test backend/stylist6/*.test.mjs`, and `npm run typecheck` with locked dependencies.
2. Review isolated GitHub workflow + general CI for the **exact** final commit. Never label canceled/pending as success.
3. Create a preview deployment for the exact SHA in `aiwardrobe-stylist-compute`; confirm READY, exact SHA, and that preview-branch `STYLIST6_CREATIVE_DIRECTOR_ENABLED=1` was injected. Do not promote to production.
4. Change only Supabase `aiw-stylist6-node-staging` pin to this READY deployment, preserving existing code/JWT/HMAC, only after verifying old pin/version. Keep `generate-outfits` pointing to Stylist 5.
5. Run an authenticated iOS and cloud test for at least Classic Work, Old Money Weekend, Semi-Classic Date, Winter Work, and a constrained personal-taste profile. Capture exact final displayed SKU IDs and PHOTOS. Inspect `creativeDirector.status`, `briefsAccepted=3`, `visualReceipt.displayedJudged`, and independent fashion judgments.
6. Measure p95 time, added AI planning costs, and % accepted plans. If negligible quality gain, disable branch flag without production disruption and improve the design rather than adding per-product regex filters.
7. Independent human blind test remains mandatory before declaring the stylist excellent or publicly scalable.

### Operational blocker
Vercel returned HTTP 402 `payment_required` with `api-deployments-free-per-day`; the API requires plan/billing change or quota reset before a new preview. **Do not** mislabel an older READY preview as the creative-planning release. No paid plan change was made.
