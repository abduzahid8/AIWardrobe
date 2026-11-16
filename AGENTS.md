# AIWardrobe — instructions for AI coding agents

**Mandatory first read** for any changes to the stylist, shop catalog, outfit-maker iOS screen, or visual AI: [docs/STYLIST6_NORTH_STAR.md](docs/STYLIST6_NORTH_STAR.md).

## Single mission
Ship an **excellent, scalable personalized AI fashion stylist** that generates up to 3 truly beautiful, coherent, varied, shoppable outfit directions based on the person's style preferences, actual wardrobe and context — not an application that merely passes backend tests or prints three outfits.

## Mandatory decision order
- Catalog and wardrobe provenance, weather/occasion, pinned anchors, verified currency/budget, auth/quota, merchant availability claims: **hard checks**.
- Understand actual fashion from product PHOTOS; confidence/unknowns explicitly tracked.
- A context-aware **Creative Director** designs 3 differentiated style blueprints before selecting available SKUs.
- Compose a substantial candidate pool; validate whole-outfit visual quality and personal taste; then choose best 1–3 with authentic photos and links.
- A model's `approved` flag is only one piece of evidence. An independently assessed human fashion-quality improvement is required before declaring success.

## Avoid
- Endless one-off product-name regex exclusions as the primary method of improvement.
- Equating `NVIDIA_APPLIED`, HTTP 200, green CI or 3/3 with good taste.
- Calling synthetic unit tests, AI-rater votes, or simulated decks owner acceptance.
- Fabricated stock, prices, price conversions, item photos, merchant permissions or model outcomes.
- Staging/production conflation; newer GitHub SHA does NOT automatically update pinned Supabase staging.
- Promoting Stylist 6 to production without proof of quality, compatibility and safe rollback.

## Evidence and workflow
- Before implementation: state the cross-scenario failure / quality gap and expected measurable correction.
- After implementation: compare SAME user context and same catalog on before/after, inspect ACTUAL product photo contact sheets, run focused + backend/type tests, then stage on exact SHA.
- Primary release acceptance: >=70% S6 wins in >=30 independent human blind S5 comparisons, 3 reviewers, no hard violations, honest shopping and availability, verified iOS generation.
- Human panel and merchant-image rights are release gates, not tasks to fabricate evidence for.
- Preserve all user-uncommitted work, secret isolation, HMAC/JWT/RLS, at-most-once quota and S5 rollback.
- End each session with the actual code/deploy SHAs, evidence, outstanding blockers, and next highest-leverage change.

## Where to work
`backend/stylist6/` (creative/editorial/ranking/critique), `backend/stylist-node/` (cloud compute), `supabase/functions/aiw-stylist6-node-staging/` (isolated staging proxy), `screens/AIOutfitmaker.tsx` and `src/services/stylist6Dev.ts` (client), `docs/reports/` (QA and acceptance).

The north-star specification is authoritative for stylist quality priorities; do not silently remove or weaken its requirements.
