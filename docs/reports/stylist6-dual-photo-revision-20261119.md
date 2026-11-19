# Stylist 6 — two independent creative studios for PHOTO-informed correction

2026-11-19 · Staging-only Stylist 6 (production Stylist 5 unchanged).
**Engineering candidate, NOT evidence of a beautiful 3×4 collection yet.**

## Actual user-visible problem before change

Exact Render LIVE SHA `b94fef19d3f3a6b46ca43131cff1e68da32712a7`
on a native iOS 26.5 catalog canary:

- Classic/Work: 7 independently original-photo-reviewed, 1 approved,
  1 displayed, 40.1 seconds server; fast 30B photo revision proposed
  two complete SKU-authentic outfits, but **0/2 PHOTO-approved**.
  PHOTO history had 2/4 held-out correct comparisons and rightfully
  did not influence candidate ranking.
- Classic/Date: 8 original-photo-reviewed, 2 approved, the final photo
  curator accepted **only 1 distinctive look**; fast 30B source
  correction proposed three real four-product looks, **0/3 approved**.
  The Date context had no previously judged historical examples yet.
- Every shown catalog look must still contain four **genuine** merchant
  source products: top, real layer, bottom and shoes. The photo editor
  must not fabricate a third look to reach three cards.

The key failing component is creative PHOTO-FEEDBACK RECOVERY, not a
numeric display target or missing hard garment rules. The quick
`nvidia/nemotron-3.5-lightning-30b-a3b` returns structurally legal
full-outfit tokens but sometimes fails independent photographic taste.
Because the code returned immediately on valid 30B output, the truly
independent `moonshotai/kimi-k3.1-pro` never saw those PHOTO critiques.

## Actual change

On the **isolated 72s Render catalog staging path** with the existing
owner-enabled fast-photo-revision flag, `composeCatalogSkuLooks()`
launches both independent creative models CONCURRENTLY. They receive
the exact same bounded, sanitized original-photo critic observations,
the currently authentic source item token menu, context/occasion and
applicable user constraints. No raw personal IDs, merchant SKU names,
private image URLs, secrets or unlicensed photo bytes are forwarded.

- Each model independently composes up to three whole **four-piece**
  outfits, not a rule-driven repair of shirt/jacket categories.
- The source-token validator checks both responses. Malformed tokens,
  wrong roles and repeats are rejected *before* photo review.
- Source-distinct proposals from both models are interleaved for the
  SAME maximum three-candidate second photographic audition. The
  separate source/hard budget/stock/weather and **independent original
  PHOTO critic** retain unrestricted veto.
- If either designer times out, refuses or emits invalid SKUs, legal
  looks from the other studio may still be used; neither can self-approve.
- No photo acceptance threshold, look diversity floor, customer-facing
  claims, brand restriction or product-category whitelist is changed.
- Total per-request original-photo review capacity and output cardinality
  are unchanged. The downside is a SECOND paid creative inference call
  and potentially longer critical-path waiting for Kimi, within the
  original ~8s revision and global 72s deadline. We MUST measure that
  trade-off on native iOS; if it displaces the photograph critic, revert.

The safe cloud diagnostic records `photoRevisionStudioCalls` and
`photoEditorialCreativeStudioCalls` (bounded integers) and enum
`parallel_kimi_nemotron_photo_revision` only when both creative
studios actually contribute genuine source-valid outfits. It never
logs their prompts, user preferences, vendor IDs or images.

## Tests / acceptance conditions

New integration tests prove both requests actually START concurrently,
their exact NVIDIA model IDs and distinct API schema contracts, actual
anonymized photo evidence in prompt, three disjoint four-SKU results,
continued legal source verification, one-studio fallback and complete
abstention when both responses fabricate merchandise. Existing
`fashion_deck` and cloud-handler suites must remain green.

This is NOT sufficient for production promotion. Deploy only the exact
CI-green SHA to isolated Render staging and run real iOS Classic/Work
and Classic/Date from the same authenticated catalog context. Compare
number independently PHOTO-approved, distinct visual directions,
shown four-item looks, second studio calls, latency and human visual
quality against the pinned b94fef19 prechange result. A weak third
look must NEVER be added to improve a count. The final release gate
remains ≥30 identical-catalog held-out contexts with ≥3 independent
blinded human judges and ≥70% preference for S6 vs Stylist 5, zero hard
merchant/physical/photo violations.
