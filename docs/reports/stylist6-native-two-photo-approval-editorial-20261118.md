# Stylist 6 — missing third fashion direction when PHOTO critic approves only two

2026-11-18 (Asia/Tashkent), isolated native QA, not human quality acceptance.

## Paid real native Work scenario (pinned 84ba6a8)

- Actual iPhone simulator showed **2 real product-backed looks** in `Your Outfits`, with 01/02 navigation, merchant prices and store links. Unlike previous lost-response regression, this deck reached the actual mobile UI.
- Render at 2026-11-17 19:09:40 UTC:
  `elapsedMs=31995`, 6 AI-composed/6 source-admitted looks, 9 actual-photo reviews, **2 approved**, displayed=2, no unreviewed padding, prior photo-feedback revision accepted 1/3 extra photos.
- At 19:09:41 UTC `stylist6_result_delivery` confirmed **recoverable:true** for the exact finished paid result. Strict user-only RLS/TTL storage was applied earlier.
- Original whole-collection Fashion Editor returned `not_run` despite 2 valid approved outfits and ~40s of server creative budget remaining. Code gate `approvedLooks.length>=3` prevented it from seeing the photographed TWO and communicating the missing third direction to the second SKU designer.
- The actual nine VLM verdicts include explicit rejects: white tee + puffer/black chinos for office; charcoal dress coat + track joggers + loafers; double-breasted blazer + wash jeans + formal patent loafers. This is a **composition quality problem**, not a test-only fault or device styling issue.
- The two displayed successful sets are both tailored. A stylist must create **a different third genuinely shoppable direction**, not repeat another office blazer or artificially weaken the independent model quality floor.

## Exact architectural correction

On 72-second isolated Stylist6 staging, the original independent photo
portfolio editor now runs with **at least two** instead of three
independently approved source looks (when the initial paid batch had
>=3 valid actual-photo observations). The VLM receives a constrained
menu of mathematically feasible 1/2-look collections plus original
photo evidence. It can select the real pair or the strong single, and
returns a bounded editorial observation identifying what novel
silhouette/texture/footwear contrast is MISSING from the already
photographed collection. The existing **second TRUE AI** composition
uses only UNUSED source SKU tokens and must pass exact source,
merchant, budget, weather, photo rights and independent new
NVIDIA full-look quality checks. The final editor must re-select a
server-verified 1/2/3 look collection including at least one newly
photo-approved garment set before replacing the original deck.

This is neither a forced third look nor a heuristic SKU ban. A
provider error or photo veto preserves the original actual-photo
accepted deck. No production S5 mutation and no deadline expansion.

## Tests

New integrated `two real PHOTO approvals still activate portfolio
fashion judgment and its NEW AI creative direction`:
six source-verified initial looks; independent critic accepts ONLY
two; ordinary first photo retry fails; formerly unreachable
PHOTO editor selects its feasible two-look portfolio and supplies
missing direction; second AI designer composes NEW real source SKUs;
independent SECOND PHOTO critic approves novel looks; final PHOTO
editor verifies and selects a genuinely three-look collection.
Verified no fake or unreviewed product and all four garment roles.

Production remains NO-GO until LIVE native requests repeatedly
show genuinely different, visually convincing, buyable works;
human blind preference >=70% against S5 with >=30 held-out cases,
three human reviewers.
