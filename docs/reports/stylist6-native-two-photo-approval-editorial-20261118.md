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

## Follow-up live QA revealed two MORE hidden gates

Pinned f87e93d REAL authenticated native Work request at 2026-11-17
19:21:34 UTC: two real PHOTO-approved looks displayed, 8 attempts,
32 seconds total server work, one verified feasible 2-look portfolio.
But `photoEditorialStatus='not_enough_approved'`,
`photoEditorialVisualJury='not_run'`,
`photoEditorialCreativeRevision='not_run'` even after the outer caller
was corrected to allow two. Reason: `selectPhotoEditorialDeck` itself
independently enforced `options.length>=3` and the REAL photo
contact-sheet builder enforced `looks.length>=3`.

Now BOTH actual source-photo downstream stages accept exactly **two**
previously approved real whole looks, build an authentic 2-row contact
sheet with their true 4-piece SKU photos, and ask the independent
120B photographic fashion jury to select its proven pair/single token
and describe the missing third direction. One look still fails closed:
one photograph is not a meaningful **collection** comparison.

New tests prove real two-row original licensed images (no invented
third), anonymized F-token visual VLM selection with sanitized
`missingFashionDirection`, and preserve refusal for one source look.
A live direct iOS request on this latest patch is still required
before declaring the end-to-end photo-feedback repair functional.

## Another actual VLM JSON omission — do not lose the artist's second decision

A further real authenticated native Classic/Work Render request on
`ecd4294` at 2026-11-17 19:27:15 UTC **successfully sent real source
JPG contact sheet to the 120B VISION editor**:
`photoEditorialStatus:applied`,
`photoEditorialVisualJury:applied`,
`photoEditorialVisualMedia:ready`, 2 exact photo-approved
portfolio choices, 33.7 seconds, no invented/weak third.
However `photoEditorialCreativeRevision:not_run` AGAIN. Once
the VLM has selected its exact server-feasible two-look F-token, its
OPTIONAL free-text `missingFashionDirection` sometimes comes back
empty/too short or omitted in the NVIDIA `json_object` response.

The proper creative solution is to preserve the model's actual choice
and transfer bounded **FACTUAL source-image observations of the
approved looks** to a separate AI composer as creative context.
This can run only after the real multimodal 120B jury returned
`visualJury:applied` with `visualMediaStatus:ready`, or when the
original editor supplied a valid explicit gap. The approved evidence
includes PHOTO-observed silhouettes/texture/color/footwear and
whole-outfit relationships, not raw product URLs, user IDs, merchant
records or untrusted commands. The follow-up AI independently decides
what *new fashion* direction to compose from unused merchant SKU
tokens. New items MUST pass all hard/source/media rights and actual
independent NVIDIA PHOTO quality gates again, and the second editor
must select from server-verified complete portfolios. No automatic
"good third", no relaxation.

Diagnostic `photoEditorialCreativeSource`: bounded enum
`visual_editor_gap` or `approved_photo_evidence`; no private
image data or raw model free-text in logs. Regression simulates a
120B real-photo editor selecting TWO but omitting the optional field,
and tests that the next AI composes a third idea from original
critic photographic observations, with an independent SECOND image
critic and final portfolio-editor decision. A separate sanitizer
test proves these real approved descriptions cannot leak URL,
identity, untrusted object keys or raw photos.

This repair remains **staging-only** until a real pinned iPhone
generation verifies that `photoEditorialCreativeRevision` actually
runs. Strict photo quality/no-padding floors remain unchanged.
