# Stylist 6 — Real native iOS aesthetic audit and actual-photo final fashion jury

Date: 2026-11-17. Owner's mission: exceptional personal styling, not three
technically valid cards. Evidence was obtained from the **booted native
AIWardrobe-S6-QA-20261115 iOS simulator**, using actual on-screen button taps,
authenticated production-shaped Supabase STAGING gateway v61, and Render
compute exactly pinned to `f350c24afcd49fedf229a531689d55a60d2851a6`.

## Baseline before the visual-jury change

| Real iOS scenario | Render request UTC | AI composed / source admitted | NVIDIA photo attempted / approved | User displayed | Render elapsed | Final editor |
| --- | --- | --- | --- | --- | --- | --- |
| Classic / Work | 11:49:54 | 5 / 5 | 8 / 5 | 3 photo-judged | 49,574 ms | text-only applied |
| Classic / Date night | 11:56:26 | 6 / 6 | 8 / 6 | 3 photo-judged | 58,001 ms | text-only applied |

Screenshots, Maestro results and the active simulator are local QA artifacts.
Do **not** commit raw screenshots showing the QA account identifiers or third-party
merchant photographs without rights clearance.

Actual **aesthetic failure** observed by reviewing every visible card:
- **Work:** three distinct SKUs yet substantially the same shirt/trousers/shoes
  uniform: brown trousers with tan loafer, white dress shirt/navy trousers/navy
  loafers, light chambray/black wide trousers/black Chelsea boots. One product
  tile appeared blank at the initial screen capture. Passing NVIDIA individual
  photo quality does NOT establish collection-level exceptional taste.
- **Date:** the first outfit repeated a plain oxford shirt/dark trousers/dress
  loafer office formula; another was black knit polo/black chinos/light casual
  shoe; one layered navy knit + grey jacket + light trousers was more
  distinctive. The editor chose three despite a conventional first look.

The prior text-only final editor received photographic observations, not the
actual garment pictures. It could choose among verified portfolios but could
not visually compare the full **collection** as a fashion editor.

## Architectural correction

Add one independent **real-photograph editorial board** after all per-look
source/merchant/budget/stock/physical/permission gates and whole-look NVIDIA
photo approval:

1. Reuse the SAME request-scoped source-photo media cache. Re-authorize
   every image and fail back to the prior editor on missing/unauthorized bytes.
2. Compose ONE bounded JPEG board of up to eight photo-approved, fully
   sourced looks. Each row is an opaque `Lxx` full outfit; the four columns
   are photographed TOP/LAYER/BOTTOM/SHOES, with an honest blank for no layer.
   Neither the board labels nor prompt reveal user ID, SKU ID, product name,
   source URL or other private identifiers.
3. Give an independent, bounded **vision-language** final fashion editor the
   actual garment photographs and its existing bounded photographic
   observations, user styling context and **pre-verified** 1/2/3-look `Fxx`
   portfolio menu. It chooses ONE already viable portfolio, not new SKUs.
4. Re-run the original exact NVIDIA-approved look combination admission
   `selectVisualDeck` after the decision. The VLM never approves an
   unreviewed product or relaxes physical/source constraints.
5. Fail gracefully to the previous text-only editor for vision-provider
   outage, invalid selection, photo/authorization failure or a deadline.
   Prevent extra user-visible cards on failed visual decisions; no fabricated
   style improvement claims.

Model `nvidia/nemotron-nano-12b-v2-vl` is explicitly a fast image-capable
model according to NVIDIA's 2026 official NIM documentation. Live endpoint
compatibility and comparative fashion quality still MUST be verified.
`sharp` is already pinned in the deployed backend/stylist-node package;
there is no new package dependency or persistent user-media store.

## Local verification

- Source-modification test matrix: **479/479 Stylist 6 node tests PASS**.
- Separate source-gated photo-board tests cover original JPEG composition,
  source authorization, no invented outerwear/photo, bounded deadlines.
- Visual-jury tests cover actual image inclusion and opaque tokens, valid
  photo-verified portfolio choice, 400 fallback, missing-image fallback,
  impossible F-token refusal.
- `npm run build --prefix backend/stylist-node`: PASS; source S5 auth/quota
  bundle hash unchanged (`f90ced33...`), Stylist 6 portable Render route built.

## No false victory

The two real scenarios above are a **baseline on f350**, not an after-change
success claim. Before promoting this visual-jury change, run authenticated
native simulator QA on the exact NEW deployed SHA for the same two scenarios.
Report whether visual jury actually applied versus text fallback, final
photographic diversity, 1/2/3 selection, latency/timeout and any product image
failures. A healthy deploy, more candidates, a VLM returning 200 and passing
unit tests are NOT fashion excellence. Keep production Stylist 5 untouched
until the >=70% independent human blind comparison succeeds.

## Staging v1 (f1f000e) — mandatory adverse LIVE native iOS evidence

Real Render deployment `f1f000e` went LIVE; the booted native iOS
simulator generated through the Supabase v61 gateway. Results worsened,
which invalidates any automatic rollout decision:

| Same scenario | UTC logged | Source-admitted AI compositions | Hosted photo approved/attempted | Shown | Visual jury | Other failure | Server elapsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classic / Work | 12:13:32 | 6 / 6 | **1 / 8** | **1** | not run (only one photo-approved) | photo revision: invalid_composition | 47,868 ms |
| Classic / Date night | 12:16:58 | 6 / 6 | **3 / 8** | **2** | image_unavailable; existing text editor used | Creative Director timed out (0 plans) | 58,008 ms |

These are counterexamples to "all green / 3 cards / new vision" and are NOT
proof of an improved fashion stylist. They show two key recovery deficits.

## Next staging revision — salvage genuine photo-feedback improvements

- Exact root for Work failure: the independent critic correctly rejected 7/8.
  A second AI apparel recomposition was attempted, but its validator rejected
  the ENTIRE three-proposal response if even ONE SKU was duplicated, leaving
  no second photographic wave. The new code distinguishes six-look initial
  audition (still needs >=3 unique full looks) from independently photo-informed
  rescue. In rescue only, 1–2 fully source-bound, physical-SKU-unique,
  structurally sound outfits survive a third malformed AI combination.
  Revalidate each one against the SAME budget/merchant/weather/source hard
  gates then independently rejudge real product photographs; no synthetic
  quality approval, SKU reuse, fee loophole or lower photo thresholds.
- Date visual jury failure had only the broad `image_unavailable` status.
  Added bounded, privacy-safe exact collage statuses: ready,
  invalid_input, unauthorized_or_missing, deadline, source_missing,
  photo_processing and oversized. Increases only the **staging** pre-existing
  image board budget from 1.35s to 2.6s without increasing paid visual
  evaluation budgets or altering production. This distinguishes missing
  merchant image, processing error and tight deadline in the next REAL log.
- Verified locally against the same source tree: **481/481 Stylist6 tests
  PASS**, backend/stylist-node build PASS, original production Stylist5
  auth+quota bundle hash unchanged.
- Still require new exact deployed SHA and actual native iOS outcomes:
  `photoEditorialVisualJury:applied` and `photoEditorialVisualMedia:ready`
  must be seen together before asserting the visual fashion jury WORKS.
  If the vision model fails (or is not permitted), only the photo-approved
  text director can pick final looks. No claim of improvement until a
  blind human assessment of the actual outfit photographs.
