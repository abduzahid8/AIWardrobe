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

## 2026-11-17 — actual AI-first architecture (not another styling rule)

The last deployed live `1599b98` Date night trace at 12:24:47 UTC was a
counterexample: the model timed out AFTER waiting for ~17.7 seconds of legacy
catalog fashion/ranker selection; only 2/8 independently photo-approved looks
remained, two cards displayed, zero final visual editorial call. The fashion
decision had a strict ~3s model timeout AFTER the expensive prebuilt candidate
generation. This is a critical architecture flaw, not a source-title rule.

The NEXT STAGING code runs the autonomous real-product outfit creator **before
the legacy look builder**:

1. Build an anonymous source-backed menu of up to 12 real, user-admissible,
   independent merchandise SKUs per garment role directly from the trusted
   inventory, WITHOUT first demanding a ranker-produced outfit. Use actual
   current photo fashion facts and the existing inventory-grounded Creative
   Director plans, with a diverse whole-outfit decision owned by the model.
2. Start the independent six-look AI call immediately on catalog readiness and
   let existing legacy retrieval continue as a CONCURRENT fallback, not a
   prerequisite. The staging-specific model call may run up to 8.5s instead of
   losing a valid decision after ~3s. This does NOT increase the trusted request
   deadline, eight-photo NVIDIA critic cap, owner quota units or production cost.
3. After the ranker returns, validate EVERY AI-chosen combination again through
   the SAME unchanged whole-outfit source, budget, size/slot/weather, physical
   constraints and real-NVIDIA photo evaluation. A loose/source-unverified set
   cannot ever be displayed; timeout/error uses the original safe selector.
4. Limit scope strictly to the existing 72s Render Stylist6 staging with
   `modelSlateEnabled`, catalog-only, no locked anchor. Wardrobe/mixed/S5
   production keep prior behavior. No private merchant URLs or user identifiers
   are sent to the composer.
5. Report `skuCompositionParallel` in sanitized Render quality traces. This
   flag MUST be true before claiming new architecture was exercised in an
   authenticated iPhone QA request.

Exact local evidence: **484/484** Stylist6 unit tests passed, including a
case where an otherwise valid model response arrives after 4.2s (was formerly
cut off at 3s), and an integration case proving AI sees direct inventory
BEFORE the legacy ranker's first hard-gate call. The portable Node/Render build
passed and the original Stylist5 auth/quota bundle hash remains `f90ced33...`.

No new LIVE aesthetic acceptance or post-deploy server result is claimed here.
Require actual iOS Classic/Work and Classic/Date night, source-admitted model
slate, full photo-review rejection breakdown, independent human blind quality
and a truthful `photoEditorialVisualMedia`/visual-jury event.


## 2026-11-17 — fixed missing fourth product in actual catalog looks

Explicit owner correction: four real items per shoppable look, not merely three.
Existing cloud input had `optionalLayer:true` and warm/optional look-policy
would omit outerwear; AI `outerwear:none` was valid, so a photo-approved deck
could still have only top, bottom and shoes.

Strict 72s Render catalog canary now passes non-optional layering to the
actual outfit planner and independently verifies the *returned* whole deck
contains exactly one each of top/outerwear/bottom/shoes, all unique
source-verified merchant SKUs. A missing fourth NEVER gets a made-up SKU
or an undeserved NVIDIA photo verdict. Smaller-budget routes and mixed or
wardrobe sources remain unchanged. A four-piece wardrobe test (photo reviewed)
and a signed, 72s full cloud handler test with actual source-backed blazers
pass. The real iOS after-deploy screenshot is still required before claiming
this reached the user. No guarantee of climate-suitable fourth product in
very hot weather until real accessory catalog infrastructure is introduced.


## Real iOS result after four-item owner correction: b958e16, Nov 17

Authenticated *native iOS simulator* (AIWardrobe-S6-QA-20261115), cloud
gateway v61, exact Render deployment `b958e1640a1b24b4aaea66d6047f1162a3dc7f28`
LIVE. Classic / Work QA completed at 13:12:52 UTC:

- Actual server `stylist6_safe_visual_quality`: 59,025ms; 8 attempted
  whole-look photo reviews, 2 independently approved, **2 displayed**.
  Primary direct-SKU model timed out despite parallel invocation (not a
  fashion-quality victory); fallback fashion selector produced the approved
  looks. No unreviewed padding.
- **Inspected both iPhone cards and scrolled to the bottom**. Card 01:
  TOP Soft Cotton Polo Shirt; LAYER Regular-Fit Wool Cardigan;
  BOTTOM relaxed chinos; SHOES suede shoes. Card 02:
  TOP Relaxed-Fit Knit Polo; LAYER Comfort Stretch Knit Cardigan;
  BOTTOM brown slim trousers; SHOES leather loafers.
  **Each has exactly FOUR real merchandise tiles and merchant links**, in
  the 2x2 grid. The entire first and second outfit screens were viewed.
- Some merchant photo tiles were blank in screenshots (not missing
  products): separate reliability defect in source image/display.
  Neither look is proven world-class by this count check.
- The new sanitized `catalogFourItemContract` field incorrectly reported
  `false` in initial b958e16 logs because it accessed
  `modelFashionSelection` rather than final deck diagnostics; corrected
  log plumbing in the follow-up commit. The actual displayed UI **did**
  contain four items; this is evidenced independently by native screenshots.
- Strict rule is intentionally limited to catalog on >=70s Render staging;
  wardrobe/mixed/production Stylist5 contracts unaffected.

## 2026-11-17 — AI-FIRST fashion decision, skip CPU-heavy formula retrieval on success

Live b958e16/0685040 outcomes proved that merely starting a network AI
composer in parallel to the OLD synchronous ranker was insufficient:
`skuCompositionParallel=true` but `status=timeout` after 16s, with
`catalog_fashion_and_selection` using ~23s CPU and fallback concepts.
On the same day, an isolated real NVIDIA API benchmark (with mock merchant
catalog; no user products or identifiers sent) finished 3 new whole looks in
1.6s and 6 new whole looks in 2.4s. This is evidence of a server scheduling
problem, not proof of aesthetic strength or typical provider latency.

**Architecture:** let the actual six-look AI composer complete its network
call BEFORE any synchronous legacy outfit-ranker work. Independently
re-validate every model-picked four-source-SKU outfit via original buyer
budget/FX, excluded-product, complete-role and weather hard gates,
`fashionScorer.rankLook`, and the source photo transmission authorization.
If at least three complete AI combinations survive, SKIP `shortlistRetrievalDeck`
entirely: use ONLY those original, source-verified AI outfits as photographic
NVIDIA critic candidates. On model timeout/invalid/insufficient
admission, invoke the existing complete legacy shortlist. No user-facing
unreviewed model decision, source substitution or extra quota.

New sanitized diagnostics: `skuCompositionAiFirst`,
`skuCompositionRetrievalSkipped`; exact `retrievalMode` is
`source_verified_ai_first` only when that source/physical validation has
succeeded. Automated regression tests assert that successful model decisions
reach photo review before the old ranker can run, and that failure restores
the original retrieval path. The long 16s model timeout and 72s request
budget remain unchanged. This is staging only.

**Do NOT count as fashion acceptance before a new pinned Render deployment
and two native iOS screenshots.** The primary benchmark, the logo, the
model API status and tests are not a human fashion panel.


## 2026-11-17 — retired visual-editor model replaced with LIVE vision AI

The optional visual portfolio editor was hard-coded to
`nvidia/nemotron-nano-12b-v2-vl`. An authenticated direct NVIDIA image
request on the LIVE provider now returns HTTP **410 Gone**; this was an
unavailable decision provider, not a successfully working luxury editor.
Official documentation and older tests do not prove current API availability.

Replaced only the optional source-photo portfolio editor model with
`nvidia/nemotron-3.5-super-vl-120b-a12b`, which is already the
**LIVE independent whole-look photographic critic** model. This is a separate
model call with a completely different final editorial prompt/portfolio
decision, not an independent model provider. It must still select one
server-verified F-token; it cannot invent an extra look or override the
earlier photo verdict.

Bounded, personally authorized synthetic-photo integration evidence:
- Direct NVIDIA full-image request to 120B-VL: **HTTP 200**, correctly
  identified the navy quadrant in a generated 2x2 image, valid JSON.
- After increasing the bounded vision-editor window from 5.5s to at most
  7.8s (the shared request deadline stays 9.6s), the real provider was
  exercised through the actual `selectPhotoEditorialDeck` code using a
  newly generated four-source-photo x three-look collage.
- **Real provider HTTP 200**, returned one allowed complete portfolio F01
  comprising two of three independently mocked photo-approved outfits;
  `visualJury:applied`, `visualMediaStatus:ready`,
  `providerCalls:1`, ~1.4s. Synthetic tests are proof the IMAGE/editor
  INTEGRATION functions, not proof these invented solid-color products
  are beautiful.
- The fallback-only text editor was previously used on most REAL native
  iOS logs; require a pinned new Render LIVE deployment and a real native
  3+ approved photograph scenario with both visual-jury flags before
  claiming the new editor works on true merchant outfit photographs.

The original independent NVIDIA visual critic is unchanged. No model
credentials, user data, product URLs or uploaded private photos are added
to reports.

## 2026-11-17 — PHOTO-FIRST creative composer, source-real merchandise

The first AI-FIRST LIVE native iOS test after 6f5a0c4 returned actual
`skuCompositionStatus:composed`, `skuCompositionAdmitted:6`,
`skuCompositionRetrievalSkipped:true`, 34,873ms, but only **2/9** full
outfits were photo-approved (2 displayed). This definitively separates
technical success from weak aesthetic choice by a TEXT-ONLY composer.

The second 6f5a0c4 successor b9b7863 went LIVE and performed
`photoEditorialVisualJury:applied` and
`photoEditorialVisualMedia:ready` on REAL merchant photographs:
Classic / Date night, 40,744ms, 6 AI-created, 5 source-admitted,
3/7 independent photo approvals, the VLM editor actually chose 2 compatible
four-product outfits. Viewed both on the booted native simulator:
- Dark merino shirt, brown chunky cardigan, navy trousers, burgundy loafers.
- Navy merino collared top, charcoal cardigan, beige trousers, brown Chelsea
  boots. Good physical four-SKU products, but architecture is similar:
  both knit+cardigan+trousers+leather shoes. Neither establishes an elite
  styling win.

**Next architecture (separate staging experiment)**: give the *CREATIVE*
AI an original merchant-product PHOTO BOARD before designing the complete
outfits, not just their names. Source-bound menu of up to 48 product tokens
(six or more per slot), four exact photograph columns and anonymous P-labels.
Every image must pass external inference source authorization. Missing/
unlicensed/invalid photo is excluded from allowed visual choices; if too few
source visuals exist, retain the existing independently verified TEXT
composition. The first actual vision VLM creates up to six distinct
source-token-unique complete 4-item outfits from a single JPEG collage.
Then the original physical+merchant+budget+occasion gates, independent
full-look photo critic and separate collection-level editor are applied.
Nothing constructed on the photographic board is automatically displayed.

Safety/performance:
- Request-bounded 48-image original-CDN fetch + 4.2s photo board and
  at most 6.7s hosted 120B VLM; original entire staging owner request
  remains 72s, independent critic paid quota unchanged, full text fallback.
- Reuse original photo fetch/cache from the cloud handler across designer
  and critic; no disk/user-photo persistence. Opaque tokens only.
- Native photo-board unit tests validate source permissions, missing image
  fail-closed, deadline, 32-source JPEG layout and no private identities.
- A REAL NVIDIA 120B vision API call with 24 completely synthetic product
  images returned 6 source-valid four-piece looks in **2,510 ms** using one
  provider call, with `visualMenuStatus:applied`. This proves the vision
  composition integration, NOT style performance on real catalog items.
- Stage log must show `skuCompositionPhotoFirst:applied` BEFORE anyone
  claims a photo-first personal stylist is already working on merchant
  photos. Always check actual native iOS silhouettes and product images.
