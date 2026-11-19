# Stylist 6 — learn which four-garment combinations deserve the NEXT photo evaluation

**Status: isolated staging feature, aesthetic acceptance NOT YET ACHIEVED.**
Four real source-bound products per generated look, original-photo independent
review and owner-authenticated staging transport remain mandatory. Production
Stylist 5 is unchanged.

## Actual problem observed

Real iOS 26.5 runs on 19 Nov were weak: Classic/Date night had 1/7
original-photo approvals, Classic/Work 3/8. The next new-SKU/photo-rescue
round had previously approved zero new looks. Repeated prompts and garment
blacklists did not solve taste. The recent owner-trained model was correctly
neutral because live Supabase showed **zero** explicit user outfit judgments.

## New operational decision learner

The independent original-photo judge now provides **a different supervised
training source**: scores and approvals/denials of complete four-product
looks in the *CURRENT* request. The new pure
`backend/stylist6/photo_feedback_teacher.mjs`:

- Is explicitly an **AI photo-critic teacher, NOT human preferences**.
  The separate human-trained owner preference model stays independent.
- Fits a bounded pairwise logistic model to source-grounded
  subtype/volume/color/texture/pattern features of complete four-role looks
  AND their cross-slot interactions. **No hand-written fashion-category
  prohibitions or positive/negative brand labels.**
- Only accepts a training label when both the actual independent photo
  critique and the existing full outfit approval/rejection decision agree.
  Conflicting or unobserved judgments cannot become ground truth.
- Requires at least two genuinely different original-photo approvals and
  three genuine photo rejections, plus a matching held-out positive/negative
  outfit pair whose ordering it learns correctly. Otherwise ABSTAINS.
  Ratings are in-memory and per request: no owner data, source SKU
  identifiers, original images, model outputs, private photo URLs or training
  weights are sent to any data store.
- After the **first** original-photo audition, may reorder a bounded menu
  of still-unreviewed, original-source-feasible **COMPLETE** outfits from the
  AI rescuer and ranker's authenticated candidates. It has a maximum of
  three new photo slots and never approves a look itself.
  Eligibility, budget, four-SKU admission, permitted image hosts and
  FINAL independent photographic veto continue unchanged.
- Reports only anonymous model status / examples / whether revised
  source-photo slots changed to QA. If insufficient signal, uses the exact
  prior untouched AI rescue order and inventory.

This aims to change WHAT is independently photographed in the corrective
round, instead of writing more style heuristics or lowering photo thresholds.
The teacher is still limited by incomplete garment visual facts (length not
calibrated) and is not a replacement for rights-approved original-photo
vision BEFORE the first composition.

## Evidence checklist

A synthetic controlled held-out fixture has balanced coat/jacket COLORS in
both positive and negative classes, so success requires learning cross-garment
relationships rather than a rule such as 'brown blazers preferred'.
Check: invalid/contradictory critic decisions, only one approved outfit,
original image URL/owner-ID leakage, no photo passes without external review,
no new garment bans, and original sequence unchanged on abstention.
The dedicated Stylist 6 CI includes these tests.

**Truthful release status:** improvement to real iOS original-photo approval
rates and independent blind human fashion taste must be checked AFTER the
exact staging deploy. This patch alone does not prove a beautiful 3/3 deck.
Binding release remains >=30 identical-SKU held-out scenarios, >=3 blinded
human judges, >=70% S6 preference over S5, and zero hard violations.

## Real original-photo native QA — exact deployed commits

- **2026-11-19 original new learner** commit `3b6558fe7e522f6e94b2a25711f2abe64b4f9f84`, isolated Render staging deployment `dep-dbv9nhjm8hqs739cpa4g`, LIVE. GitHub Actions run `44474589704` SUCCESS. Genuine authenticated iOS 26.5 Simulator Classic / Work, catalog/new clothing, S6 Cloud: the original source creator proposed **6** real combinations, **5** admitted, **6** independently photo-inspected in total, **2** photo-approved, **2** displayed, Render **54,275 ms**, `catalogFourItemContract=true`, `catalogFourItemDropped=0`. One bounded photo-informed source composition was rejected by the original-photo critic. The initial audit receipt did not include learner status, so this test does **NOT** prove the teacher affected the generated deck.
- **2026-11-19 instrumentation** commit `5ada60958c7ca1d2efe0e13ccefc7eeab7ab54ef`, staging `dep-dbv9serl550s7395ocng`, **LIVE**; GitHub Actions run `44475909423` SUCCESS. Adds ONLY anonymous, allowlisted `photoTeacherStatus`, bounded `photoTeacherExamples` and boolean `photoTeacherSelectionApplied` to the safe server trace. No model weights, original photos, merchant SKUs or owner identities are logged.
- **Exact instrumented Classic / Work iOS run:** after pressing Generate on the authenticated iOS QA form, the Maestro wait process exited nonzero before the app displayed its response, but **Render subsequently completed** and the iOS results screen independently showed **one photo-verified four-item outfit**. Exact Render receipt: **43,398 ms**, source creator **4 proposed / 3 admitted**, original photo **1 approved / 5 attempts**, **1 displayed**, photo-informed correction **2 attempts / 0 approved**, original-photo editor **not run** (insufficient photo-approved candidates), no fabricated third garment; final `catalogFourItemContract=true`, `catalogFourItemDropped=0`. The newly instrumented teacher reported `photoTeacherStatus='insufficient_independent_photos'`, `photoTeacherExamples=1`, `photoTeacherSelectionApplied=false`. It therefore correctly **ABSTAINED** rather than learning bogus taste from one approval.

**Interpretation:** The new code is correctly deployed and safe, but **no observed improvement in real fashion beauty**. In these autonomous generations, there are too few independent approved PHOTO outfit examples for an in-request binary teacher to generalize. Both canaries were stochastic and did **not** share the exact same source-SKU cohort, so comparing 2/6 to 1/5 is NOT a causal A/B experiment. The next modeling step needs either a safely retained, provenance-bounded set of photo-critic judgments across requests (separate from explicit human preferences), or a calibrated **continuous / pairwise photographic-quality teacher** that can learn from graded verified photo scores even with one approved look, while preserving held-out checks. Do not weaken the final real-original-photo aesthetic veto or invent rights to pre-composition merchant images.

**Release gate:** AESTHETIC NO-GO; user goal (three outstanding, distinct, personalized four-product looks) remains unmet. No promotion of Stylist 6 and no changes to production Stylist 5. No fabricated human ratings or photo rights.

## LIVE iOS quality feedback and graded-photo learning correction (19 Nov, exact 5ada6095)

The first monitored authentic iOS Classic / Work request on isolated staging
`5ada60958c7ca1d2efe0e13ccefc7eeab7ab54ef` returned
**1 / 5 PHOTO-approved**, **1 displayed**, 43,398 ms, 4 legitimate
merchant SKUs. Photo-informed SKU retry 2 attempts, 0 approvals. The
newly exposed truthful telemetry reported:
`photoTeacherStatus=insufficient_independent_photos`,
`photoTeacherExamples=1`, `photoTeacherSelectionApplied=false`.
Thus the earlier teacher *did not actually change candidate selection*,
and **did not improve aesthetic quality**. No editor-selected third.

Analysis exposed a real implementation defect, not the need for a new
fashion rule: a full-outfit NVIDIA verdict may have `reject=false` yet
still fail the original independently established style, occasion, formality
or overall photographic quality gate. These were legitimate photographic
**negative examples** but incorrectly dropped from preference training.

The corrected trainer accepts these source-bound genuinely photographed
negative decision reasons **only** when the same candidate has a valid
independent photo critique, while still excluding uncertain photos and
user-specific avoid-color vetoes. It does NOT reinterpret a bad photo as
approved or transfer this model's training to owner taste.

With **one real PHOTO approval + four or more PHOTO rejections**, the trainer
can additionally learn from the *graded original photo quality among the
rejected outfits* rather than waiting for a second full approval. It holds
out a matched high-quality/low-quality rejected pair, trains on the remaining
original-photo comparisons, and can only reorder a few *unreviewed* source-
valid next-round candidates when it predicts the held-out order correctly.
A failed held-out comparison causes honest abstention. This is a much
weaker supervised signal than blind **human** outfit preferences and does
not count toward their acceptance metric.

A real-data training verdict status now distinguishes `photo_validated`
(two or more true approved photo examples),
`graded_photo_validated` (photo quality learned **within rejected** sets),
`insufficient_independent_photos`, `uncalibrated_on_photos`, and related
safe abstention reasons. The safe Render log also exposes anonymous bounded
positive/negative counts, but **no user IDs, product IDs, photos, inference
weights or merchant URLs**.

Live visual evidence remains the authority: the next deployed canary must
confirm whether the teacher actually validated **and** whether subsequent
photo-approved outfit rates and independent human beauty improved. Existing
physical source authenticity, 4-SKU, merchant-rights and original-photo
critic requirements remain. This is NOT a production-ready sign-off.

Catalog data caveat from read-only Supabase at this checkpoint: 8,554 active
products, 7,556 visual profiles, only 1,093 visible volume attributes and
1,838 texture attributes. Missing photo evidence is not known fashion
information and may limit learning; building a properly authorized original-
photo first-pass fashion vision process remains strategically necessary.

**Additional real-engine integration regression:** in an authenticated-source
fashion-deck fixture with one independently photographed winner and multiple
independently photographed failures, the first actual rescue callback now
receives one genuine positive and at least three rejected photo comparisons.
The test calls the real fashion-deck entrypoint, not only a standalone
teacher. It verifies that one positive is **not** falsely labelled
`photo_validated` and that all displayed source SKUs still undergo actual
follow-up photo evaluation. The graded model can instead report its stricter
`graded_photo_validated` outcome if its real PHOTO holdout succeeds.

## Durable, **photo-AI-labelled** fashion reward learning across QA requests

The on-request teacher still has very few reliable examples (real Work
iOS canary on `eaeb8799`: **2** independently PHOTO approved, **3** rejected
out of **8** photo attempts, +3 editor retries with 0 approvals; internal
teacher **uncalibrated_on_photos**, left the source selection unchanged).
This is NOT a human taste win.

To avoid retraining from just five examples on every request, an additional
privacy-preserving original-photo training *journal* is now available on
isolated staging. Its purpose is collecting verified **full four-product**
outfit outcome labels that could support a properly cross-validated learned
fashion reward model after enough independent tests. It does NOT yet
instantiate such a trained production model.

- Pure compiler `backend/stylist6/photo_reward_journal.mjs`: only
  internally source-admitted 4-SKU **catalog** outfits, one real photo
  critique matched to the same inspected candidate, and the already
  independently adjudicated `approved` / photo quality / style / occasion /
  formality / model-veto results qualify. User color-exclusion vetoes and
  uncertain photos never become universal fashion lessons.
- **No raw attributes stored:** source-specific outfit identity, photo
  pixels/URLs, product/SKU IDs, merchant names, request/user IDs, brands,
  original photo descriptions, personal prompt and arbitrary context strings
  are NEVER persisted. Only server-secret-keyed SHA-256 HMACs of 4-slot
  semantic feature interactions and a closed two-field style/occasion
  classification, plus numeric actual-photo quality and closed reason,
  are stored. No free-form user input or personal outfit preference.
- New Supabase table `public.stylist6_photo_reward_examples` created by
  hosted migration `20261119073412_stylist6_photo_reward_journal_v1`;
  RLS enabled; `anon` and `authenticated` have NO SELECT or INSERT
  privilege; service_role only SELECT/INSERT/DELETE. Exactly 30-day
  default TTL, pg_cron deletion every six hours. SQL verified grants,
  active cleanup and **zero seeded example rows** after migration.
- A bounded and optional staging-only callback in `recommendFashionDeck`
  receives the **first real independent original-photo review** and calls
  a service-role writer only when `STYLIST6_PHOTO_REWARD_JOURNAL_ENABLED=true`,
  the server has 72s staging budget, mode is pure shop catalog, no pinned
  owned items, and no explicit owner color/pattern/fit preferences. Client
  supplies no training labels and cannot enable it. Failure to persist
  never changes a paid response, source admission, four-SKU completeness,
  independent photo acceptance or production Stylist 5 behavior.
- Bounded 8-row best-effort batch; duplicate privacy-hashed signatures
  deduplicate; record storage uses the same privileged Supabase service
  client and a 1.2 second hard deadline. The writer has a second closed
  validation boundary to reject extra fields such as `user_id`, URL
  or photos if the compiler were ever bypassed.
- Source HMAC key is derived in-memory from the existing **server-only**
  staging signing secret; never saved in SQL, logs or QA reports.
  Rotating it invalidates old hashed token matches, deliberately.

**This is infrastructure, not the North Star result.** The next step is
enable opt-in only on isolated Render staging, verify genuine catalog PHOTO
example rows arrive without personal fields, and train a later **cross-request
held-out** fashion preference model without mixing AI photo judgments with
real independent human votes. Original JPEG-before-first-composition still
requires licensed rights and remains a separate open issue. The >=30 frozen
four-SKU contexts / 3 blind humans / >=70% stylist-6 preference gate remains.
