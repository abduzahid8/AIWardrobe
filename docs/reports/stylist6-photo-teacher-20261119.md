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
