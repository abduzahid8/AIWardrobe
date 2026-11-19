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
