# Stylist 6 — learning whole-outfit decisions, not adding garment rules

**Control point:** 19 Nov 2026. Goal unchanged: THREE genuinely stylish, distinct, personalized ensembles, each with exactly FOUR real source-bound products. **Fashion quality is NOT yet accepted.**

## Real source of truth

Read-only Supabase query on `public.stylist_look_feedback`: **0 actual rows** as of this change. Therefore no production owner taste model has learned anything from real human preference yet. We must NOT manufacture positive/negative user likes from NVIDIA photo-critic verdicts, infer owner taste from taps, or pretend offline unit-test fixtures are taste judgments.

The mobile app already supports owner thumbs on the actual displayed looks. The server already queries the authenticated user's own feedback rows, bounded to the 60 most recent, and enforces owner isolation. The new algorithm uses only that existing source.

## Code: personal outfit-preference learner

- `backend/stylist6/learned_outfit_preference.mjs`: pure deterministic **pairwise logistic preference learning** over the entire four-slot outfit. Training data are explicitly human-rated full-outfit **+1/-1** rows with existing `look_signature.version=1` and source-current styling descriptors.
- Learns both individual attributes and **cross-garment combinations** (e.g. top x layer, layer x trousers, trousers x footwear), not a hand-coded "cardigans bad, navy good" rule or item-block list. Anonymous categorical semantics only: four garment roles x (subtype, volume, color, texture, pattern). No user ID, private SKU IDs, image URLs, original photos, raw text, brand names or model-generated photo vetoes are training features.
- Deduplicates identical signatures; contradictory signed ratings are discarded. Age limit and 60-row cap prevent unbounded training.
- Before allowing cross-SKU prediction, requires at least **6 distinct positive AND 6 distinct negative** owner-rated looks. Two positives and two negatives are held out to test the learned model's ordering. If held-out comparison fails, it **abstains** and has no learned styling effect. This is a minimum **evidence standard**, not a fashion rule.
- Existing `feedback_taste.mjs` per-product exact owner dislikes/likes and feedback shown to the creative AI remain. But the old hard-coded weighted `styleSimilarity` rule is removed: **a single like no longer automatically biases unrelated SKU looks simply because two subtypes match**.
- The calibrated learned preference influences already feasible real full outfits as a **soft ±0.02 component** inside the existing ±0.06 personal taste cap; it is never a product veto, original-photo approval, stock assertion or override of explicit dislikes.
- `cloud_handler.mjs` / `fashion_deck.mjs` now expose only a non-identifying status (e.g., `untrained`, `insufficient_human_comparisons`, `uncalibrated_on_holdout`, `validated_human_preferences`) so QA can verify whether the learning policy is genuinely active.

## Verified, without passing synthetic taste off as human approval

The unit fixtures deliberately balance individual garment colors between positive/negative classes. A classifier relying on a predefined coat/color rule cannot solve the task: distinguishing examples requires the **learned cross-slot relationship**. Tests also cover new SKU IDs, missing/hostile inputs, all-like imbalance, contradictory labels, order invariance and correct neutral behavior with zero labels.

`backend/stylist6/learned_outfit_preference.test.mjs` is included in the dedicated Stylist 6 CI matrix; existing taste/cloud tests updated for the deliberate version and neutral transfer semantics.

**Important limit:** This is a preference-learning capability installed in the **real selection scorer**, not a claim that a trained human-calibrated fashion model exists today. At 0 real ratings it stays neutral. The model uses typed, already extracted visual garment attributes, not raw original JPEGs. Those attributes may be incomplete; source-pixel-first inference remains conditional on suitable merchant image rights. Nothing here weakens the independent original-photo veto or forces three bad cards.

## Evidence required next

1. Obtain authentic thumbs from the actual outfit cards: at least six *distinct* positive and six *distinct* negative user judgments before enabling transferable owner taste. Do not have an AI agent impersonate a person or auto-fill reviews.
2. On the exact frozen permitted SKU/photo cohort, verify that the model generalizes to withheld user-rated looks; compare candidate ordering before and after. Verify model status is **validated**, not just `active`.
3. Separate algorithmic *photo-critic* labels from actual blinded human aesthetic evaluations. To learn general fashion judgement (rather than one owner's taste), build a consented, private outcome dataset with a documented training/evaluation split and explicit provenance.
4. Keep the binding 30+ held-out context, 3-human-reviewer, >=70% blind S6-over-S5 preference gate; zero hard breaches. Stylist 5 production remains unchanged.

**No merchant photographs, owner identities, private SKU tokens, fabricated ratings or credentials added to this report.**
