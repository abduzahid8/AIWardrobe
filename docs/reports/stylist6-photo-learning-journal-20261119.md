# Stylist 6 — real original-photo experience archive

2026-11-19. **Engineering capability, NOT an aesthetic victory.** The user
needs THREE different beautiful personalized FOUR-product looks, not a
higher number of outputs or a list of hardcoded garment/style prohibitions.

## Why this change exists

An authenticated iOS Classic/Work native QA on Render
`eaeb87992d55ec5cd7a9c6669d9af61690493f7e` at 07:25 UTC returned
2 photographed approvals of 8 real photo attempts, 2 displayed,
59.1s server compute. The *in-request* photographic teacher had two
approved and three rejected annotated outfits but reported
`uncalibrated_on_photos`, so it correctly ABSTAINED and left selection
unchanged. A second AI rescue had zero approvals out of three.
**Five real examples cannot reliably train whole-outfit judgement.**

## Private, separable, durable PHOTO training evidence

The existing photo judge's evidence can now be accumulated only in
the isolated >=72s Stylist 6 staging route, through an explicitly
server-enabled `STYLIST6_PHOTO_REWARD_JOURNAL_ENABLED=true` gate.

- `photo_reward_journal.mjs` derives a source-qualified, anonymized
  four-garment sample after the **independent original-photo** critic and
  the unchanged quality/occasion/style/photo gate have genuinely judged it.
  A photo quality failure with `reject=false` remains a *negative*.
  Color exclusions and low confidence are not taught as fashion dislike.
  Owned wardrobe and fixed anchors are excluded. Personalized catalog
  fashion preferences are partitioned by a distinct keyed context HMAC:
  full supported color, fit, pattern, brand and style goals are never
  stored in the clear or treated as universal garment bans.
- The sample includes ONLY seven bounded fields: keyed HMAC-SHA256
  fingerprint and style/occasion/fit context digest, up to 180 keyed HMAC
  hashes of observed four-slot garment and between-slot relationships,
  actual numeric photo quality 0–1, photo outcome +1/-1, fixed nonpersonal
  reason enum, schema version. **No user/request identity, SKU, brand,
  shop or image URL, product description, prompt or raw photograph**.
- The signature uses server-only domain-separated HMAC;
  no raw feature text or unkeyed hash goes to the database.
  Only six supported style codes and six supported occasion codes and
  bounded, permitted app preference fields are allowed as distinct
  HMAC-separated contexts. The user's free-text request, identifiers,
  images and body metadata never become persisted training attributes.
  App fit modes balanced/loose/fitted are kept separate. The QA app previously had a loose-fit
  setting that erroneously made an otherwise safe test untrainable.
- `createCloudIo().savePhotoRewardExamples` is service-role only,
  validates every field/row again, uses bounded 1.2-second upsert,
  and is OPTIONAL after the photo review. Failed writes cannot fail
  the paid image-verified deck. It does **NOT** train the stylist yet.
- Deployed SQL migration `20261119073412_stylist6_photo_reward_journal.sql`
  creates `stylist6_photo_reward_examples` with Postgres RLS, no
  `anon` or `authenticated` grants, service_role-only access, an
  expiry timestamp no later than 31 days, and a cron cleanup every
  six hours. Verified directly in live Supabase: RLS=true,
  anon_select=false, authenticated_select=false, service_select=true,
  active retention=true, rows=0 before this writer deploy.
- Test suite includes a real fashion-engine callback, client-access and
  write-validation tests, negative-case photoreview test, stable
  pseudonymization within a server key, and no private/merchant byte leakage.
  The S6 workflow explicitly includes these tests.

## Separation of kinds of learning

`stylist_look_feedback` = explicit real HUMAN preferences and owner-only
personal taste. Currently zero real rows.

`stylist6_photo_reward_examples` = independently AI PHOTO-reviewed
candidate combinations and graded quality, NOT owner likes,
NOT guaranteed human beauty, NOT a licensing grant to use raw image pixels
in first-pass composition.

These two sources MUST NEVER be joined as if the AI juror were the owner.
This journal is consent/context-restricted and retained <=30 days.
It never forces any additional card or bypasses the original image critic.

## Next required work, not new garment rules

1. Deploy an exact CI-green GitHub SHA to isolated Render staging, enable
   this journal flag **only there**, run the real authenticated iOS
   Work/Date photo test, verify new rows and safe no-ID schema. Do NOT
   claim samples exist before verifying the database.
2. On a sufficient genuine non-duplicated photo history from fixed
   source inventory, train a *separate* general fashion prior on
   pseudonymous outfit relationships. Require independently held-out
   evidence and fail-neutral on scarce/unreliable labels.
3. Use the reliable learned comparison ONLY to prioritize feasible
   source-SKU candidates BEFORE the next independent photo audition,
   never as a self-issued photo approval, a blocklist or an override of
   the user's explicit tastes.
4. `photo_reward_history.mjs` now reads only prior 30-day original-photo
   evidence from the private service role, groups it by protected
   style/occasion/fit context, deduplicates contradictory signatures,
   learns **pairwise four-garment photographic decisions** and checks
   2×2 held-out photo comparisons before allowing the trained score to
   prioritize real, still-unreviewed source-verified challengers. It
   requires >=6 independent positive plus >=10 negative photo examples
   and a perfect 4/4 held-out match or **abstains**. No garment blacklist,
   no source, budget or dress-code override and NO self-issued photo pass.
   With only a handful of real samples the score is neutral and the
   earlier ranking order stays unchanged.
5. Final business gate remains blind human comparisons on 30+ matched
   contexts x 3 humans, >=70% S6 preference over S5, three aesthetically
   strong distinct looks of four real products, zero hard integrity
   violations. Production Stylist 5 is unchanged.

## Real canary and honest status

At 2026-11-19 07:59 UTC, the first photographed Classic/Work canary on
the newly-enabled journal staging branch returned **2 original-photo
approvals / 6**, **2 displayed**. Its server did not report a successful
journal write and a verified live Supabase count stayed **0**, not a
training dataset. We traced a real potential data rejection to the app's
valid `loose` personal fit choice (only `balanced` had been allowed);
the now fixed context HMAC keeps `loose`, `fitted`, and `balanced`
separate. The next exact-sha live canary MUST prove new private rows
before claiming trained fashion skill. The safe diagnostic now emits
`photoJournalScope` and an enum `recorded` /
`no_anonymous_source_photo_examples` / `write_failed` if a safe
journal capture was attempted. It never logs a user's fit or identity.

## Root-cause fix verified after e24225ee

2026-11-19 09:07 UTC LIVE iOS Classic/Work on exact `e24225ee`
returned 0 independently approved outfits from 7 photos; all photo
quality gates remained intact. The first bug (Node route silently
dropping `STYLIST6_PHOTO_REWARD_JOURNAL_ENABLED`) was FIXED in
`e24225ee`; new safe telemetry reported
`photoJournalScope=personal_context_excluded`, proving that the flag
did reach the cloud engine and that privacy exclusion was the next
obstacle. DB still had **zero** real rows; a 0/7 photo verdict is not
an excuse to invent training examples.

The updated journal design accepts *only the explicitly authorized*
source-authentic catalog personal styling context and HMAC-hashes all
bounded, normalized, known preference fields (not SKU IDs, customer ID,
photos, URLs or free-text brief). Writer and reader derive the SAME
context hash from the full authenticated server request. Scores never
cross different color/fit/style-goal contexts, and never turn a
photographic AI veto into an owner preference. Only allowed catalog
four-item photo verdicts enter the 30-day RLS-private journal.
