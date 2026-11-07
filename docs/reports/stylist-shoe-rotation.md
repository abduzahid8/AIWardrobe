# Personal Stylist: minimum-overlap shoe rotation

## Source

- Branch: `stylist-four-slot-recovery`
- Starting HEAD: `26a940fbd41e2f20b160fa37022779c9720cbf36`
- Existing dirty report preserved in checkpoint commit `7ecb117`; a separate binary patch was also saved before edits.
- Final implementation HEAD: `015e235` (the following documentation commit contains this report and evidence).
- Edge: **v166**, active. All 16 downloaded TypeScript files match the tested checkout byte-for-byte.
- Metro: the existing recovery checkout on port **8082**. No second Metro was started.

## Shoe pool

Request: **New clothing / Old Money / Date Night / no anchor**. No explicit budget or weather override was added. The real screen shows weather unavailable.

- The exact active catalog slot query returns **59 rows**. Six are boot-cut trousers caught by the broad `%boot%` lookup and rejected before styling.
- **53 actual footwear rows**, representing **50 physical shoes** under the unchanged ID / normalized-name / normalized-image contract. Duplicate trainer listings explain the row/product difference.
- Before the fix: **5** admitted physical shoes.
- After checking actual photos and correcting two false negatives: **7** admitted physical shoes.
- Renegade Black, Statesman Brandy, and President StormKing Bull Brown remain rejected for rugged lug construction. Sneakers and espadrilles remain rejected.

### Admitted shoes

| Product | ID | Evidence |
|---|---|---|
| Duke / Black Coffee | `shop_thursday-15456871514202` | Existing admitted refined leather footwear |
| Executive / Black | `shop_thursday-253390159881` | Existing admitted refined leather footwear |
| Chairman / Black | `shop_thursday-6859648467034` | Existing admitted refined leather footwear |
| Lincoln / Black | `shop_thursday-6865670045786` | Structured penny strap, leather outsole and Blake stitching; photo shows a low dress heel, not driving nubs. “Moccasin toe” describes stitching. |
| Journeyman / Black | `shop_thursday-7719429898330` | Existing admitted refined leather footwear |
| Saint / Cognac | `shop_thursday-8086847258714` | Existing admitted refined leather footwear |
| Lincoln / Dark Brown | `shop_thursday-8128950501466` | Structured penny strap, leather outsole and Blake stitching; photo shows a low dress heel, not driving nubs. “Moccasin toe” describes stitching. |

Photographic audit: [current catalog shoe photos](stylist-shoe-rotation-evidence/shoe-photos.jpg). No general image-classification system was added.

### Every rejected catalog candidate

The table lists each rejected row, including duplicate physical listings. Full ID/name/image equivalence keys and both pre/post-fix reasons are in `pool-before.json` and `pool-after.json`.

| Product | Physical key and row ID | Rejection reason |
|---|---|---|
| Regular-Fit Bootcut Jeans | `n:regular fit bootcut jeans` / `shop_allsaints-usm060ef-5` | Slot metadata/name resolves to bottom |
| Vini Slim Bootcut Leather Pants | `n:vini slim bootcut leather pants` / `shop_allsaints-usm060tf-5` | Slot metadata/name resolves to bottom |
| Regular-Fit Bootcut Jeans | `n:regular fit bootcut jeans` / `shop_allsaints-usm084ef-437` | Slot metadata/name resolves to bottom |
| Leather espadrilles | `n:leather espadrilles` / `shop_md-uz-62899795` | beach/athletic shoes are not evening wear |
| Split suede leather espadrille with piece | `n:split suede leather espadrille with piece` / `shop_md-uz-62899796` | beach/athletic shoes are not evening wear |
| Era / Cloud | `n:era cloud` / `shop_thursday-15407719448666` | beach/athletic shoes are not evening wear |
| Renegade / Black | `n:renegade black` / `shop_thursday-6794356228186` | rugged lug footwear conflicts with refined evening clothing |
| Era / Navy | `n:era navy` / `shop_thursday-7824210231386` | beach/athletic shoes are not evening wear |
| Statesman / Brandy | `n:statesman brandy` / `shop_thursday-7887606677594` | rugged lug footwear conflicts with refined evening clothing |
| President StormKing® / Bull Brown | `n:president stormking bull brown` / `shop_thursday-8140117377114` | rugged lug footwear conflicts with refined evening clothing |
| RETRO STYLE TRAINERS | `n:retro style trainers` / `shop_zara-uz-545422164` | beach/athletic shoes are not evening wear |
| TRAINERS WITH HEEL DETAIL | `n:trainers with heel detail` / `shop_zara-uz-545422170` | beach/athletic shoes are not evening wear |
| LEATHER TRAINERS | `n:leather trainers` / `shop_zara-uz-545422196` | beach/athletic shoes are not evening wear |
| MONOCHROME CHUNKY TRAINERS | `n:monochrome chunky trainers` / `shop_zara-uz-545422214` | beach/athletic shoes are not evening wear |
| CASUAL TRAINERS | `n:casual trainers` / `shop_zara-uz-545423068` | beach/athletic shoes are not evening wear |
| RETRO LEATHER TRAINERS | `n:retro leather trainers` / `shop_zara-uz-545426131` | beach/athletic shoes are not evening wear |
| BAREFOOT LEATHER TRAINERS | `n:barefoot leather trainers` / `shop_zara-uz-545436416` | beach/athletic shoes are not evening wear |
| PADDED TRAINERS | `n:padded trainers` / `shop_zara-uz-545437444` | beach/athletic shoes are not evening wear |
| CASUAL LEATHER TRAINERS | `n:casual leather trainers` / `shop_zara-uz-545456761` | beach/athletic shoes are not evening wear |
| LEATHER COMBINATION ESPADRILLES | `n:leather combination espadrilles` / `shop_zara-uz-545461640` | beach/athletic shoes are not evening wear |
| CHUNKY TRAINERS | `n:chunky trainers` / `shop_zara-uz-545465740` | beach/athletic shoes are not evening wear |
| CHUNKY SOLE TRAINERS | `n:chunky sole trainers` / `shop_zara-uz-545465742` | beach/athletic shoes are not evening wear |
| BAREFOOT TRAINERS | `n:barefoot trainers` / `shop_zara-uz-545479433` | beach/athletic shoes are not evening wear |
| MINIMALIST TRAINERS | `n:minimalist trainers` / `shop_zara-uz-545479492` | beach/athletic shoes are not evening wear |
| SKATE TRAINERS | `n:skate trainers` / `shop_zara-uz-545479606` | beach/athletic shoes are not evening wear |
| LOW-RISE BOOT-CUT JEANS | `n:low rise boot cut jeans` / `shop_zara-uz-545490695` | Slot metadata/name resolves to bottom |
| LEATHER COMBINATION TRAINERS | `n:leather combination trainers` / `shop_zara-uz-549311970` | beach/athletic shoes are not evening wear |
| CASUAL LEATHER DETAIL TRAINERS | `n:casual leather detail trainers` / `shop_zara-uz-549311972` | beach/athletic shoes are not evening wear |
| RETRO LEATHER DETAIL TRAINERS | `n:retro leather detail trainers` / `shop_zara-uz-550775793` | beach/athletic shoes are not evening wear |
| KNITTED ESPADRILLES | `n:knitted espadrilles` / `shop_zara-uz-554807825` | beach/athletic shoes are not evening wear |
| HEEL DETAIL TRAINERS | `n:heel detail trainers` / `shop_zara-uz-554808166` | beach/athletic shoes are not evening wear |
| RUNNING TRAINERS | `n:running trainers` / `shop_zara-uz-554808503` | beach/athletic shoes are not evening wear |
| COMBINED RETRO TRAINERS | `n:combined retro trainers` / `shop_zara-uz-555113523` | beach/athletic shoes are not evening wear |
| FABRIC ESPADRILLES | `n:fabric espadrilles` / `shop_zara-uz-557456590` | beach/athletic shoes are not evening wear |
| CAPRA TRAIL ZERO RUNNING TRAINERS | `n:capra trail zero running trainers` / `shop_zara-uz-557850347` | beach/athletic shoes are not evening wear |
| GECKO TRAIN CROSS TRAINERS | `n:gecko train cross trainers` / `shop_zara-uz-557850352` | beach/athletic shoes are not evening wear |
| RETRO TRAINERS | `n:retro trainers` / `shop_zara-uz-558185354` | beach/athletic shoes are not evening wear |
| BOOT-CUT FIT JEANS | `n:boot cut fit jeans` / `shop_zara-uz-560271677` | Slot metadata/name resolves to bottom |
| VINTAGE-EFFECT LOW-RISE BOOT-CUT JEANS | `n:vintage effect low rise boot cut jeans` / `shop_zara-uz-561289604` | Slot metadata/name resolves to bottom |
| LACELESS TRAIL RUNNING TRAINERS | `n:laceless trail running trainers` / `shop_zara-uz-563369275` | beach/athletic shoes are not evening wear |
| BASKET TRAINERS | `n:basket trainers` / `shop_zara-uz-569277598` | beach/athletic shoes are not evening wear |
| LEATHER ESPADRILLES | `n:leather espadrilles` / `shop_zara-uz-576910913` | beach/athletic shoes are not evening wear |
| BASKETBALL TRAINERS | `n:basketball trainers` / `shop_zara-uz-576910916` | beach/athletic shoes are not evening wear |
| COMBINED LEATHER TRAINERS | `n:combined leather trainers` / `shop_zara-uz-576983279` | beach/athletic shoes are not evening wear |
| TRAINING TRAINERS | `n:training trainers` / `shop_zara-uz-577613045` | beach/athletic shoes are not evening wear |
| FABRIC TRAINERS | `n:fabric trainers` / `shop_zara-uz-577613080` | beach/athletic shoes are not evening wear |
| LEATHER COMBINATION TRAINERS | `n:leather combination trainers` / `shop_zara-uz-580723151` | beach/athletic shoes are not evening wear |
| AARON LEVINE X ZARA LEATHER TRAINERS | `n:aaron levine x zara leather trainers` / `shop_zara-uz-582244377` | beach/athletic shoes are not evening wear |
| CONTRAST BAREFOOT TRAINERS | `n:contrast barefoot trainers` / `shop_zara-uz-587323771` | beach/athletic shoes are not evening wear |
| TRAIL RUNNING TRAINERS | `n:trail running trainers` / `shop_zara-uz-589405020` | beach/athletic shoes are not evening wear |
| LACELESS RUNNING TRAINERS | `n:laceless running trainers` / `shop_zara-uz-589405021` | beach/athletic shoes are not evening wear |
| RETRO LEATHER TRAINERS | `n:retro leather trainers` / `shop_zara-uz-590145002` | beach/athletic shoes are not evening wear |

## Root cause

1. **False negatives at admission and ranking.** The driving-moccasin regex treated “Hand-Stitched Moccasin Toe” as a driving shoe, despite explicit penny-loafer / Blake-stitch / leather-outsole evidence. The same broad match existed in a later taste filter. Both stages now use the shared construction-aware shoe predicate. This admits the two photographed structured loafers without admitting actual driving soles or rugged lug footwear.
2. **Shoe identity lost during catalog sorting.** The lightweight row identity omitted `macroCategory`. Vague actual names such as “Lincoln / Dark Brown” were then cached as non-shoes by the rotation scorer. The fetcher now passes the slot, and the cache stores history facts rather than a permanent shoe/clothing classification. There are regressions for an identity-only pre-ranking call followed by a fully typed shoe.
3. **Final ranking and bounded choices.** Scalar total freshness could let freshness in other slots outweigh shoe recency, and last-use ties had no appearance-count tie-break. A bounded complete-look shortlist could also omit a valid shoe assignment. Ordered rotation objectives now precede quality tie-breaks, and a final exhaustive assignment considers the small admitted shoe pool for the three already selected looks. Every substitution passes the existing complete quality/style/occasion/weather/budget/compatibility gate. Clothing products do not change during this assignment.

## Rotation algorithm

- Minimum consecutive overlap with K=3 is `max(0, 2*K - N)` when the previous deck also has three admitted shoes. The first generation has no prior-deck requirement; a changed admissible pool uses the actual prior admitted count.
- Lexicographic objective: minimize previous shoe overlap; minimize previous overlap in other slots; prefer oldest shoe last-use; break equal last-use by fewer recent appearances; then apply existing clothing recency, outfit quality and visual tie-breaks. No random shuffle is used for shoe scheduling.
- Fresh shoes have never-used status and precede older history. Older history precedes the immediate previous deck. History boundaries, retention, scope, request IDs and physical identity are unchanged.
- The final audit records the physical pool, previous shoe IDs, selected IDs, fresh supply, theoretical minimum, and minimum feasible overlap for the accepted complete combinations. It rejects an assignment exceeding the feasible minimum. The existing client guard prevents normalization from reintroducing additional overlap.

## Stage trace

The Edge emits `[shoe-rotation-audit]` for each successful Personal Stylist response. It includes fetched, ranked, complete-candidate, pre-assignment and final shoe IDs. Supabase log retrieval returned the following sampled final-run record; the five displayed decks are independently captured from the app’s persisted accepted-deck history.

| Request | Fetched shoes | Ranked shoes | Complete-candidate shoes | Pre-assignment shoes | Final shoes | Pool / minimum / actual overlap |
|---|---:|---:|---:|---:|---:|---|
| `mvttvzy7-5` | 7 | 7 | 7 | 3 | 3 | 7 / 0 / 0 |

Exact IDs and all physical keys for the trace are in `server-audits-final.json`. The hard-admitted alternatives survive the final ranking stage; the three-look shortlist is not treated as the entire shoe inventory.

## Real acceptance: five consecutive Generate presses

The complete Maestro flow asserts **Old Money + Date night** before generation and does not tap an already-selected occasion off. Source and selections stay unchanged. Existing history is retained, not cleared. `recordDisplayedDeck` persists only the final accepted user-visible products after client repair, quality, budget and deck selection. The audit reads those records without a debugger observer and compares ID/name/image keys. Screenshots are captured for all five generations.

### G1

Request ID: `mvttu1vr-1`

1. Journeyman | Black — `shop_thursday-7719429898330`
2. Lincoln | Dark Brown — `shop_thursday-8128950501466`
3. Executive | Black — `shop_thursday-253390159881`

Clothing overlap with the preceding accepted deck: **0**. Independent shoe LRU/frequency-order comparison against every 3-of-7 set: **PASS**.

### G2

Request ID: `mvttujgg-2`

1. Saint | Cognac — `shop_thursday-8086847258714`
2. Lincoln | Black — `shop_thursday-6865670045786`
3. Duke | Black Coffee — `shop_thursday-15456871514202`

Overlap G1 → G2: **0**.

Clothing overlap with the preceding accepted deck: **0**. Independent shoe LRU/frequency-order comparison against every 3-of-7 set: **PASS**.

### G3

Request ID: `mvttv129-3`

1. Chairman | Black — `shop_thursday-6859648467034`
2. Lincoln | Dark Brown — `shop_thursday-8128950501466`
3. Journeyman | Black — `shop_thursday-7719429898330`

Overlap G2 → G3: **0**.

Clothing overlap with the preceding accepted deck: **0**. Independent shoe LRU/frequency-order comparison against every 3-of-7 set: **PASS**.

### G4

Request ID: `mvttvjqf-4`

1. Executive | Black — `shop_thursday-253390159881`
2. Lincoln | Black — `shop_thursday-6865670045786`
3. Duke | Black Coffee — `shop_thursday-15456871514202`

Overlap G3 → G4: **0**.

Clothing overlap with the preceding accepted deck: **0**. Independent shoe LRU/frequency-order comparison against every 3-of-7 set: **PASS**.

### G5

Request ID: `mvttvzy7-5`

1. Saint | Cognac — `shop_thursday-8086847258714`
2. Lincoln | Dark Brown — `shop_thursday-8128950501466`
3. Chairman | Black — `shop_thursday-6859648467034`

Overlap G4 → G5: **0**.

Clothing overlap with the preceding accepted deck: **0**. Independent shoe LRU/frequency-order comparison against every 3-of-7 set: **PASS**.

- Valid physical shoe pool: **7**.
- Theoretical minimum overlap: **0**.
- Actual maximum consecutive overlap: **0**.
- All seven admitted shoes used: **yes**, by G3.
- Stable rotation scope and unique per-tap request IDs: **PASS**.

## Controlled pool cases

| Valid physical shoes | Consecutive minimum after G1 | Observed over eight generations | Pinned shoe |
|---:|---:|---|---|
| 3 | 3 | 3, 3, 3, 3, 3, 3, 3 | All three; unavoidable |
| 4 | 2 | 2, 2, 2, 2, 2, 2, 2 | None |
| 5 | 1 | 1, 1, 1, 1, 1, 1, 1 | None |
| 6 | 0 | 0, 0, 0, 0, 0, 0, 0 | None |
| 10 | 0 | 0, 0, 0, 0, 0, 0, 0 | None |
| 50 | 0 | 0, 0, 0, 0, 0, 0, 0 | None |

Five-shoe appearance counts over eight decks: `{"shoe-0": 5, "shoe-1": 5, "shoe-2": 5, "shoe-3": 5, "shoe-4": 4}`. Counts differ by at most one; no favorite remains pinned. Hosted handler tests separately exercise five actual wire-level generations for pools of 3, 4, 5, 6 and 10 shoes. Unit assignment tests also cover 7 and 50, duplicate row/name/photo identities, hard quality rejection, an attractive favorite, and the partial-identity cache bug.

## Regression verdict

| Check | Result |
|---|---|
| Three cards in every real generation | PASS |
| Four required items in every card | PASS |
| Outerwear rotation | PASS: zero consecutive clothing overlap |
| Top rotation | PASS |
| Bottom rotation | PASS |
| Existing quality preserved | PASS: same complete gates; only the shoe false-negative distinction changed |
| Mathematical shoe-overlap minimum | PASS: 0, 0, 0, 0 |
| Shoe LRU / frequency fairness | PASS in all five real decks and forced-reuse fixtures |
| Duplicate physical-product rows | PASS |
| History/scope/request-ID semantics unchanged | PASS |
| Regression tests | 132 suites / 1,642 tests PASS |
| App TypeScript and Edge Deno checks | PASS |
| Deployed/check-out source comparison | 16 / 16 identical files |

**Final shoe rotation: PASS.**

## Limits and scope

This report certifies the shoe-rotation task for the measured catalog/request and the stated regression cases, not a new general outfit-quality approval. Existing unrelated quality work remains preserved. No shirt/jacket image-classification expansion, schema migration, provider change, history reset, request-ID change or budget/weather relaxation was introduced. An earlier automated run toggled Date Night off and was excluded; the accepted run explicitly asserts Date Night. Supabase logs are sampled, so the available stage example is not claimed to be a complete five-response server log dump.
