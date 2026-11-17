# Stylist 6 outfit ranking replay

## What was weak

The previous reference-first retrieval treated subtype and membership in a reference palette as the main scoring signal. It ranked items before judging their relationships, cut the candidate list before checking budget, and returned the first three round-robin brief picks. A correct product slot and an allowed color could therefore yield a padded parka for a 23°C office request. Three looks drawn from different briefs could still repeat the same products.

The Stylist 5 kernel already contains useful whole-outfit, climate and dress-code logic. The shared context interpreted an unqualified wedding as casual, however, and the benchmark's first Stylist 5 wedding look had denim outerwear and waterproof boots. The new Stylist 6 entry defaults an unqualified wedding to formal while respecting an explicit dress code or beach/casual cue. The deployed Stylist 5 context is unchanged.

## Fashion engine

`backend/stylist6/fashion_deck.mjs` uses the existing shortlist generator and the existing domain styling context, garment descriptors, visual profile validation, climate and occasion checks, layer compatibility, color harmony, whole-outfit coherence, styling distance and grounded explanations.

- Prepare current cached observations once and keep text-derived descriptions when no observation qualifies. A stale profile cannot supply a color or veto an outfit.
- Retrieve alternatives from existing native styling concepts and annotated lookbook briefs. Preserve source, explicit exclusions, exact locked products and budget. Apply the budget before the shortlist cap so expensive candidates do not crowd out eligible combinations.
- Rank each feasible complete outfit across palette, layer, formality, proportion, texture, footwear and occasion. Penalize the weakest composition dimension and retain the competing-pattern penalty already emitted by the shared judge. Use reference role, palette, slot-color and layer metadata when present. The ranker makes no claim of comparing reference photos.
- Select up to three looks jointly from the highest-ranked candidates. Require at least two physical products to differ, penalize repeated products and reward known styling differences. Preserve a partial or empty result when the bounded search cannot supply three feasible looks.
- Use the existing whole-outfit critic only for a close first-pick decision when the caller has cleared both candidates and the same reference. At most two judgments are requested. Missing permission, absent provider and abstention leave a catalog-ranked path. Valid confident rejection removes a candidate from consideration. Reviewed evidence is attached only to a judged pick.

This is an experimental development entry point. It has no new customer route or deployment. The existing strict visual-only pilot remains available with its original fail-closed behavior.

## Measured replay

A read-only catalog query found 7,020 garment intelligence rows, 5,467 current active visual profiles and 5,437 cached embeddings. The frozen UZS sample has 384 active genuine SKUs, 96 per garment slot and 372 with current visual profiles. The query prioritizes a low-priced row per subtype and cached color, then samples deterministically by ID hash. It joins the visual profile by SKU, exact image URL and current catalog source hash. No embeddings were compared because the reference brief collection has no compatible reference vectors. Visual descriptors provide the existing cached evidence used in this run.

The input SHA-256 is `0d29630457e80f0b1f4480a89771d9e4b6093acdbbe6bf7dafac49ac13ac31df`. The frozen input, full JSON output and HTML gallery are private evaluation artifacts. No images or provider requests were made by the replay.

The 39 cases extend the previous 35-case fixture with streetwear, formal dinner and a no-layer date. They cover 9–29°C, explicit color and pattern exclusions, fit preferences, varied occasions, native budgets and unbudgeted cases. Nine have a matching manually annotated reference brief. Both engines receive the same catalog, user inputs and hard evaluation criteria. The Stylist 5 arm uses its pinned v207 deterministic planner and native gates with current profile inputs. Its separate historical 12-case hosted replay is not a hosted result for these 39 cases. The previous Stylist 6 arm uses its retrieval shortlist on the same feasibility gates.

| Arm | Complete decks | Looks | Coherent by diagnostic | Mean diagnostic coherence | Mean unique SKUs in complete deck | Styling distance | Hard violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stylist 5 deterministic | 37/39 | 111 | 108/111 | 0.798 | 10.78 | 0.650 | 0 |
| Previous Stylist 6 retrieval | 35/39 | 107 | 99/107 | 0.676 | 10.66 | 0.599 | 0 |
| New Stylist 6 outfit ranker | 37/39 | 111 | 111/111 | 0.839 | 11.11 | 0.656 | 0 |

The ranker improved the automated score in all 36 paired cases against previous retrieval and in 33 of 37 against Stylist 5. Four Stylist 5 decks scored higher. Reference palette fit across the nine eligible cases was 0.562 for the new ranker, 0.354 for previous retrieval and 0.281 for Stylist 5. With the wedding case removed, mean coherence is 0.837 for the new ranker, 0.695 for previous retrieval and 0.820 for Stylist 5. Two tight-budget cases produced no look in any arm. The final local median/p95 for Stylist 6 ranking was 122/178 ms. All 39 selected decks and metrics reproduced across two runs. The previous 12-case retail review replay also reproduced exactly.

These are deterministic diagnostic measures using some of the same signals the ranker optimizes. There are zero independent fashion votes, so they cannot establish attractiveness, visual reference fidelity or superiority. The fixture measures frozen catalog eligibility. It does not verify merchant stock, delivery or shopper fit. One warm date-night deck still includes catalog-labeled waterproof trainers in its third look, a useful case for manual styling review.

## Example looks

- **Business-casual office at 23°C.** Previous Stylist 6 chose an anthracite merino turtleneck, reversible padded parka, blue wool trousers and black dress loafers. New Stylist 6 chose a regular-fit long-sleeve shirt, double-breasted blazer, suit trousers and cognac leather Oxfords. Total snapshot cost changed from 25,682,243 to 6,169,867 UZS. The replay was unbudgeted.
- **Wedding.** Stylist 5's first look paired a knit polo, denim jacket, chinos and waterproof ankle boots. New Stylist 6 chose a poplin shirt, double-breasted blazer, brown wool flannel trousers and black leather Derbies. The gallery lists the exact SKUs and prices for both decks.

## Run and inspect

```sh
node scripts/stylist6/fashion_replay.mjs /private/catalog-uz.json /private/fashion-replay.json
python3 scripts/stylist6/fashion_gallery.py /private/fashion-replay.json /private/outfit-gallery.html
```

Open the gallery locally. It renders outfit names, exact SKU identities, snapshot prices, and the three comparison arms. Merchant image URLs only load in an opened case after the viewer opts in for authorized private display. The benchmark does not fetch or send these photos. The gallery is not a redistributable blind-review archive.
