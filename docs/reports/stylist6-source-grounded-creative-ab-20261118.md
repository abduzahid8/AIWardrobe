# AIWardrobe Stylist 6 — creative concepts mapped to real merchandise: live native quality experiment

**18 November 2026. Objective: three outstanding, genuinely distinct, source-authentic, purchasable four-product looks.** This is a report about real authenticated iOS generations and actual photo-critic approvals, **not** acceptance that an excellent scalable personal stylist exists. Production Stylist 5 has not changed.

## Structural defect, not more little outfit rules

A professional artistic concept was created independently of the actual SKU composer. The first artist could propose a 3-item warm-weather direction despite the 72s catalog path's strict **four-real-SKU** contract (top + meaningful layer + bottom + shoes); the composer then had to invent the layer relationship itself. Even a valid creative plan was passed only as subtype text to a single 48-token menu, so a carefully planned palette, material, silhouette and shoe relationship often dissolved into generic shirt + blazer + trousers.

The experiment `0157f5e` introduced `sourceGroundedCreativeCommissions`: for each of three first-pass AI concepts, enumerate **actual available opaque merchandise tokens** per garment role and convey the original artistic concept, intended color and proportions. The consumer AI decides the precise whole four-item styling (the options are NOT restrictions; it can choose any other source-eligible product), then a second independent model can attempt alternative concepts using genuinely UNUSED SKUs. Additionally, the first AI plans a meaningful fourth layer when the catalog actually requires four items, instead of silently planning three and relying on later padding. No garment subtype/brand/color combinations were prohibited, no user feedback was converted into arbitrary hard bans, and independent source, budget, rights and original-photo critic gates were unchanged.

This is **not** original-photo-first creative SKU generation: the source-licensed original JPEG cache has zero READY rows, so the first designer still uses genuine verified photo-derived attributes and tentative source-text clues. Its model cannot claim to see unseen original image pixels. The photo critic and final portfolio jury do inspect actual permitted product image bytes.

## Staging-only repeatable source cohort

To stop mistaking changing catalog discovery for fashion improvements, `159a322` added server-owned `STYLIST6_STABLE_COHORT_ENABLED`, active only on isolated >=72-second catalog staging with no locked items. The source sampler's seed includes user identity, style, occasion, weather and budget but excludes ephemeral generation request IDs and earlier UI replay exclusions. **Authentication, billing, exclusion filtering, source validation, photograph veto, full-model editorial review and production paths remain unchanged.** `5d26fbd` forwarded the server-only flag through the signed Render adapter. Live receipts confirmed `stableSourceCohort:true`.

**Precision limitation:** this is a *repeatable sampling seed and the same query context*, not a physically immutable snapshot of every merchant database row. Inventory may still change between requests. The model is also nondeterministic. One-shot A/B differences are evidence for directing the next experiments, NOT a statistically proven causal uplift.

## Real mobile A/B, Classic / New / Date night

Authenticated native iOS 26.5 simulator `com.aiwardrobe`, saved settings and one Generate action each time, Render staging, same source-sampling-seed inputs and explicit `stableSourceCohort:true` in independent server receipts:

| Condition | Source creative plans | Real composed SKU proposals | Photo approvals / attempted | Source-real 4-item decks displayed | Wall compute | Final photograph jury |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A — new source-grounded creative commissions (`5d26fbd`, `0157f5e` active) | 3 | 5 proposed, 4 admitted | **4 / 7** | **3** | **52.3 sec** | real-photo primary applied, independent photo jury `standalone`, plus independent Kimi-approved third replacement |
| B — prior Creative Director (`eaa90d03` reverts `0157f5e`, same stable seed) | 2 (`planned_partial`) | 8 proposals | **5 / 11** | **2** | **62.5 sec** | real-photo primary applied and independent arbiter preferred primary two-look collection |

A candidate trio exists in both real source contexts, but the final editor does not necessarily include a weak third even if its individual critic score passed. A did produce an admitted photo-reviewed third without lowering the critic gate. A's observed merchandise composition used classic knit layering, a softer cardigan/chino direction, and dark blazer/trouser tailoring. B's two cards included a black knit/light tailored layer with dark trousers, and a more conventional blazer/tailored-trouser pair. A is *more complete* in this one comparison but **not indisputably more beautiful**; significant generic darkness and tailoring remain. These screenshots contain copyrighted real merchant thumbnails and were inspected locally, not committed or rehosted.

**Before the source seed was correctly wired**, the new design had also yielded 1/9 approvals and only 1 displayed Date-night card on one stochastic live request, and 3 displayed on another. Do not suppress that contrary evidence or claim that the source creativity change guarantees three looks.

## Same-source repeated A2 generation — independent real photo outcome

After restoring the exploratory source-grounded creative architecture in code SHA **`4b2442e88111bcc5bd5b518677d48f96bf637867`** and deploying isolated Render staging **`dep-dbuma9mk1f9s73an78ag` (LIVE)**, a second separately debited authenticated native `Classic / New / Date night` run used the same production-style code path, same style/weather/budget/user sampling cohort (`stableSourceCohort:true`). This was a **new** actual model inference, not a cached response or playback of A1.

- **3 final 4-product looks**, `catalogFourItemContract:true`, `catalogFourItemDropped:0`.
- **5 photo approvals / 7 photographed attempts**; 8 exact source-SKU outfits initially proposed by Nemotron + independent Kimi; `creativePlanCount:3`, 29 source-current photographic feature examples supplied to initial creative planner; `photoEditorialChosen:3`, original photo board `ready` and visual jury `applied`. Other independent jury `unavailable`, so **do not claim two-fashion-lab photo agreement** on this run. No post-editorial creative revision needed.
- **48,574 ms** engine and **48,805 ms** delivered wall time. The real live signed service returned a completed paid deck. iOS showed 01/02/03 tabs and three real source-pictured cards.
- Native Maestro output had an intermittent **`No space left on device` JVM file-log exception** during its saving/cleanup stage (Mac had ~218 MiB free); despite `job_result` reporting exit code 0, **Maestro's whole-run success is not asserted**. Exact server real-photo quality receipt confirms generation, and resulting iOS screenshot was recovered and inspected. Only our own regenerable /private/tmp QA log directories were deleted, leaving comparison screenshots and Git code intact.
- **Visual human inspection remains a FAIL for the owner's aesthetic bar:** A2 01 `Smart knit layering` uses nearly all-dark knit/jacket/trousers; 02 `Blazer & Chinos` has a navy-toned jacket, dark trousers and dark loafers; 03 `Polo & Chinos` is another mostly black knitted top, dark pants and dark loafers. The names signal variety more than the actual photographs do. Some top/layer thumbnails were not yet rendered at screenshot capture, so no unobserved item material/color claims are made. **Do not conflate AI/photo-veto passes, three tabbed looks, or repeating a count with fashionable, beautiful, diverse styling.**

**A1 and A2 both returned three 4-item cards on the same server-owned stable catalog sampling seed**, while the one baseline B returned two. This is a promising delivery/reliability sign but insufficient A/B repetitions, immutable source-image proof, expert human scores and held-out situations to make statistical or acceptance claims. AI nondeterminism remains. The root aesthetic issue is repeated dark tailoring/formality and weak silhouette invention, not missing UI cards.

## Verification and decision

- All core provenance and identity contracts remain as before. Source token creativity is not an AI photo quality pass.
- `546 / 546` complete Stylist 6 tests and `42 / 42` Node server tests pass on the restored creative design + stable-cohort code. The unchanged production Stylist 5 authorization and quota implementation are bundled separately.
- **Human visual acceptance has not been met.** Three cards must be three genuinely stylish, distinct design decisions; a model-generated editorial title or AI photo reviewer agreement is insufficient.
- Next milestone: controlled multiple-run same-catalog source-photo fashion preference comparisons and rights-cleared full visual-first SKU creative decision, followed by >=30 held-out contexts with three independent blind human reviewers and >=70% Stylist 6 preference over Stylist 5. Do not revive subtype exclusions (0/7 real photographic counterexample) or artificially fill a third card.
