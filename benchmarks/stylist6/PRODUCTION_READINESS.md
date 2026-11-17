# Stylist 6 development readiness

## Verified baseline

The development checkout was fast-forwarded from `4b10e2b` to `4a71ed1`. GitHub reported success for all six development commits after `8ca1198`. The baseline blind-review tests, Stylist 6 tests and TypeScript check ran before the changes. Production v207, staging v4, the compute branch and main are outside this work.

## Whole-outfit evidence

Earlier Q01 receipts had two false missing-outerwear claims with four supplied slots, a generic contract failure, and a 0.0005 selection margin. The scored difference did not establish which outfit was better. The earlier visibility flag asserted presence without asking the model to demonstrate recognition. Evidence could also discuss garments individually while assigning a high outfit score.

The critic now asks for a concrete visible description of each supplied slot before scoring, two relationships between candidate slots, and one observation relating the candidate to the reference. An unreadable or explicitly missing image, a single-slot outfit relationship, repeated evidence, malformed JSON and invalid numeric fields abstain. The adapter still emits the existing selector contract. NVIDIA JSON-object mode is enabled. Provenance retains the slot descriptions and relationships for adjudication. These checks establish coverage and consistency, with human or independently verified visual labels still required for accuracy. No garment-specific fashion rejection rule was added.

### Live CC0 checks

These were separately claimed, zero-retry internal tests against the existing NVIDIA free endpoint and the same Nemotron Super VL model. They used the two problematic Q01 CC0 combinations from The Met, their original CC0 reference collage, and reordered versions of those same images. No original AIWardrobe reference images, retailer photographs or customer information were submitted. Receipts remain private.

| Run | Calls | HTTP 200 | Valid | Abstentions | Rejected | Selected | Wall time | Reported tokens |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Structured evidence, text JSON instruction | 4 | 3 | 1 | 2 malformed JSON, 1 deadline | 0 | 0 | 12.006 s | 8,695 |
| Structured evidence, JSON-object mode, two waves | 4 | 2 | 2 | 2 deadlines | 2 | 0 | 12.004 s | 5,757 |
| Reversed slots, JSON-object mode, one wave | 2 | 2 | 2 | 0 | 2 | 0 | 7.226 s | 5,796 |

All four completed JSON-mode responses passed structural validation, but two of six JSON-mode calls timed out. The failed second wave cannot isolate an order effect. The single-wave check returned evidence about all four slots on both candidates. One canonical response claimed embroidery was absent from the reference although the candidate and reference contain the same exact coat photo. Score and semantic fidelity stability remain open. These selected challenge cases are unsuitable for an acceptance-rate or latency SLA. No independent fashion labels were available. False approvals and false rejections are unmeasured.

The critic remains capped at four candidates, two in flight, a shared 12-second deadline and zero retries. HTTP success and a complete-looking response do not guarantee a valid judgment. Calls that hit the deadline may still have used provider quota, so reported tokens exclude unknown timeout usage. Use the existing small licensed protocol to measure recognition and score stability once independent labels are available. [The Met Open Access](https://www.metmuseum.org/hubs/open-access) · [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf)

## Authenticated integration seam

`backend/stylist6/development_shadow.mjs` consumes the original validated look plan and the baseline deck after authentication, subscription/quota, personalization, catalog retrieval and hard feasibility. The adapter accepts only explicitly marked synthetic private trials, with an upstream atomic run claim. It verifies the exact SKU and catalog image, active per-asset rights, and fresh size, stock, destination and landed-price evidence. Changed catalog or offer prices make a candidate ineligible for that snapshot. If it can produce a complete three-look visual deck, it returns the original look objects. The existing explanation path can format those objects. Withheld results leave the separately identified baseline untouched.

No API route imports the bridge. There is no deployed feature flag, customer exposure, new quota or entitlement system, or database migration. A future isolated integration should invoke it after the original selector returns, reuse caller admission and its durable replay claim, and expose the result only to authorized internal evaluation. Tests exercise the bridge with synthetic grants, offers and transport, including duplicate claims and stale price or mismatched photo rejection.

## Retail evidence

A fresh read-only catalog query found 7,912 rows, 7,020 active, 12 active rows with an external ID, 4,466 active without an import timestamp, and none seen in the preceding 24 hours. All active rows had an HTTPS listing URL. All 59 fixture rows matched active live rows with no price, currency or URL drift. None of the 59 had an external ID. HTTPS format is separate from link validity and stock. The prior direct listing audit recorded 46 HTTP 200s, eight 403s and five 429s, with no verified exact stock offers.

The existing feed importer uses a supplied SKU or derives one from the source and merchant ID, with name and photo URL as its fallback. It upserts by catalog ID and updates price and last seen time for rows received. It does not deactivate absent records. The Zara sync has explicit stale and duplicate deactivation logic. The live schema has no `shop_offers` or `shop_merchants` table and no size, destination shipping or image-rights fields on `shop_catalog`. The dev review still has zero verified purchase looks, no authorized offer rows and zero cleared retailer/reference image assets.

Start with one willing merchant, one destination and its authoritative variant/stock/price feed. The written asset schedule must grant display and reviewer distribution separately from external NVIDIA processing, and must address retention, geographic scope and provider use under the applicable terms. Catalog URLs and affiliate promotion alone do not establish those rights. The exact required fields and manifest examples are in [Retail validation](RETAIL_VALIDATION.md). [Shopify ProductVariant](https://shopify.dev/docs/api/storefront/latest/objects/productvariant) · [Awin image permissions](https://success.awin.com/s/article/Can-I-use-any-images-from-the-advertiser-s-website?language=en_US)

## Blinded evaluation

The existing 12-case inventory/budget/occasion fixture reproduces 72 looks. Both arms use the same frozen catalog and shared feasibility constraints. Neither arm has a real destination and size eligibility snapshot. Stylist 5 deterministic and its earlier 12/12 hosted text ranking results are both retained. The live rankings passed through the pinned selector and did not change the 12 final decks. The ranked replay omitted real user state and cached visual facets. Stylist 6 retail remains deterministic retrieval, preserving 9.25 unique SKUs per three-look deck compared with 7.92 before diversification. Retail visual selection and preferences are unmeasured.

The review tool supports selecting deterministic or hosted arms separately and refuses a hosted Stylist 6 package with incomplete judged decks. It adds occasion suitability to attractiveness, compatibility, reference fidelity, variety, purchase plausibility and serious-error flags. A private persistent random seed counterbalances sides 6/6 and keeps assignments unrecoverable from the package identifier. The scorer requires three complete distinct reviewer submissions, reports disagreement separately from ties and splits, and computes the exact sign test on decisive cases plus paired scenario bootstrap intervals.

A 72-look owner-local preview was produced from byte-verified media. The distributable archive was intentionally withheld by the new grant check. Distributable packages require a unique current exact-asset grant whose `privateReview` is `{"redistribute":true,"audience":"assigned_independent_reviewers"}` plus the usual grant evidence. The owner-only mode does not export ratings or generate an archive. There are no independent submissions or paired quality outcomes.

## Next milestone

Run a one-merchant, one-market cleared retail shadow evaluation. Acceptance requires an authoritative current offer for every exact selected variant, same SKU/price/budget and requested size/delivery compatibility, and a documented grant for every submitted and reviewer-visible asset. Both arms use the same eligible pool. Stylist 6 must produce three distinct valid looks for each feasible case without catalog substitutions. On the 12-case pilot, require at least 90% valid visual verdict coverage with no more than four requests per case, record deadlines and reported tokens, and require at least eight majority preference wins from three independent reviewers. Treat ties, splits and abstentions as non-wins. A release needs a larger held-out production-like trial and explicit owner authorization.
