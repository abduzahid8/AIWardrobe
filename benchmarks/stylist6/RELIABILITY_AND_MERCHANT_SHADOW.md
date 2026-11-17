# Reliability and first merchant shadow

## Deadline diagnosis and policy

The baseline first wave finished in 6.940 s and 7.061 s. The shared 12-second limit left the two second-wave requests approximately 5.06 s and 4.94 s after dispatch. A separate single-wave request took 7.225 s. The old traces did not isolate media, provider response, and parsing, but the remaining budget was demonstrably smaller than that observed turnaround.

The revised critic preserves a maximum of four calls, two concurrent workers, cancellation and zero retries. An attempt receives eight seconds from dispatch, rounded from the 7.225-second observation, bounded by an 18-second batch limit. The batch cap is two eight-second windows plus two seconds of local headroom. The default internal latency ceiling increases by six seconds, with no increase in concurrency or call count. Queue, media, dispatch-to-body, headers, parse and validation timings are recorded independently. Timeouts distinguish queued, media, in-flight batch and individual-provider expiry. Payment required, rate limits and access denial stop subsequent dispatches. Cancellation does not resubmit an ambiguous call. Client-side dispatch-to-body timing is not GPU-only inference time.

The deterministic simulator uses explicit stage times. Four seven-second provider responses with 20 ms media each finish 1/4 sequentially under 12 s, 2/4 with two workers under 12 s, and 4/4 under the new limits in 14.04 s. With one 12-second provider response and three six-second responses, the revised policy isolates the slow attempt and completes the other three in 14.04 s. These are simulated timings.

```sh
node scripts/stylist6/visual_schedule_simulation.mjs
```

## Controlled CC0 calibration

The private plan hash-pins the exact Met source photos, the reference collage, candidate slot order and critic code. Eight cases cover exact same-coat and same-look identity, reversed order, contrasting silhouette and trim, formality, layered and unlayered outfits, ambiguous material and a footwear-only change. The live paired subset used four cases. Remaining cases were media-validated and are queued for a future labeled study. No retailer or personal photos were sent. No human styling labels were simulated.

The two paired arms used the same model, files, image order, JSON mode, temperature and deadlines. Only the grounded arm received verified exact source-photo matches from the trusted reference composition manifest. Those facts are derived from verified hashes, with only slot names sent to the model. Reference evidence cannot assign a difference to a matched slot. The absence of a reference-only garment is represented separately and must be addressed. Without trusted composition provenance, no identity or absence fact is inferred.

| Run | Calls | HTTP 200 | Valid | Abstentions | Wall | Reported tokens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| v7 control | 4 | 4 | 3 | 1 incomplete outfit evidence | 11.522 s | 10,908 |
| v7 exact-photo grounding | 4 | 4 | 3 | 1 incomplete outfit evidence | 9.720 s | 11,095 |
| v8 reference-only layer follow-up | 2 | 0 | 0 | 2 individual-provider deadlines | 8.031 s | Unreported |

The v7 second wave queued for 4.06–6.14 s and all four calls completed. Both v8 timeouts queued under 1 ms and did not return headers, so neither came from second-wave starvation. Across eight completed responses, client-side provider p50 was 5.38 s and nearest-rank p95 was 6.12 s. This is a descriptive sample, not an SLA, and it excludes the censored calls. Total reported usage was 22,003 tokens, with unknown usage for the two timeouts. There were zero retries.

Exact-source annotations are independent of the model. The grounded canonical same-coat verdict correctly excluded the known identical coat and trousers from its difference statement. The new control also recognized the same coat in this run, so the matched pair does not show a measured accuracy win for that specific judgment. Both reversed-order responses failed the evidence contract. An analyst spot-check of the pixels found a false high-stiletto claim in the control and a false block-heel claim in grounded output for the same short curved heel. Grounded output also described both outer layers as dark even though one jacket was predominantly cream. Across six admitted v7 verdicts, the 22 broad garment-role descriptions aligned with the supplied photos, while these feature errors remained. Analyst spot-checks are not independent human styling ratings.

The v7 three-item verdict returned 0.92 reference fidelity while overlooking the reference's fourth garment. The v8 validator now distinguishes reference-only outerwear from a missing input photo and rejects evidence that omits it. Deterministic tests verified that behavior. Both live follow-up attempts timed out before returning evidence. Accuracy and score stability on that revision remain unmeasured. Fabric hallucination rates and false approval/rejection rates require adjudicated labels. Do not infer them from model confidence or the challenge-case rejections.

Run a dry check first. Live replay requires the exact plan SHA, a private free-tier credential and the explicit flag. The runner uses an exclusive local lock and one claim per plan and arm, with a persistent stop on payment, access or rate limits. It only loads verified local CC0 assets.

```sh
node scripts/stylist6/cc0_grounding_trial.mjs --plan /private/grounding-plan.json --arm grounded
node scripts/stylist6/reliability_report.mjs /private/q01-cc0 /private/adjudicated-observations.json
```

## Merchant contract and private shadow

The read-only catalog query found 987 active UZS listings under the Zara Uzbekistan URL across two source groups, with upper-body, lower-body and shoes represented. None of the 987 had an external ID or an import observation in the preceding 24 hours. This is a candidate catalog slice. No authorized variant feed, size-specific stock, checkout shipping evidence or retailer image grant was available for it. No outreach, scraping around access controls or merchant agreement was assumed.

`merchant_snapshot.mjs` normalizes an authorized single-merchant, single-country, single-currency export after the caller verifies its origin. The export bytes are pinned by SHA-256 and evidence reference. Supply an independently verified binding from exact catalog SKU and requested size/size system to merchant variant ID. The export provides listing URL, per-variant in-stock state, observed timestamp, confirmed destination, currency, item price, shipping and landed amount in integer minor units. Its photo row supplies exact source URL and derivative hash. The reviewed grant registry supplies holder, agreement reference and separate scopes for NVIDIA processing and reviewer distribution. The adapter rejects duplicate bindings and offers, mismatched variants, stale observations, unconfirmed delivery and pricing drift. A feed hash verifies integrity, while merchant credentials and a reviewed agreement establish authority.

The shadow reuses the existing plan and baseline unchanged. For real transport it requires an authorized-merchant snapshot. Every judged item must match the snapshot's exact SKU, variant, current offer, image URL and photo bytes, with current rights. A synthetic scope can be used with mock transport only. The test for one synthetic UZ destination supplied 16 exact variants, excluded one out-of-stock SKU, and returned three distinct looks with 12 exact SKU identities. A provider timeout in a separate synthetic run returned a partial candidate result and withheld the whole visual deck while keeping the baseline. No customer route imports the adapter.

```sh
node --test backend/stylist6/development_shadow.test.mjs
```

## Quality and release gates

The existing 72-look comparison remains a 12-scenario frozen catalog study. Stylist 5 deterministic and its 12 prior successful hosted ranks yielded identical final decks in that fixture. Stylist 6 retrieval retained 9.25 unique SKUs per three looks. The retailer images are uncleared for hosted visual selection and reviewer distribution, and the real offer audit remains at zero verified outfits. There are zero independent reviewers or preference results. No new retail review archive was produced.

Next milestone is one authorized merchant, one UZ destination, and one private shadow study. Require current, exact size/variant stock and destination prices for every eligible product, both permissions on exact photo bytes, no SKU substitutions or budget violations, and at least 90% valid verdict coverage with four requests maximum. Independently adjudicate a locked visual-fact set and compare three complete looks per feasible case against the unchanged Stylist 5 baseline. Eight of twelve majority wins is an exploratory target. A later release requires a powered held-out comparison with uncertainty, an acceptable measured latency budget, and explicit authorization.
