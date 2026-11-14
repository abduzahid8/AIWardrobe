# Stylist 6 isolated visual selection validation

## Execution audit

| Path | Call graph and evidence |
| --- | --- |
| Stylist 5 app request | `src/services/outfitGenerationService.ts` builds the personalized request, including source, exclusions, selected items, history, preferences and budget, then invokes `generate-outfits` with the user's bearer token. |
| Stylist 5 compute | `backend/stylist-proxy/index.ts` or `staging.ts` signs the exact request. `backend/stylist-node/api/generate-outfits.ts` verifies HMAC and invokes the original handler. `backend/stylist-node/build.mjs` bundles the pinned Edge entry. |
| Stylist 5 admission and retrieval | `backend/supabase/functions/generate-outfits/index.ts`, `generateOutfitsHandler`, admits and meters the request, resolves auth, budget/FX and preferences, reads wardrobe and history, applies cached garment intelligence and retrieves fresh catalog candidates by slot. It preserves anchors, source modes, exclusions and exact SKU IDs. |
| Stylist 5 construction and selection | `generateOutfitsHandler` passes its `evaluate` callback into `createOutfitPlan` after slot, source, price and compatibility checks. `_shared/nvidiaLookRanking.ts` ranks complete planned looks using text, concepts and cached image-derived facets. The payload contains no reference or product photos. The handler calls `plan.select(ranks)`, falls back to its feasible seed when needed and returns `groundedLookExplanation` from the final SKUs. |
| Previous Stylist 6 retrieval | `backend/stylist6/catalog_adapter.mjs` projects active rows into wearing roles. `lookbook_briefs.mjs` selects annotated reference briefs by style, occasion and season, or resolves a specific reference ID. `prototype.mjs` builds and gates complete looks, then seed-ranks a shortlist. `pilot_12.mjs` stopped at retrieval. The old `stylist6_nvidia_pilot.mjs` called the critic and wrote a receipt without selecting a deck. Its one-to-four reference input was also incompatible with the six-brief minimum. |
| Previous Stylist 6 HTTP diagnostic | `backend/stylist-node/api/stylist6-vision-test.ts` verified HMAC and ran one NVIDIA judgment on one submitted candidate. The successful Preview receipt used one CC0 catalog-like garment photo and a CC0 reference. It returned an incompatibility judgment to the operator. That verdict reached neither Stylist 5 nor Stylist 6 selection. |
| New Stylist 6 isolated flow | `proposeVisualFirstDeck` runs deterministic retrieval with mandatory trusted gate and FX callbacks, then calls `evaluateVisualShortlist`. The latter limits the input to four candidates and calls the hosted adapter. `selectVisualDeck` ranks valid, non-rejected verdicts using cohesion, separate reference fidelity, style, occasion and formality, then applies the existing diversity objective. It returns original SKU objects and bounded evidence. The pilot calls this same path on the frozen, validated pool and replays the verdicts for the separate Stylist 6 arm. |

The offline V5 control uses the deterministic planner from the pinned Stylist 5 Edge source with entry SHA-256 `f90ced33ca3b0034c61cd2dfa289b189728dc75398cd72329f0b7c1854efc5e1`. It omits live client state, live retrieval, cached visual facets and the hosted text ranker. Both arms receive the same catalog snapshot, occasion, brief and native-currency budget where specified. V5 receives the reference palette as a preference. V6 retrieves from the annotated reference and the hosted plan pins the same reference photo across candidates. There is no mobile or staging entry point for the new selector.

## Security and resource bounds

- The development branch returns HTTP 410 on the former vision diagnostic before reading a body or contacting the network. Tests submitted 100 signed replays across two independent handler instances with zero outbound attempts. Previously deployed Preview builds still have their prior behavior. Stylist 5 HMAC, authentication, quota, Node config and 1 MiB transport are outside this change.
- The new judging path resolves only local, trusted media keyed by catalog SKU and exact image URL. It verifies SHA-256, caps each JPEG at 512 KiB and 2,048 pixels per side, and sends at most five images per call. The fixed endpoint rejects redirects and receives a synthetic scenario brief, slot order and image bytes. It receives no original filenames, merchant URLs, user identity, JWT or wardrobe data. The separately invoked local merchant-image ingestion script now rejects non-HTTPS URLs, credentials, alternate ports and redirects and streams up to its size cap. It still relies on the explicit merchant host allowlist and local network trust.
- The pilot allows four provider attempts, two in flight, a 12-second batch deadline, 900 output tokens per attempt and a streamed 64 KiB response limit. It stops scheduling further POSTs after a 429. Two already-dispatched requests can complete. It never retries an ambiguous attempt. A per-plan exclusive file claim protects concurrent runs against replay on the same organizer filesystem. A distributed feature would require a shared atomic admission/spend ledger and verified egress and media policy.
- The old diagnostic had a 4 MiB application body cap and configured 60-second function duration, with per-operation timeouts that could accumulate. The new pilot runs locally, outside Vercel. Any future deployment must fit the 4.5 MB Vercel Function body limit and the configured execution budget. [Vercel Functions limits](https://vercel.com/docs/functions/limitations)

## Twelve-scenario evidence

The original private evaluation fixture contains 66 exact snapshot SKU records and cached merchant photos. A read-only live catalog query matched all 66 IDs. Seven were inactive and eight changed price. Currency and both URLs matched for all 66. The refreshed replay removed the seven inactive records and used current prices for the other 59. This is a targeted comparison against the review fixture, with one frozen catalog and one synthetic persona per scenario.

| Case | V6 exact-reference shortlist | V5 unique SKUs across three looks | V6 retrieval unique SKUs across three looks |
| --- | ---: | ---: | ---: |
| Q01 | 6 | 12 | 9 |
| Q02 | 8 | 12 | 8 |
| Q03 | 16 | 12 | 6 |
| Q04 | 16 | 12 | 9 |
| Q05 | 16 | 12 | 7 |
| Q06 | 7 | 12 | 6 |
| Q07 | 7 | 12 | 10 |
| Q08 | 6 | 12 | 8 |
| Q09 | 16 | 12 | 7 |
| Q10 | 16 | 12 | 10 |
| Q11 | 7 | 12 | 7 |
| Q12 | 4 | 12 | 8 |

All cases produced three complete outfits per arm. All 72 emitted looks passed frozen snapshot identity, four-slot and native budget checks. The first picks differed in all 12 cases and shared four SKUs across 48 first-pick slot comparisons. V6 retrieval uses fewer distinct products in every case. The fixture covers varied contexts and aesthetic challenges with four required slots throughout. It leaves unlayered requests, anchor/size edge cases and other demographics outside the comparison.

Merchant listing checks over the 59 refreshed rows returned 46 HTTP 200, eight HTTP 403 and five HTTP 429, with redirects disabled. The bounded page scan extracted zero exact matching stock offers. Merchant size, stock and shipping remain unverified, including for HTTP 200 pages. No outfit is certified purchasable by this benchmark.

The 12-case mock run exercised 48 whole-outfit requests through the real adapter and selector with an injected transport. The mock rejected the retrieval winner and gave an alternative higher fidelity. All 12 switched to the intended alternative and preserved fixture feasibility. The archive contains 24 anonymous top picks, with 108 image placements across 12 pages, and has zero independent votes. Ratings of style, coherence, reference fidelity and explanation quality are pending. The benchmark does not compare hosted explanation quality at this stage.

## Reproduction and provider authorization

Keep the organizer bundle and product/reference images outside Git. It contains the V5 offline replay script, scenario inventory, exact queue, product IDs and blind key. Generate the shortlist with its `replay.mjs`, then use:

```sh
node scripts/stylist6/benchmark_visual_selection.mjs /private/organizer
node scripts/stylist6/stylist6_nvidia_pilot.mjs --organizer /private/organizer --scenario Q01
```

The second command is a zero-call dry run. It writes a plan to stdout with the implementation, scenario, candidate, inventory and media digests. A live retail Q01 run requires confirmed image-use rights for the exact submitted bytes and enough free-tier quota for up to four attempts. The owner has given standing authorization for eligible free-tier tests. Pin the run with `--approve-plan <SHA-256>` and supply the existing server-side `NVIDIA_API_KEY` and the CLI environment gates. The receipt stays private. The experiment compares three V6 retrieval candidates and one V5 control, all with four products and the same 6.2M UZS cap. It tests hosted ranking behavior. Merchant stock and release quality are separate gates.

## Release gates

| Gate | Required | Observed |
| --- | --- | --- |
| Feasibility | Zero hard failures and zero unauthorized/replaced product IDs. Verify selected size, stock and shipping in the target market. | Zero frozen-fixture violations. Merchant purchasability remains open. |
| Evidence | No fabricated scores or fallback marked as visual. At least 90% valid verdict coverage on eligible photo-complete requests. | Mock coverage only. Invalid, missing, refused and timed-out verdicts abstain. |
| Latency and cost | Proposed isolated gates are added visual p50 ≤5 s, p95 ≤8 s, at most four attempts, and average billed cost ≤USD 0.05 per successful deck including failed attempts. | Existing single-call receipt reports 3.904 s handler and 5.302 s end-to-end. Whole-outfit hosted distribution and billing unknown. |
| Visual quality | Three independent reviewers. At least 8/12 majority pilot wins over V5 and positive lift over V6 retrieval on the fixed pool. Full-path held-out test with ≥30 feasible cases, ≥70% majority wins and two-sided 95% binomial lower bound above 50%. Count ties, abstentions and fallbacks as zero wins. | Zero independent ratings. The offline comparison cannot establish hosted improvement. |
| Security | Zero critical defects and outbound requests to unauthorized endpoints, and zero replayed dispatches. Verify shared admission before any hosted rollout. | Local mock checks pass. Historical Preview remains outside the patch. |

**Release decision — NO-GO.** Source is ready for isolated, authorized model evaluation. Production promotion requires live merchant feasibility, independent scores, whole-outfit latency and billing measurements, and a full-path comparison with Stylist 5's actual ranker.

## Free-tier follow-up

The owner's standing authorization covers bounded, internal free-tier inference. The service listing is a non-inference credential check and does not expose the account's credit balance. Use the trial endpoint only, verify asset-level rights, and stop further dispatch on a provider payment requirement, access denial or rate limit. Record available token counts and HTTP status without storing error bodies. Keep provider receipts and model output in the private evaluation workspace.

The original retailer/reference images lack verified permission for submission under the NVIDIA API Trial Terms. CC0 archive imagery can be used for a separate fixed-pool calibration. It provides neither merchant feasibility nor a head-to-head Stylist 5 comparison. [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf) · [The Met Open Access](https://www.metmuseum.org/hubs/open-access)

Each image now carries its slot label. A hosted verdict qualifies only if the model explicitly reports seeing each supplied slot. An invalid or incomplete report abstains. A visual quality difference of 0.015 or less keeps the earlier eligible seed in first place. Explicit rejection and minimum quality/confidence thresholds still apply.

The 12-scenario frozen retail replay measured 9.25 unique products per three Stylist 6 retrieval picks, up from 7.92. Diversity increased in eight scenarios and stayed level in four. All 12 first picks and Stylist 5 control decks were identical to the earlier replay. Each per-brief shortlist preserves its first pick, then prefers less overlapping candidates within one palette-match weight of the strongest remaining seed. This measures product coverage. Blind style and purchase results remain open.
