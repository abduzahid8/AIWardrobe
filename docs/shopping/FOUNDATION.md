# Shopping implementation map

Local implementation based on `5784fc5` on `stylist-four-slot-recovery`.
No hosted state has been mutated by this task.

## Identity and family presentation

`catalogIdentity.ts` distinguishes listing, physical product, product family,
variant and visual family. It exports Unicode-safe brand/name/base-name
normalization and color evidence. Physical identity delegates to the unchanged
`outfitDiversity.productIdentityOf`; shopping grouping never replaces it.

`catalogFamilyView.ts` builds indexed virtual families over loaded rows. Trusted
source group/model references or canonical product URLs provide high confidence.
A merchant-scoped Jack & Jones adapter uses its official article-number URL
component; it does not infer model relationships from arbitrary external SKU
suffixes. Known color query parameters also preserve J.Crew variant URLs while
sharing their canonical product key.
Distinctive exact titles plus compatible colors/images/currency/prices can
provide medium confidence. Generic merchandising titles stay separate. Every
variant retains its original object, ID, price, currency, image and destination.
Grouping is O(N log N) including stable output ordering, with conservation and
confidence diagnostics. No schema or destructive deduplication is needed.

Inspo groups after existing search/saved filtering. Home Essentials selects
family representatives through the extracted pure `homeShoppingSelection.ts`.
Both expose the shared accessible, virtualized `CatalogVariantPicker`; unknown
colors use thumbnails and numbered English/Russian/Uzbek labels. Its buttons are
outside merchant buy targets. Explicit selection chooses the original row.
Inspo saves that row, not its presentation wrapper. Home adds that concrete
variant to the closet. Neither family objects nor image blobs are persisted.

`shopCatalogStore`, Profile, OutfitShopSheet and generation contracts continue
consuming concrete rows. Family keys never appear as generation request IDs.

## Bounded discovery

`catalogDiscovery.ts` supplies shared requests, deadlines, cancellation per
reader, a five-minute cache and bounded cached query/count entries. It prefers
`catalog_discovery_candidates` when available. Missing-function capability is
cached briefly; permission/auth errors are not misclassified as missing RPCs.

Before migration, exact head counts and seeded circular ID windows reach
interior/later rows. The count is never guessed. A request returns at most its
limit (100 max); wraparound uses at most two bounded row reads. Pagination pins
its seed and traverses a stable inventory without overlap. Raw counts can include
rows later rejected for provenance, so a visible page can be short. Global
window fallback does not guarantee brand/slot balance; the prepared RPC supplies
those guarantees, while slot-specific callers retain separate slot queries. Counts and
rows are live read evidence, not guaranteed snapshot-isolated pagination while
ingestion changes the table.

The additive invoker RPC filters active/product-only rows before ranking the
eligible relation on the server. It retains native-currency price floors, then
spreads model buckets, brands, slots and shoe archetypes with seeded tie-breaks.
Its model bucket is only a retrieval diversity relation, never a family merge
or physical-identity claim. Output is bounded to 100 concrete rows.

Consumers:

- Home: discovery through `useShopCatalog`, with the original 100-row payload.
- Inspo: category/search discovery with normal incremental page append.
- AIOutfitmaker: discovery for initial inventory and 30 rows per slot per seeded
  request. The old guessed total and global random offset are removed. Refined
  requests retain a bounded 80-row curated supplement alongside broad discovery.
- Stylist evidence: four discovery queries, at most 100 rows per slot; existing
  verified-empty semantics, cache sharing, refresh and cancellation are retained.
- Shopping completion: at most 30 rows per slot; name-derived slot validity,
  product-only provenance, style ranking, native floor representatives and
  shopping diversity precede the existing final outfit gates.
- Generic `searchProducts`: real repository results, bounded search, lazy bundled
  fallback, original destinations and recent-search/wishlist behavior. Optional
  strict brand/color/min-price queries retain direct read filters. Color terms
  in styling descriptions cannot relabel the product's title color.

No full catalog is downloaded or sent to the model. Existing Edge generation
retrieval and final deterministic validation remain unchanged.

## Shopping intelligence

`shoppingIntelligence.ts` ranks a bounded result set with explicit eligibility,
soft recency and family/role/brand/color redundancy penalties. Diversity can
select only within 0.15 of the best remaining relevance score. Mandatory anchors
and budget representatives survive trimming; an impossible mandatory bound is
reported instead of silently dropping them. Thin pools remain reusable.

`shoppingEvaluation.ts` ranks alternatives for overall, cheaper, closest style,
wardrobe gain and predicted fit. Missing evidence stays missing. Cheaper requires
observed comparable native currencies; wardrobe improvement requires positive
validated sampled gains; fit requires measured evidence and confidence.

Useful outfit evaluation builds bounded complete looks using the candidate and
owned slot partners, then calls deterministic validation. It removes equivalent
owned substitutions and near-identical role/color/material/pattern/silhouette
looks. It reports evaluation caps, truncation and low sampling confidence.
Low/Medium/High wardrobe value is relative usefulness, not financial ROI or an
exhaustive count of possible outfits.

`shoppingAdvisor.ts` connects these primitives to the existing occasion,
activity, weather/request-intent, budget/FX, whole-outfit and fit engines.
Owned garments cost zero; missing FX cannot pass a foreign-currency budget.
All observed title colors participate in avoidance checks. Only real supplied
physical profiles are used for fit; no seed size chart or invented body data is
substituted. Expensive wardrobe comparisons are restricted to 24 retrieved
candidates, with bounded per-candidate outfit evaluations. These APIs are exposed
through the existing shopping service; unsupported commercial claims are not
added to UI.

## Protected contracts and rollout

No changes to physical deck identity, rotation, source modes, anchor contracts,
auth, entitlement/rate limits, budget RPC/FX, fit engine, wardrobe contracts,
scan pipeline or offline synchronization. Legacy Express receives no new code.
Ingestion jobs were not run or changed by this task. Independently changing
catalog/expansion files in the worktree were preserved.

See `LOCAL_REPORT.md` for evidence and `SERVER_EVOLUTION.md` for exact remote
objects, activation, rollback and the later normalized model proposal.
