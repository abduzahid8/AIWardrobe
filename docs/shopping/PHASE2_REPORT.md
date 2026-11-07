# AIWardrobe Purchase Intelligence — local implementation report

2026-11-03 (Asia/Tashkent). Baseline: `5784fc5` on
`stylist-four-slot-recovery`, including the completed, uncommitted Phase 1 work.
This phase builds on that foundation rather than replacing it.

Final verification: see section 4. No remote mutation was performed by this task.

## 1. What was implemented

### Fit Intelligence — 2A through 2F

- Canonical FitEvidence, FitProfile, reference, size-recommendation and physical
  evidence models in `src/types/purchase/fit.ts`.
- Owner-tagged, ordered local persistence in `purchaseIntelligenceStore`.
  History is bounded to 160 observations; aggregates retain at most 80 priors and
  24 references. The key is registered for account cleanup. Stale hydration and
  another account's persisted records cannot become the current owner's state.
- Contextual feedback on garment detail: one-tap overall fit, optional zone,
  size/reference and preferred-fit details, Not now, Done, a 14-day dismissal
  cooldown and a settings toggle. Recorded garments are not repeatedly prompted;
  explicit updating remains available. Nothing is required to use the wardrobe.
- Positive, sufficiently confident brand/category/size feedback can become a
  reference passively or explicitly. Negative, returned, sold and body-derived
  observations do not become misleading positive references.
- Fit DNA separates brand, category, subtype, size, cut, materials/stretch and
  subjective fit. Reported snugness is not silently learned as preferred snugness.
  Conflicts, age, unknown cut/material and unsupported brands reduce confidence.
  Cached aggregates also lose confidence as they age. No universal user size is
  stored, and no measurements are invented.
- Size recommendations expose size, confidence, desired fit when known, zone
  notes, count, provenance and alternatives. Existing body/garment fit engine
  behavior is reused unchanged.
- Physical-data priority is exact chart → merchant variant → stored profile →
  exact reference relationship → category/cut prior → text → unknown. Priors and
  text never become synthetic centimeters. Explicitly verified size charts can
  convert named cm/inch measurements while preserving provenance.

### BuyCheck — 2G, 2H and UI integration

- `buyCheck.ts` returns BUY / CONSIDER / SKIP / INSUFFICIENT_EVIDENCE with explicit
  fit, wardrobe, purchase and evidence dimensions, not a single opaque score.
- Weak fit, sparse wardrobes, uncertain checkout costs, stale prices and unavailable
  selected sizes remain visible. Explicit budget conflicts are preserved.
  Shipping/tax/fee uncertainty cannot silently become a fully confirmed budget.
- Duplicate risk is role overlap, not family identity or deck physical equality.
  It considers role, color family, silhouette, observed material, inferred formality
  and style. It returns closest owned substitutes and explanations. A title alone
  does not prove the exact same purchase product. Intentional multiples are an
  explicit option, and even SKIP is advice: opening the merchant is not blocked.
- The new screen displays concrete variant selection, size evidence, owned
  substitutes, validated sampled looks/context breakdown, price uncertainty,
  original merchant links and per-offer links. A changed budget, size, context,
  fit history or wardrobe invalidates the displayed decision snapshot.

### Wardrobe Intelligence — 2I, 2J and 2U

- Useful unlocks v2 reuses the existing structural/style/occasion/activity/climate
  and budget gates. It excludes equivalent owned substitutions and trivial color
  changes unless contextual validity differs. A look valid in multiple contexts
  is not multiplied into multiple distinct outfits.
- Work is bounded: at most four supplied contexts, 64 base evaluations per context,
  bounded equivalent checks, six representative looks, and explicit sampling/cap
  indicators. These are validated samples, never Cartesian counts or exhaustive
  wardrobe totals.
- Maximum Wardrobe Gain evaluates at most 16 diversified eligible candidates,
  compares actual incremental gains, gaps, duplication, fit uncertainty and native
  budget, and abstains when no useful gain was validated.
- Missing-piece detection evaluates eligible slot counts across supplied contexts,
  bottleneck frequency and actual candidate gains, rather than recommending more
  tops merely because the catalog contains more tops.

### Money-Saving Shopping — 2K through 2M

- Same Look, Less Money identifies named style-critical versus basic pieces and
  changes one piece, or at most two when a requested target requires it. It respects
  pinned IDs, archetype, color family, formality, known material/cut, selected sizes
  and whole-look validation. Savings are explicitly item-price savings; unknown
  checkout costs are not assumed free. Inferred text/attribute similarity is not
  represented as an image comparison or guaranteed quality/fit.
- Upgrade This Look ranks supported single-piece changes by stated-style match,
  fit evidence and future validated looks. Higher price is not itself an upgrade;
  a cheaper improvement is tested. Additional spend for an owned piece uses zero
  current purchase cost rather than inventing its historical retail price.
- Find Better v2 has seven different policies: overall, cheaper, closest style,
  fit, wardrobe value, lowest duplicate risk and missing-piece value. Results
  expose concrete advantages and relative tradeoffs, including weaker fit,
  reduced wardrobe gain and higher item price. Unsupported improvements abstain.

### Price Intelligence — 2N through 2R

- Merchant offers, price observations, price assessment, True Price and watches
  are separate domain models. Product/variant IDs remain distinct from sellers.
- Price assessment supports 30/60/90-day windows, observed time-weighted median,
  observed minimum, discount evidence, freshness and sufficient distinct-day
  coverage. GOOD / NORMAL / HIGH / WAIT / INSUFFICIENT_HISTORY are deterministic.
  WAIT is not a forecast of a future sale.
- True Price retains item, shipping, tax/duty, required fees, additional discounts
  and coupons with provenance/confidence. Unknown mandatory components leave the
  checkout total unknown. A known zero must be supplied explicitly. A reported
  list-price discount is not subtracted a second time as a checkout discount.
- Offer comparison requires a trusted canonical variant or compatible authoritative
  identifier; fuzzy titles do not merge sellers. It considers currency, real total
  when comparable, selected-size stock, region, merchant evidence and return policy
  when supplied. BuyCheck compares offers before deciding the budget outcome.
- Local price repositories are bounded and volatile, not full-history AsyncStorage
  stores. Price changes and actual new-day observations are retained; redundant
  same-day unchanged observations coalesce. A stable observed price can acquire
  real historical coverage without fabricated daily backfill.
- **Viewing/checking a cached catalog/manual estimate is not a market observation.**
  Estimates cannot be inserted as real history or masquerade as fresh merchant
  evidence. Current production data is not seeded into synthetic histories.
- A bounded public read adapter is connected to BuyCheck/watch reads. It is off
  unless `EXPO_PUBLIC_PURCHASE_PRICE_READS=true` is explicitly configured after
  the schema/data rollout. Missing/failed remote reads use the local provider.
- Up to 40 local watch targets retain size, merchant/currency/threshold and status.
  Manual evaluation requires fresh matching in-stock evidence, records a local
  check/trigger state, and does not pretend that a notification was sent.
  No external automation, scheduler or push channel was created.

### URL / Share Flow and shopping entrances — 2F, 2S and 2T

- Merchant adapter registry supports Zara, Uniqlo, COS, Massimo Dutti, Jack & Jones
  and J.Crew structured snapshots. Exact known/catalog URL matches preserve the
  original ID; high-confidence model matches require a concrete variant choice.
- Structured ProductGroup objects are not emitted as physical variants. Child
  variants retain their supplied original links. Missing metadata/unknown merchants
  remain partial and can use the manual details flow; no scraping or script
  execution occurs in the UI or this task.
- Public URL normalization rejects unsafe schemes/credentials/private literal
  hosts and removes tracking/session tokens without discarding concrete variant
  selectors. Parsers and payloads are capped. Unknown size charts stay unknown.
- Android cold/hot ACTION_SEND text is converted into the existing app scheme.
  A bounded pending-input queue waits for authentication/navigation, binds an
  anonymous cold intent to the first authenticated owner, and does not carry it
  across later account changes. Existing auth/recovery callbacks still use their
  original processor.
- iOS extension/controller, App Group entitlements/plist, owner-tagged inbox,
  optional React Native bridge and app-side receiver are prepared locally.
  There are no unsupported attempts to force-open the containing app from the
  Share Extension. On opening the app, a valid owner-bound inbox can route to
  BuyCheck. Target membership/provisioning remains a separate activation step.
- The shared Phase 1 variant picker is reused, including bounded thumbnail
  rendering. Unverified external photos do not enter the product-only catalog or
  generation pools. Image-only sharing is not activated.
- Profile, Home Essentials, the Shop header and shopping cards lead to one intent
  hub: complete wardrobe, find, check, improve, save and budget. All use the same
  fit/BuyCheck/wardrobe/money/price services. Results and caches stay bounded; no
  new endless marketplace or parallel outfit state was introduced.

### Server evolution

Prepared `20261102173109_purchase_intelligence.sql` creates only new objects:

- Private owner-RLS tables: `purchase_fit_evidence`, `purchase_fit_profiles`,
  `purchase_price_watches`.
- Public active-data read tables: `shop_merchants`, `shop_offers`,
  `shop_price_history`, with service-only writes.
- Invoker trigger function `record_shop_offer_price()` and its offer trigger;
  required new-object indexes and explicit grants.

Offer identity is immutable. The compatibility listing FK is nullable with
ON DELETE SET NULL: existing catalog hard-deletes are not blocked, archived offer
identity/history remain, and orphan quotes are hidden by read policy. Existing
catalog/RLS/RPC definitions are not rewritten. The data-preserving rollback revokes
new exposures/writes and disables the new history trigger; it does not DROP user
fit evidence or observed prices. No backfill or sync job was run.

## 2. Existing behavior preserved

Protected contracts: original catalog IDs; listing/family/variant/physical identity
separation; physical deck uniqueness; product and shoe rotation; selected-item and
anchor pinning; wardrobe-only/shop-only generation isolation; native budget/price
floors and trusted FX; weather/climate/activity/occasion/style gates; product-only
provenance; deterministic/offline fallback; reader cancellation/request sharing;
five-minute cache behavior; wishlist/recent-search APIs; fit engine golden behavior;
auth, subscriptions, entitlements/rate limits and account cleanup.

No edits to active Edge handlers/shared outfit rules, legacy Express, fit engine,
wardrobe data contracts, scan pipeline, applied migrations or existing RLS policies.
New purchase evaluation cannot inject synthetic URL/manual IDs into generation.
The original account-cleanup registry now also drains/clears the new private key.

## 3. Files changed

Exact Phase 2 authored/shared paths and the untouched pre-existing/concurrent
worktree inventory are in `PHASE2_FILES.md`. Shared files already contained Phase 1
edits; this phase layers its changes on top rather than claiming the entire diff.

No local commit, push or remote PR was created. Catalog JSON/manifest/expansion
changes from other activity were preserved, including their new/deleted files.

## 4. Tests and checks

FINAL_VALIDATION_PENDING

Relevant scopes include fit evidence/reference/aging/ownership, feedback UI,
BuyCheck dimensions, all Find Better modes/tradeoffs, money tools, useful-look
bounds, price/freshness/True Price/offer/watch behavior, URL/shared variants,
owner-bound native receiving, UI stale-input protection and synthetic performance.
Existing regression scopes include outfit variety/physical and shoe rotation,
source modes, anchors, budget/FX/slot/occasion/climate/activity, auth/recovery,
subscription authority, garment detail and private-persistence cleanup.

## 5. Performance

PERFORMANCE_MEASUREMENT_PENDING

These are synthetic development-host measurements, not device/production latency
or quality claims. No full catalog, embeddings or thousands of products are sent
to mobile/LLM. New reasoning has no LLM calls. Shop retrieval remains bounded;
intelligence loads with the new lazy screen rather than the startup catalog path.
Only thin share intake is added at startup. Images remain bounded/lazy.

Local retention limits: 160 fit observations, 80 compact priors, 24 references,
20 bookmarks, 40 watches; public/session price cache 100 offers and 90 observations
per offer. History reads are bounded to 90 per offer and eight offers per read.
Older local evidence can be evicted before remote persistence is activated; this
is not a promise of unlimited on-device history.

## 6. Works without remote activation

Local feedback/settings/references/Fit DNA and evidence-based sizing; BuyCheck,
duplicate/value/gain/gap evaluations; seven alternative modes; cheaper/single-piece
look tools; known catalog URL matching, supplied offline merchant snapshots and
manual fallback; existing scheme intake and locally built Android receiving;
True Price/price assessment/comparison on supplied real observations; local watches
and manual checks. Missing data yields explicit unknown/insufficient results.

Existing catalog reads/fallbacks work without the new Phase 2 tables. Price reads
are off by default and fit data is not uploaded. The Phase 1 discovery RPC remains
separately prepared; its pre-activation bounded fallback limitations still apply.

## 7. Requires remote/platform activation

### New Supabase schema

Target from the supplied project context: `wardrobe` / `fyqpifmrsftsfqibhwhy`.
First verify current schema/collisions read-only; do not assume the old snapshot
is today's production state. Apply only the reviewed migration after approval.

Purpose/benefit: durable owner-private fit/watch data and real multi-merchant
price history. Risk: MEDIUM — new grants/RLS, wrong identity/source data from a
future collector, query/index load and client/schema version mismatch. Existing
objects are not replaced; app remains usable through local stores/fallbacks.
Rollback: `backend/supabase/rollback/purchase_intelligence_disable.sql` after
stopping any separately activated producer. It preserves data.

Schema application alone does not upload fit data, collect merchant prices or
start notifications. Cloud fit synchronization/consent and ingestion/observer
jobs are not active runtime features in this phase. They require separately
reviewed implementation/configuration, real evidence and approval before writes.
Public price reads can then be enabled in an approved client rollout; errors still
fall back locally. No source/secret/app_config change was applied remotely.

### Native distribution / iOS App Group

Android source compiles locally; an updated installed build is needed for OS share
receiving. No build was published. iOS requires an extension target such as
`com.aiwardrobe.PurchaseShare`, App Group `group.com.aiwardrobe.purchase` on both
app/extension, adding bridge/controller files to their proper targets, signing,
and device runtime validation. Source, plist and entitlement inputs are prepared
under `native/purchase-share/ios`; no developer-account capability or provisioning
was changed. Existing app remains usable with paste/deep links when the bridge is
absent. Rollback is to remove/disable the new target/filter and keep paste intake.

### Price watching

Background collectors/schedulers/notifications are intentionally absent. Their
future activation needs a real observed-data pipeline, size-stock evidence,
notification consent, delivery checks and separate approval. Local targets and
manual evaluation work without it. No Codex automation was created.

## 8. Remaining gaps / practical limits

- This is the local Phase 2 foundation, not the completed worldwide shopping vision.
- Merchant live fetching is not activated; unsupported/non-catalog links can need
  supplied structured metadata or manual entry. No retailer-wide scraper coverage
  or universal size/offer matching is claimed.
- Real size charts, feedback coverage, regional prices, checkout fees and dated
  history are needed for confident results. Native MYR amounts work; unsupported
  cross-currency comparisons cannot invent new exchange rates.
- Useful outfits follow the existing four-slot architecture. Unsupported dress /
  accessory look structures do not get fabricated outfit counts. Existing physical
  identity may conservatively reduce gains for ambiguous generic-name products;
  its production semantics were deliberately preserved.
- Wardrobe value and fit confidence policies are deterministic evidence heuristics,
  not calibrated fit probabilities, financial ROI or guaranteed fit/quality.
- Native device visual/end-to-end share QA and signed iOS extension integration
  remain before shipping. Detailed evidence text is primarily English; intent
  entrances are localized in the existing en/ru/uz resources.
- Remote fit sync, live collectors and background notifications are not silently
  implied by prepared schema or source code. They are separate activation work.

## 9. Remote actions

None performed by this task: no hosted SQL/DDL, migrations, RLS/Auth/secret change,
Edge deploy, import/sync/backfill, storage upload, GitHub push/PR, application
publication or external scheduler/notification activation.

**REMOTE ACTIVATION PENDING APPROVAL**
