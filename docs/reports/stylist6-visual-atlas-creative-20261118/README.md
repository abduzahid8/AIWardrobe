# Stylist 6 — visual-grounded personal Creative Director: real native QA, 18 November 2026

**Release verdict: NOT excellent yet; production NO-GO.** This is a foundation-level creative-decision change, NOT a small list of fashion category bans, and NOT an AI-generated taste victory. No new outfit subtype veto or lowered photographic approval floor was introduced.

## Root cause and architecture change

Before commit `963333af08f27dddf1c5f080f139c6624a3726c1`, the first 120B Creative Director received primarily counts of garment subtypes and optional text reference captions. The richer source-hash-verified original-photo observations and the user's *explicit* look-level feedback were available later for SKU selection, but were not present when the initial designer decided which **three complete styling ideas** to pursue. In real iOS tests, this generated too many near-duplicate blazer + shirt + trouser capsules.

On the existing 72-second **isolated** staging path, the Creative Director now receives:

- `verifiedGarmentVisualAtlas`: anonymous, bounded maximum eight garment subtype families per role, maximum two genuine *single-product* visual-feature examples per family; actual up-to-date source-hash + original URL-matched v1 photo profile mandatory, with confidence and margin checks. Permitted observations describe **color, texture, volume, length, formality, pattern, sole profile, collar**. Stale source, mismatched original image, low-confidence or unsupported attributes are excluded; **no SKU IDs, raw garment photos, model prompt injection, price, merchant URLs, or personal identity** leave the planner.
- `explicitLookTaste`: opt-in positive/negative entire-outfit relationship signals already sanitized by the same authenticated source adapter as the SKU composer. Strongly disliked ensembles are not translated into one-item bans; explicit user taste now influences artistic planning rather than only a late product ranking.
- Stable bounded inventory metadata and original user style/occasion/fit/weather; the full existing SKU source/price/budget/physical/stock checks and independent real-photo critic still have authority. The model does not claim it saw the raw product photos, that image-rights licenses exist, or that it has approved products.
- New server-only `creativePlanVisualExamples` aggregate counter means actual live QA can distinguish zero real visual evidence from properly hydrated inputs, without logging the user or an item.

This is *visual-feature-grounded planning*, **not** full-original-photo visual-first SKU design. The legally licensed four-role source-photo cache still contains zero ready rows. Do not rename it photo-first creation or claim acquired merchant inference/rehosting rights.

## Implementation and real verification

- Code SHA: `963333af08f27dddf1c5f080f139c6624a3726c1`, feature branch `codex/stylist6-visual-first-20261110`, PR #15.
- Render staging service `srv-dbtd336i0phs73fd0sbg`; deployment **`dep-dbull2lckfvc73ciu56g` LIVE**, exact code SHA above; same JWT/HMAC Supabase gateway, production Stylist 5 untouched.
- Full local Mac matrix: **544 / 544** backend Stylist 6 tests PASS, **42 / 42** Node/gateway tests PASS, TypeScript check and portable Render build PASS. Regression tests cover visual-source hash freshness, confidence, no SKU/URL leaks, separate real-item observations, and explicit taste entering first planning prompt.
- Native simulator: real booted iOS 26.5 `AIWardrobe-S6-QA-20261115`, app `com.aiwardrobe`; Maestro flows verified the chosen request context and actual generated cards before exiting successfully.

| Real scenario | Real visual feature examples in FIRST Creative Director | AI-composed source candidates | Original-photo critic approvals / attempts | Final shoppable 4-item looks | Real server elapsed | Final photo jury |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Classic / New / Work | **23** | 6, all 6 admitted as genuine source sets | **4 / 11** (all waves) | **2** | **52,751 ms** | primary applied, independently arbitrated primary (3 votes) |
| Classic / New / Date night | **29** | 8 from independent Nemotron+Kimi initial creators | **4 / 11** (all waves) | **2** | **59,777 ms** | primary applied and independent real-photo jury agreed (2 votes) |

Both exact hosted runs set `catalogFourItemContract=true`, `catalogFourItemDropped=0`, `photoEditorialVisualMedia=ready`, `creativePlanStatus=planned`, with no padding. They also attempted real photo-critic feedback revisions; a partial third candidate does not automatically make a three-look **excellent** portfolio. Actual photos were inspected on iOS, but remain in private QA artifacts, not copied to GitHub without rights clearance.

### Honest aesthetic assessment

The two Work cards still emphasize precise office tailoring; the Date cards combine a light fine knit + dark layer/tan trousers/boots and a pale blue shirt + dark blazer/trousers. They may be wearable, but **do not demonstrate three genuinely different, outstanding fashion directions**. The new feature changes what the first AI can consider, not guaranteed artistry. The sampled merchandise differs between separate authenticated generations because each new logical request generates a new source-discovery cohort: even the rise in photo approvals relative to some earlier runs is **not controlled causal evidence**.

## Next PRIMARY engineering milestone, not another rule

1. Build and run an immutable **source-verified same-SKU cohort** fashion baseline with exact product photos/contact sheets under appropriate rights and frozen user occasion/style. Compare the initial AI design directions, real SKU composition, photographic critic rejections, portfolio quality and latency on the **same inputs**. No new one-off title regexes or forced third-card fill.
2. Make a substantive original-photo-first source composer when both independently verifiable product media and external AI inference permissions exist for all top/layer/bottom/shoes. Until that legal/media inventory exists, use the honest photo-derived fashion features above; do not silently pretend the 120B first-pass model sees original pictures.
3. Require >=30 held-out Stylist 5 vs 6 comparisons, three independent human blind reviewers and >=70% S6 preference, 3 excellent diverse looks where inventory supports them, with zero source/budget/stock/identity rule violations. The independent photo model cannot substitute for this panel.

**No promotion to production.** Results support a genuine source-grounded Creative Director input improvement, but do not demonstrate owner acceptance.
