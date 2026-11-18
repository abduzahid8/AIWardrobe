# Stylist 6 — actual iOS fashion quality checkpoint, 18 November 2026

**Status: STAGING FEATURE AND TRUE PHOTO REVIEW WORK; GREAT PERSONAL FASHION STYLIST NOT ACCEPTED.**
This is a release decision checkpoint, not a design-model marketing claim.

## Deployed exact revision

- Repository: `abduzahid8/AIWardrobe`, feature branch `codex/stylist6-visual-first-20261110`, PR #15.
- **LIVE Render staging source SHA:** `bd65cd8094f6eece571ef03d1c7be5291254550c`, deployed via `dep-dbukf9lg1s2s73c4sc1g`.
- GitHub commit checks: **verify-visual-first-prototype SUCCESS**, **verify-integrated-stylist6 SUCCESS** on that exact SHA.
- Locally: **251/251 profile tests PASS** with the appropriate TypeScript test loader.
- No release to `main`, no switch of production `generate-outfits` away from Stylist 5.
- Strict shoppable SKU deck: a top, one meaningful outer layer, a bottom and shoes; no synthetic fourth item, no unapproved photo fill, no unauthorized rehosting.

## Actual authenticated native iOS, not a mock

Device: booted `AIWardrobe-S6-QA-20261115` iOS 26.5; the app uses the staging Supabase edge JWT/HMAC gateway and the 72-second Render visual-fashion compute.

| Input | First-stage compositions | Independently photo assessed / approved (all waves) | Feasible reviewed triples | Final native iOS cards | Render report |
| --- | ---: | ---: | ---: | ---: | --- |
| Classic / New clothing / Date night | 8, Nemotron + Kimi | 9 / 4 | 4 | **3** | 46,923 ms; photo editor applied; different-lab jury `arbitrated_primary`; 3 selected |
| Classic / New clothing / Work | 8, Nemotron + Kimi | 11 / 2 | 0 | **2** | 37,051 ms; photo editor applied; independent jury not run; creative correction 3 attempts, 0 approved |

Both native flows completed exit code 0. Images and source product media stay in local iOS QA artifacts; **no unlicensed merchant photographs were committed to GitHub**.

### Qualitative manual visual review, Date night

The three real four-product cards differ, but this is **not** yet a convincingly excellent date-night capsule:

1. White collared shirt, long charcoal coat, dark tailored trousers and dark leather loafers, labeled **“Precise office tailoring”**. Visually quite office-oriented for a requested date, despite the individual photo approvals.
2. Pale blue shirt, navy blazer, brown trousers and brown dress shoes. Coherent quiet business-casual tailoring, but again strong office associations.
3. Dark high-neck sweater, dark blazer, gray trousers and black ankle boots. Different top/footwear; a more appropriate evening mood, but still shares jacket-and-trouser architecture with #2.

The three-card count is therefore a **measurable engineering milestone, NOT proof of artistic quality**. There are residual architecture similarities and an explicit occasion mismatch. The owner's main acceptance requirement is not met.

### Root-cause correction in this release

The owner-provided archive of 46 captions contains real different wardrobe directions. Previously Classic / Work and Classic / Date night yielded **three almost identical turtleneck / high-neck knit inspiration examples** even though the archive contains collared shirts, tees/polos and alternate outer layers.

- `editorialTextMoodboards()` now retrieves more **human-captioned** top *and* outerwear constructions; it does **not** invent source clothing or make a regex outfit rule.
- The initial independent Kimi stylist can also examine these bounded user-captioned *ideas*; it still chooses only from unused **source-bound SKU tokens**.
- No copyrighted image is rehosted or sent to that studio; explicit human captions are not wrongly advertised as source-photo observations.
- CI now runs moodboard regression tests. The original NVIDIA whole-outfit photo critic, audited permissions, final true-photo collection jury and protected budget/authorization remain unchanged.

## Inventory truth and outstanding limitations

Current source/URL-matched visual attribute profiles: active top ~2,402/3,082, outerwear ~1,659/2,117, bottom ~1,290/1,672, shoes ~1,643/1,649. Many present profiles contain only slot/category evidence; source photo *color, fabric or volume* observations remain sparse. **There are zero licensed READY cached source-photo records**, so *actual-photo-before-initial-composition* remains OFF. Do not silently turn on photo-first flags or infer rehosting rights from an accessible merchant image. A trusted source profile is not a license to cache or transmit the source JPEG.

## Next decision-oriented work — not more arbitrary apparel rules

1. Improve the **complete-look fashion decisions** before photo review, using actual source-verified attribute evidence and the original owner's intentionally distinct silhouette concepts; inspect failures in real SKU photo boards rather than changing token-count thresholds.
2. Repeat native authenticated Date and Work with different catalog windows. Record full-look rejection reasons **and** hand-evaluate all four product photographs: period, fabric rhythm, layer-to-top relationship, pants/shoes, distinctiveness, personal relevance. Three is not a success if one looks like officewear on a date.
3. Obtain legitimate hosting/external inference rights and actual first-party media coverage to activate the previously built photo-first creator **only when provenance requirements pass**. Alternatively prove that text + source-verified image feature descriptors meet the same human quality bar; do not claim the original photos were viewed in the text-only path.
4. Run at least 30 **held-out**, 3-reviewer **blind human S5-vs-S6** comparisons; require >=70% preference for S6 and verified 3×4 shop-real look quality before promoting. Audit quota, latency, merchant evidence and rollback.
5. Keep production Stylist 5 untouched until those objective and owner gates have been met.

**Bottom line:** One authentic 3-look Date result is a real milestone. The 2-look Work counterexample and visual office-style mismatch prove the world-class stylist goal remains open.

## 2026-11-18 follow-up — photo-worker download transport, bounded source pilot

A deeper LIVE source-data audit explained another visual quality ceiling:
**1,520 active top/layer/bottom source-hash-current SKUs had no attached v1
photo profile**, and every corresponding `catalog_visual_jobs` row was
`unknown`, `attempts=3`, `reason=download_unavailable`.
These are NOT merely unqueued jobs. Host-level distribution among the
failed source photographs included Armani CDN 768, Loro Piana 263,
Fred Perry 153, Zara 145, Lacoste 105, Massimo Dutti 85 and one Jack & Jones.

Repro from a neutral macOS GET sample per host:
- The exact original URLs for Armani, Loro Piana, Zara, Massimo Dutti
  and Fred Perry returned **HTTP 200 JPEG** with a truthful
  `User-Agent: AIWardrobe-Stylist6-Cloud/1.0` and neutral
  `Accept: image/*,*/*;q=0.8`.
- Several of these URLs rejected the previous plain Python Requests
  default UA with **HTTP 403**. This is evidence of a *transport
  compatibility issue*, not proof that every failed image is now
  recoverable from a GitHub cloud runner.
- The sample Lacoste image still returned **403**, and is deliberately
  excluded. No anti-bot/session/cookie spoofing was attempted.
- Worker change `b954367eb979bb4b01374a618df487800b6e57b6` gives the
  original product-photo download an honest client identity and Accept
  header **without changing** its public HTTPS/DNS, redirect, 12 MB byte,
  30 MP pixel, source-hash and timeout checks. Two dependency-free
  fail-closed transport regression tests pass.

**Pilot, not a completed profile claim:** requeued ONLY **48 previously
failed, currently source-hash-matching, active and priced original SKUs**
under `catalog_visual_jobs` (16 Armani, 8 each Loro Piana, Zara,
Massimo Dutti and Fred Perry). No wholesale rescan or Lacoste retry.
Before GitHub Actions startup Supabase showed `pending=48`,
`succeeded=6,992`, `unknown=1,509`, zero invalid pending hashes.

Run: `https://github.com/abduzahid8/AIWardrobe/actions/runs/44254765390`.
Four bounded GitHub workers confirmed started; they passed the new
transport regression checks and were installing the pinned CPU model
at last verification. **Do not report the pilot as successful until
Supabase confirms finished, source-matching `catalog_visual_fashion`
rows**. If cloud runner still reports `download_unavailable`, do not
indefinitely requeue; classify status/rights and source merchant feed
before making any further changes.

Original photo hosting licenses remain **unverified**; this offline
feature extraction is NOT proof of any merchant caching/rehosting right.
The licensed original-photo-first creative cache still has **zero rows**.

### Confirmed controlled transport recovery and follow-on bounded cohort

After the new-source worker image loader deployed in commit
`b954367eb979bb4b01374a618df487800b6e57b6`, **ALL 48 of the 48**
originally failed pilot products completed SigLIP2 inference and acquired
**current URL + source-hash verified** v1 visual profiles, separately checked
against the live Supabase catalogue. The durable visual job total rose
from **6,992 to 7,040 succeeded**; 48 moved out of the old
`unknown` bucket without new merchant data or fabricated source URLs.
The pilot GitHub Actions run **44254765390** completed **all four workers
SUCCESS**. This confirms the neutral truthful UA compatibility recovery
on the five permitted source hosts tested.

A second, still bounded improvement tranche queues **exactly 512**
additional same-source, already-failed legitimate top/outerwear/bottom
products: Armani 256, Loro Piana 104, Fred Perry 60, Zara 60, Massimo
Dutti 32. These are **QUEUED, NOT YET PROVEN RECOVERED** at the time
of this note. Existing workflow runs four capped 128-image CPU workers,
all with `STYLIST_VISUAL_SKIP_SEED=1`: the job cannot accidentally
reseed the whole catalogue. Lacoste's consistently blocked 403 URLs
were not retried or bypassed.

Improving SigLIP2 evidence should give the AI source composer a stronger
photo-derived color/construction menu; it is **not** artistic acceptance
or evidence of a source-photo commercial caching/rehosting license.
