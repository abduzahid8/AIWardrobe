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
