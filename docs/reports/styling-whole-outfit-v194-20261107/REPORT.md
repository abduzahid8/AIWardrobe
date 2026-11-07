# Whole-outfit compatibility after v193

Baseline: ACTIVE generate-outfits v193, deployed from 141c83b9c7b3ff523a9d849a038f43b4acec6caf.
The downloaded original source, recorded v193 manifest and freshly built baseline
bundle have identical SHA-256 (see checks.json). Implementation starts at 204bf1a,
which adds deployment evidence to 141c83b without runtime changes. The dirty primary
workspace and main were not used as release sources.

## Root causes and targeted changes

- Aurélien `aurelien-15179534139722` is named only Beige Flannel Trousers in
  local catalog snapshots. v193 treats flannel as tailoring; an explicit drawstring
  would already fail refined Work. The [manufacturer](https://aurelien-online.com/products/beige-flannel-trousers)
  confirms its drawstring waist and slim cut. Add a derived exact-ID construction
  override, preserving catalog ingestion/data and unrelated flannel trousers.
- `garmentContextReason` accepts a clean wool zip jacket as an office layer.
  Generic shirt classification gives Tencel the same credit as a woven dress
  base; no relation measured fluid fabric + cropped zip layer + tailored trousers
  + dress shoes. Score and reject that combined refined-office register split.
  A plain shirt with the same wool zip jacket and suede loafers remains valid.
- A denim leisure blazer receives generic structured-tailoring credit. `washed`
  alone deliberately does not prove fading, while the dress-silhouette check
  excludes Derby. Add separate wash/leisure/drape evidence, reduce tailoring
  credit and reject only washed slim denim + denim leisure blazer + formal shoe
  last on dates. Clean denim, washed straight denim with suede loafers, and
  good denim-blazer smart-casual looks remain available.
- `loose-fit` was unknown volume. Recognize it and score wide denim hems against
  dress shoes, with a smaller penalty for smart loafers. The checked-shirt,
  merino, white-loose-denim example is evaluated softly rather than categorically
  banned. Its palette may be fine while its silhouette loses ranking credit.
- Layer role checks reward T-shirt/blazer without considering oversized graphics.
  Single green accents can receive good palette scores even when the silhouette
  is disconnected. Penalize wide expressive bases under structured tailoring;
  reject the simultaneous graphic-volume + structured blazer + vivid lifestyle
  sneaker conflict. Preserve graphics, bright sneakers and soft blazers in
  coherent casual combinations. Casual/Travel gets a modest 0.60 quality floor.
- v193 penalizes sport sneakers against tailoring but not casual denim. Training
  shoes thus pass Travel; `CROSS TRAINERS` can even become ordinary sneakers.
  Recognize cross trainers and gate training footwear only with non-athletic
  bases/bottoms and no explicit active travel intent. Lifestyle sneakers,
  gym travel and outdoor intent remain usable.

## Quality path and unchanged safeguards

`garmentDescriptor` provides derived construction evidence. `layerCompatibility`
scores base/layer fit; `outfitCoherence` scores the four-item relationships.
`occasionRules.assessOutfitQuality` invokes coherence and `outfitQualityFloor`;
`styleJudgeReason`/`passesStyleJudge` retain existing eligibility checks.
`validatedQuartets` applies those gates within its original bounded search.
`curatedOutfitShortlist` retains up to nine approved alternatives and a feasible
seed for one NVIDIA ranking call. Final deck selection and shoe reassignment
recheck acceptance while retaining physical identity, LRU and anchor safeguards.

Only four of 38 bundled runtime inputs change; all other input hashes are exactly
v193. The Edge handler, candidate/query limits, shortlist, rotation, diversity,
source/ownership, budget/FX, weather, auth, quota and subscription code are unchanged.
No new provider calls, queries, candidate loops, schema or catalog writes.

## Minimal verification

46 tests pass across five focused suites, including 16 new garment cases and
three new 40-candidate Edge-handler cases for the reported style/occasion pairs.
Each hosted case yields nine ranking options, one mocked NVIDIA call, three final
outfits, all four slots and 12 unique product IDs. Budget and injected-provider-ID
checks pass in the existing focused hosted coverage. New regression cases had
seven failures on v193 before the patch. No full suite or AI visual batches run.

Focused Deno frozen-lock check, TypeScript dependency-graph check, lint and diff
checks pass. Seven offline post-provider replays pass for fresh, anchored and
quality-replacement paths; candidate CPU 81.8–261.8 ms versus baseline 78.8–261.3 ms,
all below 500 ms. These local measurements do not predict production wall time.

## Release and remaining limits

Release script checks live v193 version, fingerprint and original-source hash,
requires the v193 ancestor and pushed committed bundle inputs, and deploys only
generate-outfits. The verified v193 source archive supports a guarded rollback;
all prior v192 artifacts are preserved. Post-deploy checks compare downloaded
original source to the committed-input bundle, check ACTIVE/auth configuration,
assert other functions unchanged and probe OPTIONS/missing/invalid auth.

Visual acceptance remains with the owner's iOS Simulator QA. Sparse catalog
metadata can still omit visible construction details; only the verified trouser
ID receives an override. Narrower compatibility gates can return 422 when the
bounded pool cannot supply three coherent independent outfits. No authenticated
production inference or paid visual generation was run for this release.
