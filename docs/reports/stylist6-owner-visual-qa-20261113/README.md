# Stylist 6: owner iOS visual QA, 2026-11-13

Input: 12 owner-submitted screenshots, 20:41–20:42 Tashkent.
Client commit: cd2fd63e. Engine: stylist-6-cloud-staging-1.
The displayed result status for all inspected decks is NVIDIA_ABSTAINED.

## Fashion evidence

Old Money/Work, Classic/Work, Semi-Classic/Romantic Date produced three
complete looks each, but completion does NOT equal fashion quality.
Some outfits are wearable. Actual concerns visible in photos:

- Old Money/Work mixes a utilitarian Navy Magnetic Field Jacket with
  traditional trousers; another dressy look has heavily lugged loafers.
- Classic/Work repeats the exact same black wool Zara blazer in two
  different looks, limiting meaningful wardrobe directions.
- Semi-Classic/Date is generally wearable but formality/register
  contrasts between cards are sometimes marginal.
- Most displayed looks combine USD/EUR and UZS original seller prices,
  and correctly have no fictional converted total. A separate UZS
  budget already activates the verified single-currency hard cap.
- There is no verified size-specific stock or user-delivered checkout
  total for these screen-selected SKUs, nor evidence of independent
  human preference wins. Stylist 6 remains production NO-GO.

## Root cause: unavailable reference media

Key-NAME-only Vercel Preview inspection confirms STYLIST6_REFERENCE_BUCKET
is NOT configured. The cloud engine still attached reference IDs from
46 prebuilt fashion reference metadata to its candidate shortlists.
The critic attempted to resolve those missing references before judging
candidate clothes. The result was reference_unresolved, with zero
provider calls and NVIDIA_ABSTAINED, as reproduced with strict mocks.

The fixed critic has an explicit referenceMediaConfigured flag:
- false ONLY when no owner reference bucket configured: judge the outfit
  using its actual product photographs only; never send a reference
  photo or claim a reference-fidelity score; use a reference-less
  cache key and prompt
- true when configured: any inaccessible/missing reference still
  abstains and cannot be silently bypassed

The returned visualReceipt includes referenceBucketConfigured, which is
not a secret.

## Root cause: image format admission

The old Node media path only admitted small JPEG files (max 512 KiB).
This excluded real PNG, WebP, AVIF and larger JPEG seller photos;
a live photo from the ECCO catalog, for example, has a PNG URL.

The new Node-only Sharp 0.35.5 media path handles the above formats
using actual decoded source pixels, bounded to compressed input <=4 MiB,
<=16 megapixels, <=1024 px on output and <=512 KiB of JPEG output.
EXIF is stripped; transparency is flattened; no arbitrary SVG or
animated format is admitted. Seller URL host checks and inference host
allowlists are preserved, cross-host redirects are rejected. No blocked
seller hosts are added and no external AI calls are made during tests.

Vercel installation is a direct dependency of the isolated
backend/stylist-node package, NOT production Stylist 5 or the iOS app.

The regression tests now use valid synthetic real JPEG/PNG/WebP pixels
rather than fake JPEG marker bytes. In-memory conversion occurs only
after existing photo transmission allowlist checks.

## New diagnostic

The developer-only screen reports safe receipt counters and first
allowlisted abstention reason, e.g.:

    NVIDIA_ABSTAINED c0/v0/a4 · reference_unresolved

c/v/a = dispatched calls / valid model verdicts / abstentions. Never
echo image URLs, SKUs, auth tokens or arbitrary server error messages.

## Staging and release

The owner screenshots came from cloud staging v20 pinned to an earlier
compute SHA. Updating PR #15 and its Vercel Preview does NOT update
the separate Supabase staging proxy until explicitly repinned.

Manual testing is only conclusive after the stage pin is verified;
avoid automatic paid retries. Before an owner-approved production
release, independently evaluate taste and diversity with complete
outfit cards, actual merchant availability and owner-approved
image-inference permissions. Production Stylist 5 is untouched.

## Follow-up Owner Cloud Test (21:33-21:35, 2026-11-13)

User-provided iOS screenshots: client SHA 94dd603e1344, cloud stage v21
pinned to READY Vercel compute 56402783. NVIDIA now returns actual
NVIDIA_APPLIED verdicts, unlike the prior universal NVIDIA_ABSTAINED.

| Scenario | Provider calls | Valid verdicts | Abstentions | Failure |
| --- | ---: | ---: | ---: | --- |
| Old Money / Work | 3 | 2 | 1 | provider_deadline |
| Classic / Work | 3 | 2 | 2 | product_image_unresolved |
| Semi-Classic / Romantic Date | 2 | 2 | 1 | product_image_unresolved |

These are attempted critic candidate counts (including a possible
challenger), NOT the number of displayed looks that passed visual review.
NVIDIA_APPLIED does not mean all three displayed cards were judged.
The one provider deadline has not been declared fixed.

Visible fashion defects: repeated merchant shoe SKU and burgundy merino
top across looks; extremely bright zip-front "Rust Reversible Resort Jacket"
paired with tailored date-night clothes and black loafers; mixed EUR/USD/
UZS price totals remain unconvertible without trustworthy FX and seller
destination data. Size availability and image-use rights remain unverified.

### Engineering follow-up

- Request-scoped in-flight photo Promise deduplication: shared footwear
  cannot issue duplicate concurrent downloads to the same retailer CDN.
  Failed photo requests are also shared (no 403/429 download storm).
  No cache crosses user requests or persists to database.
- Deadline now covers HTTP response streaming (not only headers).
  Image body streamed with 4 MiB cap; blocked redirects fail closed.
- Safe visual diagnostics: bounded media slot and HTTP failure category,
  with no raw URL, user data, auth token or arbitrary error echo.
- Separate displayedJudged/displayedLooks counters from all NVIDIA
  attempts. Dev UI explicitly says shownN/3.
- Narrow semantic admission: a new-clothing quiet smart romantic date
  rejects only an explicitly named reversible resort jacket/blouson,
  not ordinary jackets, casual contexts, owned pieces, or user-requested
  resort/technical layering.

No additional real NVIDIA calls were made to build/test these changes.
Full live fashion/merchant acceptance and owner-approved deployment are
still pending. Production Stylist 5 unchanged.

## Second owner iOS test — 22:09–22:11 2026-11-13, cloud v21

Screenshots show local client `c14fb14a` but the isolated Supabase
staging Edge function (v21) was **still pinned to Vercel compute
`56402783`**. Never attribute these cloud pictures or NVIDIA receipt
counts to the local client's SHA. Both services used separate versions.

- Old Money / Work: `NVIDIA_APPLIED c4/v4/a0` (four valid candidate
  assessments, not automatically three displayed approvals).
- Classic / Work: `NVIDIA_APPLIED c4/v3/a1`, `provider_deadline`
  (three valid candidate assessments; one provider deadline).
- A single `Failed to send a request to the Edge Function` banner
  also appeared. Supabase `function_edge_logs` showed three subsequent
  successful `POST 200` responses between 17:09–17:11 UTC; the failed
  call is not visible in that function's logs. This suggests an
  intermittent client-to-Supabase transport problem, **not** proof of
  an Edge Function defect. Do not auto-retry an uncertain paid request.

Fashion inspection (not model scoring): the Old Money result sometimes
looks refined (check shirt, navy blazer, walnut trousers, oxford shoes),
but the bright yellow sweater/cream cardigan with grey pants is arguably
outside a quiet office brief. Classic Work repeats the *same grey loose
trousers and same black dress shoes* in two displayed outfits, only
changing the upper garments. Bounded `footwearVarietyAdjustment` already
exists but does not specifically discourage an identical trouser+shoe
pair. Existing ECCO Helsinki/Spherica SKUs remain in catalog and appear
in recommendations; this report does not delete inventory or silently
change merchant brand policies.

### Applied deployment

GitHub general CI for `c14fb14a` and isolated S6 CI were successful;
Vercel compute `4j4645qbs` was READY. Commit `e33dc6cc` changed only
the staging proxy's pinned Vercel URL/commit comment, checked byte-for-byte
against its prior source. Supabase `aiw-stylist6-node-staging` was
updated to **v22 ACTIVE** with `verify_jwt=true`, retaining original HMAC
signing, security limits, and quota. `generate-outfits` production is
still **v209 ACTIVE (Stylist 5)**. No real NVIDIA inference was triggered
by the deployment or its verification.

### Code-only follow-up (unreleased)

A bounded, evidence-based 0.025 ranking adjustment now discourages
identical *bottom AND shoes* pairs across a three-look deck if
comparably good alternatives exist. Existing exact feasibility, budget,
identity, physical-SKU overlap and editorial-quality gates stay intact.
A sparse catalog may still return fewer than three credible directions.
Selector diagnostics report `repeatedLowerPairs` to measure this
quality problem rather than rename the cards. No paid calls made;
version `s6-three-directions-editorial-v4` is **branch only** until
separate staging deployment and owner visual acceptance.

### Real frozen-catalog regression, read-only, no NVIDIA

Replayed the same 384-row frozen live UZS sample, eight scenarios,
reference-first and creative-first arms (42 displayed looks) using
`pilot_12.mjs --skip-v5 true`. The previous archived baseline had **0**
repeated exact trousers+shoes pairs. The new selector also had **0**,
with **0/42** look identity changes. This is a useful *non-regression*,
not evidence that the new tie-break improves real buyer preference:
that frozen sample simply did not contain the owner-observed failure.
A focused 3-candidate deterministic test *does* reproduce the lower-pair
near-tie and verifies the new alternative ordering. No hosted fashion
judgments, user images, remote inference or merchant writes were made.

Do not report a live fashion win until an actual owner scenario finds
an eligible equivalent-quality alternative, the selector uses it, and
independent reviewers prefer the resulting complete outfit deck.
