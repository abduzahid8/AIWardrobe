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
