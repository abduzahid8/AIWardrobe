# Stylist 6 – Product completion, November 13 2026

Objective: a trustworthy Stylist 6 release candidate, NOT Stylist 7.
This engineering checkpoint does NOT constitute a product-quality or release GO.

## Confirmed live baseline

- PR #15 is isolated and remains Draft. Production Stylist 5 Edge generate-outfits
  v209 ACTIVE, Stylist 6 staging v22 ACTIVE at mission start.
- Read-only Supabase SQL: 7,020 active items: 5,267 upper-body, 1,638 lower-
  body, just 115 shoes. Currencies: 3,006 UZS, 2,037 USD, 1,401 EUR,
  576 GBP. Only 2,554 active rows have last-seen import timestamps; zero
  were observed in seven days, only 12 have an external ID.
- Import observation != size-specific inventory, shipping, checkout
  total, guaranteed availability, or third-party photo inference rights.
- Owner screenshots 22:38–22:39: NVIDIA_APPLIED with shown2/3 and
  top/decode_failed. Candidate judgments != judgments on final shown cards.

## Engineering changes

1. The visual reviewer now chooses the final displayed deck, rather than
   contributing only a yes/no veto or swapping the lead. This repairs a
   long-standing gap where an UNJUDGED replacement could masquerade as an
   NVIDIA-reviewed look.
2. The iOS app now accepts partial, genuinely visually approved cloud decks
   (one or two results) after successful server admission. It shows a clear
   RU/EN/UZ explanation instead of reporting a false error after debit.
   Unverified partial responses are still rejected by the client.
3. A spare fourth visual candidate is selected for styling diversity within
   the already accepted editorial quality window (0.16); the former 0.025
   near-tie-only alternative could be little different from the lead.
   Budget, provenance, ownership and hard feasibility remain inviolate.
4. The cloud full-deck review uses the editorial refreshed-look novelty
   boundary 0.12 instead of the prototype discovery-only 0.30 threshold.
   Physical duplicate outfits remain excluded; the independent prototype
   scoring defaults have NOT changed.
5. Every displayed look carries its OWN visual evidence status and the
   server receipt distinguishes full-three-verified, verified-partial,
   abstained fallback, unavailable, disabled, not-requested. A real visual
   rejection of ALL candidates produces visual-quality-unmet instead of
   quietly restoring unapproved looks.
6. Added automated release-candidate gates requiring human blinds,
   visual evidence, true shopping offers, merchant/image rights,
   measured hosted latency, validated security/CI and rollback. The
   tests use synthetic examples ONLY; no test result certifies fashion taste.

## Invariants and boundaries

- No production migration, deployment, commit merge, catalog write or
  SKU reactivation.
- No live NVIDIA calls, charged requests, or retailer data changes here.
- No customer credentials, tokens or images committed.
- No new merchant- or reference-image permissions are assumed.

## Mandatory release gate

A private, owner-reviewed JSON evidence manifest is required.
Run without any outbound calls:

    node scripts/stylist6/release_candidate_gate.mjs .private/stylist6-release-evidence.json

With missing or incomplete proof it returns NO_GO. An entirely synthetic
positive test verifies the gate itself only. Even if the gate finds an
ENGINEERING_RELEASE_CANDIDATE it sets deploymentAuthorized false.
No production change is authorized by passing this script.

## External blockers

- Partner-authorized variant offers, sizes, country delivery, stock,
  landed price, refresh time, and merchandising quality. Need 200 genuine
  locally eligible distinct shoes as a MEASURED coverage goal.
- Explicit photo rights for merchant display AND NVIDIA inference,
  separately; public image URLs are not licenses.
- Three independent human judges for 30+ paired S5-versus-S6 held-out
  complete-deck comparisons. Existing NVIDIA-family 5W/5L is not a win.
- Owner-authorized staging verification on exact Vercel+Supabase SHAs,
  actual 3/3 proof, partial rendering, call cost, p95 and safe quota.
- PR #15 changed-file hygiene, signed native build and rollback review.

Production Stylist 5 remains the fallback until every gate and owner
authorization succeed.
