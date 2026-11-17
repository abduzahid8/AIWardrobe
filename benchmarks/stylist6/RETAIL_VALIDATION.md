# Stylist 6 retail validation

## Obtain permissions and current inventory together

| Source | What it can supply | Admission decision |
| --- | --- | --- |
| Direct merchant feed and signed addendum | Exact product and variant identifiers, photographs, live size stock, market prices, delivery conditions and a grant from the rights holder. A merchant-operated Shopify storefront can expose per-variant availability, SKU, selected options and contextual price. | Preferred pilot route. Request one merchant and one destination first. Confirm who owns the photos, and obtain an asset schedule and explicit permission before enabling inference. [Shopify ProductVariant](https://shopify.dev/docs/api/storefront/latest/objects/productvariant) |
| Awin advertiser program or feed | Platform-supplied promotional images and product data under advertiser program terms. | An accepted publisher can use supplied promotional assets within scope. Review the governing program and obtain a written AI processing addendum. Awin distinguishes its supplied material from images copied from a merchant site. [Awin image permissions](https://success.awin.com/s/article/Can-I-use-any-images-from-the-advertiser-s-website?language=en_US) |
| Rakuten Product Catalog or Product Search API | Partner-provided product links and catalog data. | Check network and advertiser terms. Promotional placement, resizing and tracking rules do not establish an NVIDIA submission grant. Use a specific merchant/network addendum where both hold relevant rights. [Rakuten Product Links](https://pubhelp.rakutenadvertising.com/hc/en-us/articles/10623933503373-Product-Links) · [Publisher agreement](https://rakutenadvertising.com/legal-notices/publisher-membership-agreement/) |
| Amazon Associates or its product API/feed | Product advertising content under a purpose-limited license. | Exclude from this pilot unless Amazon gives express written authorization covering the planned processing. The standard license restricts analysis and repurposing. [Amazon Associates policies](https://affiliate-program.amazon.com/help/operating/policies) |
| AIWardrobe-commissioned or merchant-licensed product photography | Cleared photos keyed to the exact merchant SKU, variant and color. The inventory feed remains a separate source. | Suitable if the chain of title and signed rights cover the intended external processing. For references, use original commissioned work or a documented open license compatible with NVIDIA terms. |

Ask the merchant or licensor for a dated agreement and an asset schedule identifying the exact SKU/variant, image URL and revision. Record independent grants for catalog display, local copying and transformations, and uploading copies to NVIDIA for internal outfit evaluation. Specify global processing, any vendor retention or service-improvement rights under the applicable trial terms, attribution, access controls, expiry, revocation and deletion obligations. A requirement that prohibits provider service improvement is incompatible with the trial scope used here. Obtain approval from the rights holder and network where both control the asset. [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf)

The first merchant export should cover the four outfit roles, with exact merchant variant ID and size system, in-stock state, currency and price, target country, delivery acceptance, landed price and observation time. A Google Merchant Center-shaped export is a useful input format. A feed row and a reachable URL still need destination and size verification. [Google product data specification](https://support.google.com/merchants/answer/7052112)

## Private pilot controls

Place `image-rights.json` next to the private organizer inventory. The manifest is a trusted reviewer’s decision log. Retain the actual agreement separately and record its reference and SHA-256. For a product, the source URL must equal the catalog photo URL. The SHA-256 binds the exact submitted derivative. One asset can be renewed without changing unrelated decisions.

```json
{
  "version": 1,
  "assets": [{
    "kind": "product",
    "id": "exact-catalog-sku",
    "sha256": "<64 hex digits of the submitted JPEG>",
    "sourceUrl": "https://merchant.example/exact-photo.jpg",
    "rightsHolder": "Contracting rights holder",
    "reviewedBy": "Internal reviewer identifier",
    "evidence": { "reference": "private-agreement-and-asset-schedule", "sha256": "<64 hex digits>" },
    "status": "active",
    "validFrom": "2026-11-10T00:00:00Z",
    "expiresAt": "2026-12-10T00:00:00Z",
    "display": false,
    "nvidiaTrial": {
      "submit": true,
      "territory": "global",
      "providerServiceImprovement": true,
      "terms": "nvidia-api-trial",
      "purposes": ["internal_outfit_evaluation"]
    }
  }]
}
```

A reference uses `kind: "reference"` and its scenario reference ID. Dry runs show per-asset decisions and zero calls. Live entry requires a unique active grant for every image, before the dispatch claim. A display-only, expired, revoked, ambiguous or mismatched grant fails closed. Remove or revoke grants promptly and reconcile any derivative and receipt retention obligations with the provider agreement. The software checks the trusted decision, while the reviewer verifies the underlying contract. Keep both the grant registry and media out of Git.

Use separate `merchant-offers.json` and `purchase-requirements.json` files for the development audit. An offer supplies exact `sku`, `variantId`, `url`, `size`, `sizeSystem`, `country`, `stock`, `shippingConfirmed`, `currency`, `price`, `landedPrice`, `observedAt`, `source`, `evidenceRef` and `evidenceSha256`. A scenario requirement supplies `country`, `currency` and size plus size system per garment slot. Audit requires exact match, 24-hour freshness, an authoritative merchant feed/API or verified checkout, all mandatory roles and total landed budget compliance. The audit never substitutes a different SKU or invents FX.

## Actual results and review

The read-only production catalog query found 7,912 rows, 7,020 active, 12 active with an external ID, 2,554 active with an import timestamp and zero seen in the preceding 24 hours. All 59 frozen rows matched active live rows with no current catalog price, currency or URL drift. The 72 review looks use 45 distinct SKUs. The deployed catalog has no variant stock, size, shipping or image-license columns. The optional offer and merchant tables found in repository migrations were absent from the live schema. No authorized merchant offer was available for the study, and zero looks are purchase verified.

The same frozen SKU and price pool, scenario brief, occasion, style, budget, structural validity and hard compatibility are used for both arms. Stylist 5 retains its native eligibility and taste filters. Applying its shirt-only base-layer rule to Stylist 6 eliminated the reference shortlist in seven of twelve cases, so the replay shares structural and hard compatibility gates while preserving each engine’s fashion taxonomy. For the control, `retail_replay.mjs` verifies the exact Stylist 5 handler source hash, reuses its ranking prompt and actual ranker, and passes the returned ranks to the actual planner’s selector. The run used the same Nemotron Super VL model with a text-only payload. Twelve HTTP 200 responses returned valid rankings. Final decks matched the seed decks in all 12 cases. This is a hosted ranker replay on a frozen local planner, without live user state, cached visual features or verification of the production provider model override. Stylist 6 uses retrieval because the exact retail images are uncleared. Thus the 72-look package measures the baseline and retrieval arms, with hosted visual lift still unmeasured.

The browser package contains 12 randomized anonymous deck pairs, keyboard-operable score controls, local progress persistence when available, and a downloadable submission. Keep the key separate. Assign three unique reviewer codes and gather one submission per reviewer. Scoring reports per-scenario votes, majority wins/losses/ties/splits, an exact sign test on decisive cases and paired scenario-bootstrap intervals. Purchase scores measure plausibility only. Retain critical flags for adjudication. Independent review count is currently zero.

```sh
node scripts/stylist6/retail_replay.mjs /private/fresh-organizer
node scripts/stylist6/retail_audit.mjs /private/organizer
python3 scripts/stylist6/blind_review.py build /private/organizer /private/review
python3 scripts/stylist6/blind_review.py score /private/organizer/retail-blind-key.json reviewer1.json reviewer2.json reviewer3.json
```

For a separate free-tier text ranking replay, add `--free-tier-v5` in a fresh organizer directory with the licensed trial credential in the environment. It makes at most 12 sequential requests with at least 3.5 seconds between starts, rejects redirects and non-text payloads, uses the original ranker deadline, makes no retries, and stops on payment, access or rate-limit responses. The exclusive organizer claim prevents accidental replay. Credentials and hosted outputs stay private.

## Reliability protocol

The archived Q01 receipts contain nine valid verdicts in ten attempts across three prompt revisions. Two false missing-slot claims are verifiable against the packed assets. The final revision produced one valid result and one generic contract failure in two attempts. That historical failure cannot be distinguished as malformed JSON, truncation or a schema mismatch from the receipt. The current critic records separate redacted reasons for JSON, schema, provider truncation/refusal and slot coverage. Self-reported visibility is a model claim. A correct slot name alone does not establish recognition.

Preregister a 16-call licensed experiment using the four CC0 outfits, one fixed prompt/model revision, and four runs per outfit. Use two runs in canonical slot order and two in a rotated order with labels still adjacent to the correct image. Balance order across batches, cap concurrency at two and stop at four calls per batch with the existing deadline and zero retries. Lock the exact image hashes, reference collage, order, scenario, schema, thresholds and model before dispatch. Use an offline missing-image test to verify pre-dispatch denial separately. Do not reuse historical prompt variants as repeatability measurements.

Before dispatch, three independent reviewers record which slot and observable garment attributes each photograph contains, and separately score outfit coherence, fidelity, occasion fit, and accept/reject/unclear against the common rubric. Reviewers see neither candidate origin nor model answers. Keep a majority ground-truth label or mark the case ambiguous. Use actual image identity for missing-slot errors and compare cited attributes against reviewer consensus for recognition. Report schema failures, abstentions, false approvals/rejections against decisive labels, per-outfit score ranges, within-order versus cross-order differences, latency and tokens. Four outfits support debugging and calibration only. Repeat on cleared retail cases before making general claims.

## Release blockers

Cleared exact retail images and references, trustworthy variant-level offers, independently scored evidence, production-like visual ranking of complete decks and measured hosted stability are required for release. The free-tier trial is for internal evaluation.
