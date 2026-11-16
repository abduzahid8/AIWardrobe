# AIWardrobe Stylist 6 — bounded footwear photo intelligence (2026-11-16)

Product goal: better actual shoe-shape/material/sole compatibility, not another regex patch. See `docs/STYLIST6_NORTH_STAR.md`.

## Before

As read from Supabase immediately before this commit, `catalog_visual_jobs` has 5,461 succeeded, 1,557 unknown, 47 pending; **45 pending are active shoe SKUs**. Active shoe catalog count and photo intelligence coverage in the last full audit: 1,649 active shoes / 114 visually profiled. All active products already had garment-intelligence metadata, but title metadata is NOT a photographed visual profile.

## Bounded action

This branch contains `.github/workflows/stylist6-footwear-visual-backfill.yml` which triggers on its first push and supports manual dispatch. It reuses the existing scoped `catalog-visual-worker` control plane with its existing service-role authorization and model revision. It uses one CPU GitHub runner, limits processing to **at most 48 already queued** products, does NOT invoke `seed` (which would enqueue an uncontrolled part of the full catalog), and does not alter merchants, price, availability, customer tables or production Stylist 5 route. A completed item is durably acknowledged per image by the existing RPC. Invalid/downloading-failed imagery stays `unknown` or `failed` with provenance; never fabricated.

The stage uses the already-pinned SigLIP2 revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. Local CPU similarity is a weak image-derived signal; it is NOT an outfit-wide aesthetic acceptance or a substitute for human evaluation. Image rights and actual inventory remain separate gates.

**Important: the code/trigger existing does not prove that GitHub Actions completed inference.** Verify the workflow run and query `SELECT state, count(*) FROM public.catalog_visual_jobs WHERE product_id IN (SELECT id FROM public.shop_catalog WHERE category='shoes' AND is_active) GROUP BY 1` after completion before reporting coverage gains. Do not claim any shoe has a profile until it is written successfully to `public.catalog_visual_fashion` and its source hash matches.

## Evidence to report

- Exact GitHub workflow run ID/conclusion and model revision; counted `succeeded`, `unknown`, `pending`, `failed` shoe jobs before and after.
- Image confidence coverage and random sample review of leather dress shoes vs sneaker/hybrid soles. Treat uncertain descriptions as unknown, not confident.
- Current Stylist 6 cloud staging SHA, independent fashion visual approval, and bottlenecks; do not confuse this catalog process with synchronous 3-look inference.
