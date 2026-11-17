# Stylist 6 — 120B autonomous fashion designer native iPhone QA
Date: 2026-11-17. Pinned LIVE Render staging SHA 4cb6aa2e2289fb20e5817df96798a01c060463ab.
User-facing native app on booted AIWardrobe-S6-QA-20261115, Supabase JWT stage gateway v61.
Production Stylist5 was not changed. Rights-verified photo-FIRST product board remains OFF
until real permitted media coverage; its cache table currently has 0 authorized images.

## Correct New Clothing / Classic / Work
Pre-generation iOS screenshot verified Work selected, shop source selected.
Render log 17:04:12 UTC: 41,122 ms; six 120B AI source SKU compositions;
all six admitted; independent photographic critic attempted nine verdicts,
three approved; independent final photo VLM editor applied and chose two.
Both cards contained exactly FOUR real merchant items and seller links.

Outfit 01 Refined Tailoring: white dress shirt, dark wool suit jacket, black
pleated trousers and dark Belgian loafers. The jacket photo was BLANK on iPhone.
Outfit 02 Blazer & Trousers: pale blue poplin shirt, brown single-breasted
blazer, dark dress trousers and a Norwegian shoe. These are both almost the
same office tailoring architecture, NOT two impressive independent directions.
Mixed EUR/INR/UZS/USD source offers also impair intelligible price comparison.

The 30B predecessor 7fb8df2 in CORRECT Work returned no approved outfit
(stylist6_visual_quality_unmet). Its earlier 1/9 approval was in Classic/ANY:
the test accidentally deselected Work; it must NOT count as Work evidence.
Live SKU discovery changes across calls, so this is NOT a controlled A/B.

## Correct New Clothing / Classic / Date night
Pre-generation iOS screenshot verified Date night selected.
Render log 17:08:36 UTC: 31,335 ms; six 120B composed real SKU looks,
six admitted; six full-photo NVIDIA verdicts, THREE approved; final photo
editor applied, seeing actual JPEG images, chose TWO. All displayed looks
contained four real source items and links, no quality padding.

Outfit 01 Smart knit layering: black Wulf merino top, chocolate chunky
merino cardigan, slim light-grey trousers, dark Florence Chelsea boot.
Outfit 02 Cardigan & Chinos: light-grey AURI long-sleeve polo, ivory
cable-knit cardigan, beige chinos, brown Florence Chelsea boot.
The two looks share knit/polo+cardigan+trousers+Chelsea boot. A changed
color/polo does not create an independent fashion concept.

## Engineering and quality conclusion
The model generates valid source-backed 4-garment structures. Independent
PHOTO critic and whole-portfolio VLM editor are REAL and working. However
editor selection still returns near-clone outfit architectures despite being
able to select fewer than three. Fashion creative autonomy is NOT accepted.
The photographic critic approved only 3/9 Work and 3/6 Date.
The initial comparison on a matched synthetic garment SKU menu showed
30B NVIDIA ~2.61s and 120B ~1.98s for six structurally valid complete
outfits. Those numbers measure latency, not taste. 506 automated suites'
tests passed and the original Stylist5 production bundle was untouched.

## P0 next step
Finish the licensed source- and rights-verified product photo media pipeline
documented in docs/STYLIST6_LICENSED_IMAGE_CACHE.md and issue #17.
Use source ORIGINAL product photographs in the fashion designer BEFORE SKU
composition, not just descriptions, without request-time merchant CDN fanout.
Then give the independent portfolio editor a creative feedback output:
what silhouette/texture/color relationships are duplicated and WHAT new
direction is missing. If timing permits, run bounded SECOND AI composition
from unused verified source SKUs, re-judge actual product photos with the
independent model and never pad a weak third look. No regex product bans
or source-bypassing approvals.

Human blind panel still NOT run: target >=70% Stylist6 over Stylist5 in
>=30 held-out identical conditions with 3 real human judges. Production NO-GO.
