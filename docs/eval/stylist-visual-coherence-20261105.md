# Stylist Visual Coherence — real-catalog review manifest

Branch `feat/stylist-visual-coherence-20261105`. Baseline `e384864`.
No product images are committed; tiles below are source URLs for review.
Sheets (contact PNGs) and the sealed A/B key live in the ephemeral eval workspace,
not in the repo. Scores below are the author’s blind rubric scores (1-5 overall),
recorded before unblinding. No independent human reviewer was available.

## Snapshot provenance
- Built from `data/shop_catalog_*.json` (180 feeds, deterministic sorted-id order).
- Caps: top 150 / outerwear 150 / bottom 150 / shoes 120 (570 items).
- `is_active === false` items excluded; every kept item has an image URL.
- Excluded entire brand feeds with 100% dead image URLs: `alden-*` (HTTP 404),
  `a_p_c-*` (HTTP 403), `alfred_sargent-*` (DNS NXDOMAIN).
- Every kept shoe image PIL-verified (download + open + verify);
  tops/bottoms/outerwear resolve on media.i.allsaints.com, assets-cf.armani.com, media.arket.com.
- Same snapshot, scenarios and seeds for before/after; 8 contexts x 10 generations x 3 cards = 240 looks each.

## Metric summary (240 looks per version, local generation)
| metric | before (e384864) | after (this branch) |
|---|---|---|
| four-slot validity | 240/240 | 240/240 |
| judge pass | 182/240 | 186/240 |
| formal-shoe-with-athletic-bottom (new rule) | 4 | 0 |
| other fail categories | 54, count-identical | 54, count-identical |
| avg full-outfit coherence | 0.640 | 0.622 |
| avg palette dimension | 0.899 | 0.751 |
| avg footwear dimension | 0.747 | 0.749 |
| avg layer score | 0.637 | 0.637 |
| within-deck product overlap | 0.032 | 0.032 |
| exact repeated looks | 0 | 0 |
| semantic near-duplicates (>=0.8) | 2 | 2 |
| unique products | 246 | 246 |
| footwear subtypes | oxford,loafer,unknown,monk,derby,chelsea,boot | same |
| local generation latency (mean) | 35.0ms | 43.8ms |

Cross-judge check: the 240 before-looks re-judged with the NEW rules fail 58x
(54 pre-existing + 4 formal_footwear_with_athletic_bottom); the after run fails
exactly the same 54 pre-existing categories with 0 in the new class. 
Fail-category multisets are otherwise count-identical: zero regressions.

## Before/after pairs (blinded A/B, unblinded after scoring)
After won 3, tied 1, lost 0 on blind overall scores.
### Pair 1 — minimalist/date gen4 card2
- A row = **after**, B row = **before**
- BEFORE look (formal shoe + athletic bottom, fails new rule):
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Aspen Waffle-Knit Short-Sleeve Polo (id `allsaints-usm025ka-10066`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM025KA-10066.json?_i=AG>
  - [bottom] Bonded Stretch Cotton Sports Pants (id `armani_exchange-xm001763_af10818_uc001`) — <https://assets-cf.armani.com/image/upload/v1761678349/XM001763_AF10818_UC001_F_SS2026.jpg>
  - [shoes] METROPOLE LONDON Leather Derby Shoe - Black (id `ecco-525604-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/0db587ef-12ff-40f1-beb8-b08600eae0ca/525604-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>
- AFTER look (steered alternative, judge-clean):
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Aspen Waffle-Knit Short-Sleeve Polo (id `allsaints-usm025ka-10066`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM025KA-10066.json?_i=AG>
  - [bottom] Bonded Stretch Cotton Sports Pants (id `armani_exchange-xm001763_af10818_uc001`) — <https://assets-cf.armani.com/image/upload/v1761678349/XM001763_AF10818_UC001_F_SS2026.jpg>
  - [shoes] Outdoor boots (id `brunello-262MZURCCS890CUX81`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779263856/262MZURCCS890CUX81_F.jpg>

### Pair 2 — casual/travel gen9 card0
- A row = **after**, B row = **before**
- BEFORE look (formal shoe + athletic bottom, fails new rule):
  - [outerwear] Raven Slim-Fit Cotton Sweatshirt (id `allsaints-usm037nb-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM037NB-5.json?_i=AG>
  - [top] Raine Waffle-Knit Long-Sleeve T-Shirt (id `allsaints-usm020jf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM020JF-162.json?_i=AG>
  - [bottom] Bonded stretch cotton joggers with ribbed cuffs Icon (id `armani_exchange-xm000373_af10818_uc001`) — <https://assets-cf.armani.com/image/upload/v1776367191/XM000373_AF10818_UC001_F_FW2026.jpg>
  - [shoes] METROPOLE LONDON Leather Derby Shoe - Brown (id `ecco-525604-01482`) — <https://d2gdyivv6mfknv.flashedgecdn.net/8f5cd010-9ac7-4be0-9aa5-b08600eb12b0/525604-01482-o_eCom.png?v=2&format=auto&width=2560&quality=80>
- AFTER look (steered alternative, judge-clean):
  - [outerwear] Raven Slim-Fit Cotton Sweatshirt (id `allsaints-usm037nb-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM037NB-5.json?_i=AG>
  - [top] Raine Waffle-Knit Long-Sleeve T-Shirt (id `allsaints-usm020jf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM020JF-162.json?_i=AG>
  - [bottom] Bonded stretch cotton joggers with ribbed cuffs Icon (id `armani_exchange-xm000373_af10818_uc001`) — <https://assets-cf.armani.com/image/upload/v1776367191/XM000373_AF10818_UC001_F_FW2026.jpg>
  - [shoes] Lace-up boots (id `brunello-262MZUATMR674`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1790329033/262MZUATMR674C9843_F.jpg>

### Pair 3 — casual/weekend gen3 card0
- A row = **after**, B row = **before**
- BEFORE look (formal shoe + athletic bottom, fails new rule):
  - [outerwear] Chaiyo Suede Trucker Jacket (id `allsaints-usm054lf-10718`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM054LF-10718.json?_i=AG>
  - [top] Non Regular-Fit Graphic T-Shirt (id `allsaints-usm116pf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM116PF-162.json?_i=AG>
  - [bottom] Cotton twill drawstring trousers (id `armani_exchange-xm002742_af23202_u6229`) — <https://assets-cf.armani.com/image/upload/v1758566308/XM002742_AF23202_U6229_F_SS2026.jpg>
  - [shoes] HELSINKI 2 Leather Penny Loafer - Black (id `ecco-500354-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/0241db6b-5d41-4abc-b3d5-b36900b6810c/500354-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>
- AFTER look (steered alternative, judge-clean):
  - [outerwear] Chaiyo Suede Trucker Jacket (id `allsaints-usm054lf-10718`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM054LF-10718.json?_i=AG>
  - [top] Non Regular-Fit Graphic T-Shirt (id `allsaints-usm116pf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM116PF-162.json?_i=AG>
  - [bottom] Cotton twill drawstring trousers (id `armani_exchange-xm002742_af23202_u6229`) — <https://assets-cf.armani.com/image/upload/v1758566308/XM002742_AF23202_U6229_F_SS2026.jpg>
  - [shoes] Lace-up boots (id `brunello-262MZUVHRR676CQV46`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1780555338/262MZUVHRR676CQV46_F.jpg>

### Pair 4 — minimalist/casual gen1 card2
- A row = **before**, B row = **after**
- BEFORE look (formal shoe + athletic bottom, fails new rule):
  - [outerwear] Wulf Merino Wool Cardigan (id `allsaints-usm061kf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM061KF-5.json?_i=AG>
  - [top] Xander Oversized Cotton T-Shirt (id `allsaints-usm030jc-10261`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM030JC-10261.json?_i=AG>
  - [bottom] Cotton twill drawstring trousers (id `armani_exchange-xm002742_af23202_u6229`) — <https://assets-cf.armani.com/image/upload/v1758566308/XM002742_AF23202_U6229_F_SS2026.jpg>
  - [shoes] METROPOLE LONDON Leather Derby Shoe - Brown (id `ecco-525604-01482`) — <https://d2gdyivv6mfknv.flashedgecdn.net/8f5cd010-9ac7-4be0-9aa5-b08600eb12b0/525604-01482-o_eCom.png?v=2&format=auto&width=2560&quality=80>
- AFTER look (steered alternative, judge-clean):
  - [outerwear] Wulf Merino Wool Cardigan (id `allsaints-usm061kf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM061KF-5.json?_i=AG>
  - [top] Xander Oversized Cotton T-Shirt (id `allsaints-usm030jc-10261`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM030JC-10261.json?_i=AG>
  - [bottom] Cotton twill drawstring trousers (id `armani_exchange-xm002742_af23202_u6229`) — <https://assets-cf.armani.com/image/upload/v1758566308/XM002742_AF23202_U6229_F_SS2026.jpg>
  - [shoes] Metropole London Leather Moc-Toe Shoe - Black (id `ecco-525654-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/af763a4b-d432-4663-bc2e-b106008d2b57/525654-01001-o_eCom.png?v=1&format=auto&width=2560&quality=80>

## Singles (blinded version, 14 before + 14 after; mean overall after 3.86 vs before 3.68)
### semi_classic/business gen1 card2 — BEFORE (baseline) — judge: denim never belongs in classic or work decks
  - [outerwear] Milford Cotton Peacoat (id `allsaints-usm134of-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM134OF-162.json?_i=AG>
  - [top] Aspen Long-Sleeve Polo Shirt (id `allsaints-usm095kb-10919`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM095KB-10919.json?_i=AG>
  - [bottom] 5 pockets pant Skinny fit stretch cotton denim (id `armani_exchange-xm001512_af25551_mb010`) — <https://assets-cf.armani.com/image/upload/v1782933221/XM001512_AF25551_MB010_F_FW2026.jpg>
  - [shoes] METROPOLE VERONA Leather Loafer - Black (id `ecco-552264-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/0c115def-4237-40da-b5c0-b2e4009f3314/552264-01001-o_eCom.png?v=1&format=auto&width=2560&quality=80>

### semi_classic/formal gen0 card2 — BEFORE (baseline) — judge: item "Redact Relaxed-Fit Cotton Hoodie": formal_requires_structured_layer
  - [outerwear] Redact Relaxed-Fit Cotton Hoodie (id `allsaints-usm001nc-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM001NC-162.json?_i=AG>
  - [top] Ward Regular-Fit Tencel Shirt (id `allsaints-usm109sf-9688`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM109SF-9688.json?_i=AG>
  - [bottom] Blunt Slim-Fit Tailored Pants (id `allsaints-usm033tf-2732`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM033TF-2732.json?_i=AG>
  - [shoes] METROPOLE MILAN Leather Derby Shoe - Black (id `ecco-523364-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/53c5f018-868f-474e-9e54-b20b00e2423c/523364-01001-o_eCom.png?v=1&format=auto&width=2560&quality=80>

### semi_classic/business gen5 card2 — BEFORE (baseline) — judge: denim never belongs in classic or work decks
  - [outerwear] Milford Cotton Peacoat (id `allsaints-usm134of-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM134OF-162.json?_i=AG>
  - [top] Mode Merino Wool Long-Sleeve Polo Shirt (id `allsaints-usm085kb-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM085KB-5.json?_i=AG>
  - [bottom] Skinny fit 5 Pockets Pant washed cotton-blend denim (id `armani_exchange-xm001512_af28744_mb003`) — <https://assets-cf.armani.com/image/upload/v1782933147/XM001512_AF28744_MB003_F_FW2026.jpg>
  - [shoes] Chelsea boots (id `brunello-262MZUSOLE818C9832`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779171351/262MZUSOLE818C9832_F.jpg>

### semi_classic/formal gen4 card2 — BEFORE (baseline) — judge: item "Harlow Ramskull Raglan Sweatshirt": formal_rejects_casual_fabric
  - [outerwear] Harlow Ramskull Raglan Sweatshirt (id `allsaints-usm032nf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM032NF-162.json?_i=AG>
  - [top] Ward Regular-Fit Tencel Shirt (id `allsaints-usm109sf-9688`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM109SF-9688.json?_i=AG>
  - [bottom] Blunt Slim-Fit Tailored Pants (id `allsaints-usm033tf-2732`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM033TF-2732.json?_i=AG>
  - [shoes] HELSINKI 2 Leather Derby Shoe - Black (id `ecco-500164-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/df398aef-cb8c-4a12-9a8b-b08c0093593a/500164-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>

### semi_classic/formal gen6 card2 — AFTER (this branch) — judge: item "Redact Oversized Zip-Up Hoodie": formal_requires_structured_layer
  - [outerwear] Redact Oversized Zip-Up Hoodie (id `allsaints-usm025nd-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM025ND-162.json?_i=AG>
  - [top] Duke Regular-Fit Striped Shirt (id `allsaints-usm111sf-3969`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%205%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM111SF-3969.json?_i=AG>
  - [bottom] Blunt Slim-Fit Tailored Pants (id `allsaints-usm033tf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM033TF-5.json?_i=AG>
  - [shoes] METROPOLE LONDON Leather Derby Shoe - Black (id `ecco-525604-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/0db587ef-12ff-40f1-beb8-b08600eae0ca/525604-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>

### minimalist/work gen0 card1 — AFTER (this branch) — judge: classic short-sleeve shirts read cheap (Hawaiian camp excepted)
  - [outerwear] Milford Cotton Peacoat (id `allsaints-usm134of-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM134OF-162.json?_i=AG>
  - [top] Reform Short-Sleeve Polo Shirt (id `allsaints-usm075pb-4411`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM075PB-4411.json?_i=AG>
  - [bottom] slim fit 5 pockets pant in denim stretch (id `armani_exchange-xm000070_af25530_mb001`) — <https://assets-cf.armani.com/image/upload/v1782933516/XM000070_AF25530_MB001_F_FW2026.jpg>
  - [shoes] Outdoor boots (id `brunello-262MZURCCS890CUX81`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779263856/262MZURCCS890CUX81_F.jpg>

### semi_classic/formal gen8 card2 — AFTER (this branch) — judge: item "Liev Ribbed Crew-Neck Sweater": formal_requires_structured_layer
  - [outerwear] Liev Ribbed Crew-Neck Sweater (id `allsaints-usm062kf-4009`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM062KF-4009.json?_i=AG>
  - [top] Duke Regular-Fit Striped Shirt (id `allsaints-usm111sf-10175`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM111SF-10175.json?_i=AG>
  - [bottom] Blunt Slim-Fit Tailored Pants (id `allsaints-usm033tf-2732`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM033TF-2732.json?_i=AG>
  - [shoes] Chelsea boots (id `brunello-262MZUBMLE818C9855`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1781677945/262MZUBMLE818C9855_F.jpg>

### semi_classic/formal gen4 card1 — AFTER (this branch) — judge: item "Miro Perforated Leather Bomber Jacket": formal_requires_structured_layer
  - [outerwear] Miro Perforated Leather Bomber Jacket (id `allsaints-usm045lf-203`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM045LF-203.json?_i=AG>
  - [top] Duke Regular-Fit Striped Shirt (id `allsaints-usm111sf-3969`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%205%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM111SF-3969.json?_i=AG>
  - [bottom] Blunt Slim-Fit Tailored Pants (id `allsaints-usm033tf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM033TF-5.json?_i=AG>
  - [shoes] HELSINKI 2 Leather Derby Shoe - Black (id `ecco-500164-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/df398aef-cb8c-4a12-9a8b-b08c0093593a/500164-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>

### minimalist/work gen0 card0 — AFTER (this branch) — judge: pass
  - [outerwear] Coltra Double-Breasted Wool Trench Coat (id `allsaints-usm133of-10715`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM133OF-10715.json?_i=AG>
  - [top] Ward Regular-Fit Tencel Shirt (id `allsaints-usm109sf-9722`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM109SF-9722.json?_i=AG>
  - [bottom] Cotton Trousers - Black (id `arket-1268001004`) — <https://media.arket.com/assets/006/48/07/48077e3669d2512a9f480ef06dca9c3ec3507061_xxl-1.jpg>
  - [shoes] METROPOLE LONDON Leather Derby Shoe - Black (id `ecco-525604-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/0db587ef-12ff-40f1-beb8-b08600eae0ca/525604-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>

### casual/travel gen5 card1 — BEFORE (baseline) — judge: pass
  - [outerwear] Aspen Waffle-Knit Wool-Blend Sweater (id `allsaints-usm089kb-4093`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM089KB-4093.json?_i=AG>
  - [top] Emblems Slim-Fit Graphic T-Shirt (id `allsaints-usm083pf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM083PF-162.json?_i=AG>
  - [bottom] Interlock trousers (id `armani_exchange-xm002374_af10365_uc001`) — <https://assets-cf.armani.com/image/upload/v1758651044/XM002374_AF10365_UC001_F_SS2026.jpg>
  - [shoes] Outdoor boots (id `brunello-262MZUANRW668CTR28`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1783435841/262MZUANRW668CTR28_F.jpg>

### minimalist/casual gen2 card2 — BEFORE (baseline) — judge: pass
  - [outerwear] Full-zip vest in stretch cotton with a high collar Essential (id `armani_exchange-xm001565_af10818_ub101`) — <https://assets-cf.armani.com/image/upload/v1776367865/XM001565_AF10818_UB101_F_FW2026.jpg>
  - [top] Brace Short-Sleeve Cotton T-Shirt (id `allsaints-usm030jb-4411`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM030JB-4411.json?_i=AG>
  - [bottom] Loose-Straight Leg Jeans (id `allsaints-usm070ef-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM070EF-162.json?_i=AG>
  - [shoes] S LITE HYBRID Leather Derby Shoe - Black (id `ecco-520304-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/86da56d1-5587-4932-a1a1-b06f00dbce4b/520304-01001-o_eCom.png?v=3&format=auto&width=2560&quality=80>

### semi_classic/business gen3 card1 — BEFORE (baseline) — judge: pass
  - [outerwear] Milford Cotton Peacoat (id `allsaints-usm134of-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM134OF-162.json?_i=AG>
  - [top] Mode Merino Wool Long-Sleeve Polo Shirt (id `allsaints-usm085kb-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM085KB-5.json?_i=AG>
  - [bottom] York Straight-Fit Twill Pants (id `allsaints-usm035tf-2732`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM035TF-2732.json?_i=AG>
  - [shoes] Outdoor boots (id `brunello-262MZURCCS890CUX81`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779263856/262MZURCCS890CUX81_F.jpg>

### casual/weekend gen0 card2 — AFTER (this branch) — judge: pass
  - [outerwear] Harlow Ramskull Raglan Sweatshirt (id `allsaints-usm032nf-4192`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM032NF-4192.json?_i=AG>
  - [top] Archer Oversized Viscose Shirt (id `allsaints-usm048sg-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM048SG-5.json?_i=AG>
  - [bottom] Comfort slim five-pocket jeans in fleece denim Essential (id `armani_exchange-xm000049_af14370_mc002`) — <https://assets-cf.armani.com/image/upload/v1776367373/XM000049_AF14370_MC002_F_FW2026.jpg>
  - [shoes] Chelsea boots (id `brunello-262MZUSOLE818C9832`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779171351/262MZUSOLE818C9832_F.jpg>

### minimalist/date gen9 card1 — BEFORE (baseline) — judge: pass
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Ward Regular-Fit Tencel Shirt (id `allsaints-usm109sf-9722`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM109SF-9722.json?_i=AG>
  - [bottom] Blunt Slim-Fit Tailored Pants (id `allsaints-usm033tf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM033TF-5.json?_i=AG>
  - [shoes] HELSINKI 2 Leather Derby Shoe - Black (id `ecco-500164-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/df398aef-cb8c-4a12-9a8b-b08c0093593a/500164-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>

### semi_classic/business gen2 card1 — BEFORE (baseline) — judge: pass
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Statten Long-Sleeve Ramskull Polo Shirt (id `allsaints-usm038ka-6936`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM038KA-6936.json?_i=AG>
  - [bottom] Cotton trousers (id `armani_exchange-xm002415_af22642_ub101`) — <https://assets-cf.armani.com/image/upload/v1775589004/XM002415_AF22642_UB101_F_SS2026.jpg>
  - [shoes] Lace-up boots (id `brunello-262MZUVHRR676CQV46`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1780555338/262MZUVHRR676CQV46_F.jpg>

### casual/travel gen1 card0 — BEFORE (baseline) — judge: pass
  - [outerwear] Fine-Weave Mercerized Cotton Sweater (id `armani_exchange-xm001976_af11966_uc001`) — <https://assets-cf.armani.com/image/upload/v1762369574/XM001976_AF11966_UC001_F_SS2026.jpg>
  - [top] Hale Band-Fit Regular T-Shirt (id `allsaints-usm015jf-2426`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM015JF-2426.json?_i=AG>
  - [bottom] Classic Comfort Cotton Twill Trousers (id `armani_exchange-xm002089_af11882_u0009`) — <https://assets-cf.armani.com/image/upload/v1773927675/XM002089_AF11882_U0009_F_SS2026.jpg>
  - [shoes] Lace-up boots (id `brunello-262MZUATMR674`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1790329033/262MZUATMR674C9843_F.jpg>

### minimalist/date gen8 card1 — AFTER (this branch) — judge: pass
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Raine Waffle-Knit Long-Sleeve T-Shirt (id `allsaints-usm020jf-10728`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM020JF-10728.json?_i=AG>
  - [bottom] Haydon Slim-Fit Cotton Chinos (id `allsaints-usm049tf-5881`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049TF-5881.json?_i=AG>
  - [shoes] Lace-up boots (id `brunello-262MZUSXRR676`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1781764351/262MZUSXRR676CVK68_F.jpg>

### casual/travel gen4 card2 — AFTER (this branch) — judge: pass
  - [outerwear] Technical fabric jacket (id `armani_exchange-xm002202_af21302_uc001`) — <https://assets-cf.armani.com/image/upload/v1758565880/XM002202_AF21302_UC001_F_SS2026.jpg>
  - [top] Aspen Long-Sleeve Polo Shirt (id `allsaints-usm095kb-10725`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM095KB-10725.json?_i=AG>
  - [bottom] York Straight-Fit Twill Pants (id `allsaints-usm035tf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM035TF-5.json?_i=AG>
  - [shoes] S LITE HYBRID Leather Derby Shoe - Black (id `ecco-520304-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/86da56d1-5587-4932-a1a1-b06f00dbce4b/520304-01001-o_eCom.png?v=3&format=auto&width=2560&quality=80>

### minimalist/date gen4 card0 — BEFORE (baseline) — judge: pass
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Statten Long-Sleeve Ramskull Polo Shirt (id `allsaints-usm038ka-6936`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM038KA-6936.json?_i=AG>
  - [bottom] Moor Straight-Fit Melange Pants (id `allsaints-usm087tf-3860`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM087TF-3860.json?_i=AG>
  - [shoes] Chelsea boots (id `brunello-262MZUNSMQ677`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1784800980/262MZUNSMQ677COL80_F.jpg>

### minimalist/casual gen7 card1 — BEFORE (baseline) — judge: pass
  - [outerwear] Galactic Jacquard-Knit Alpaca-Blend Cardigan (id `allsaints-usm083kf-10784`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM083KF-10784.json?_i=AG>
  - [top] Brace Cotton T-Shirt (id `allsaints-usm030jb-3595`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM030JB-3595.json?_i=AG>
  - [bottom] Faded fleece denim jeans with a comfort slim fit (id `armani_exchange-xm000049_af14369_m8013`) — <https://assets-cf.armani.com/image/upload/v1776367458/XM000049_AF14369_M8013_F_FW2026.jpg>
  - [shoes] Outdoor boots (id `brunello-262MZURCCS890CUX81`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779263856/262MZURCCS890CUX81_F.jpg>

### minimalist/casual gen8 card2 — AFTER (this branch) — judge: pass
  - [outerwear] Full zip jacket with pockets (id `armani_exchange-xm001579_af16926_uc001`) — <https://assets-cf.armani.com/image/upload/v1758566485/XM001579_AF16926_UC001_F_SS2026.jpg>
  - [top] Aspen Long-Sleeve Polo Shirt (id `allsaints-usm095kb-10725`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM095KB-10725.json?_i=AG>
  - [bottom] Cotton trousers (id `armani_exchange-xm002415_af22642_ub101`) — <https://assets-cf.armani.com/image/upload/v1775589004/XM002415_AF22642_UB101_F_SS2026.jpg>
  - [shoes] S LITE HYBRID Leather Derby Shoe - Black (id `ecco-520304-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/86da56d1-5587-4932-a1a1-b06f00dbce4b/520304-01001-o_eCom.png?v=3&format=auto&width=2560&quality=80>

### minimalist/date gen3 card0 — AFTER (this branch) — judge: pass
  - [outerwear] Milford Cotton Peacoat (id `allsaints-usm134of-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM134OF-162.json?_i=AG>
  - [top] Statten Long-Sleeve Ramskull Polo Shirt (id `allsaints-usm038ka-6936`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM038KA-6936.json?_i=AG>
  - [bottom] Tower Slim-Fit Cotton Chinos (id `allsaints-usm039tf-8834`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM039TF-8834.json?_i=AG>
  - [shoes] METROPOLE MILAN Leather Derby Shoe - Black (id `ecco-523364-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/53c5f018-868f-474e-9e54-b20b00e2423c/523364-01001-o_eCom.png?v=1&format=auto&width=2560&quality=80>

### minimalist/casual gen7 card0 — BEFORE (baseline) — judge: pass
  - [outerwear] Enzo Cashmere Sweater (id `allsaints-usm093kf-3860`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM093KF-3860.json?_i=AG>
  - [top] Burnout Oversized Cotton T-Shirt (id `allsaints-usm064pe-7266`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM064PE-7266.json?_i=AG>
  - [bottom] Skinny-fit five-pocket pants in washed comfort cotton denim (id `armani_exchange-xm001512_af28150_mb002`) — <https://assets-cf.armani.com/image/upload/v1774533287/XM001512_AF28150_MB002_F_FW2026.jpg>
  - [shoes] Lace-up boots (id `brunello-262MZUSXRR676`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1781764351/262MZUSXRR676CVK68_F.jpg>

### semi_classic/business gen2 card2 — AFTER (this branch) — judge: pass
  - [outerwear] Coltra Double-Breasted Wool Trench Coat (id `allsaints-usm133of-10715`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM133OF-10715.json?_i=AG>
  - [top] Aubrey Short-Sleeve Ramskull Polo (id `allsaints-usm020ka-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM020KA-5.json?_i=AG>
  - [bottom] Interlock trousers (id `armani_exchange-xm002374_af10365_uc001`) — <https://assets-cf.armani.com/image/upload/v1758651044/XM002374_AF10365_UC001_F_SS2026.jpg>
  - [shoes] HELSINKI 2 Leather Derby Shoe - Black (id `ecco-500164-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/df398aef-cb8c-4a12-9a8b-b08c0093593a/500164-01001-o_eCom.png?v=2&format=auto&width=2560&quality=80>

### minimalist/casual gen6 card1 — AFTER (this branch) — judge: pass
  - [outerwear] Wulf Merino Wool Henley Sweater (id `allsaints-usm048kf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM048KF-5.json?_i=AG>
  - [top] Figure Regular-Fit Cotton T-Shirt (id `allsaints-usm025jb-451`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM025JB-451.json?_i=AG>
  - [bottom] Comfort slim-fit jeans in faded cotton denim Essential (id `armani_exchange-xm000049_af14266_mb001`) — <https://assets-cf.armani.com/image/upload/v1776368157/XM000049_AF14266_MB001_F_FW2026.jpg>
  - [shoes] METROPOLE VERONA Leather Loafer - Black (id `ecco-552264-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/0c115def-4237-40da-b5c0-b2e4009f3314/552264-01001-o_eCom.png?v=1&format=auto&width=2560&quality=80>

### minimalist/date gen6 card0 — AFTER (this branch) — judge: pass
  - [outerwear] Tilson Leather Peacoat (id `allsaints-usm049lf-5`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM049LF-5.json?_i=AG>
  - [top] Raine Waffle-Knit Long-Sleeve T-Shirt (id `allsaints-usm020jf-10728`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM020JF-10728.json?_i=AG>
  - [bottom] slim fit 5 pockets pant in denim stretch (id `armani_exchange-xm000070_af25530_mb001`) — <https://assets-cf.armani.com/image/upload/v1782933516/XM000070_AF25530_MB001_F_FW2026.jpg>
  - [shoes] Lace-up boots (id `brunello-262MZUVHRR676CQV46`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1780555338/262MZUVHRR676CQV46_F.jpg>

### casual/weekend gen4 card2 — BEFORE (baseline) — judge: pass
  - [outerwear] Aspen Waffle-Knit Wool-Blend Sweater (id `allsaints-usm089kb-10725`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM089KB-10725.json?_i=AG>
  - [top] Maddox Relaxed-Fit Cotton T-Shirt (id `allsaints-usm016jf-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM016JF-162.json?_i=AG>
  - [bottom] 5 pockets pant Skinny fit stretch cotton denim (id `armani_exchange-xm001512_af25551_mb010`) — <https://assets-cf.armani.com/image/upload/v1782933221/XM001512_AF25551_MB010_F_FW2026.jpg>
  - [shoes] Outdoor boots (id `brunello-262MZURCCS890CUX81`) — <https://media.brunellocucinelli.com/brunellocucinelli-dynamic/image/upload/v1779263856/262MZURCCS890CUX81_F.jpg>

### casual/weekend gen8 card0 — AFTER (this branch) — judge: pass
  - [outerwear] Senner Wool-Blend Striped Sweater (id `allsaints-usm076kf-7`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM076KF-7.json?_i=AG>
  - [top] Brace Short-Sleeve Cotton T-Shirt (id `allsaints-usm030jb-10732`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM030JB-10732.json?_i=AG>
  - [bottom] Slim-Straight Leg Jeans (id `allsaints-usm065ef-162`) — <https://media.i.allsaints.com/image/list/fn_select:jq:first%28.%5B%5D%7Cif%20has%28%22metadata%22%29%20then%20select%28any%28.metadata%5B%5D%3B%20.external_id%20%3D%3D%20%22sfcc-gallery-position%22%20and%20.value%20%3D%3D%201%29%29%20else%20empty%20end%29/f_auto,q_auto,dpr_auto,w_960,h_1200,c_fit/USM065EF-162.json?_i=AG>
  - [shoes] Metropole London Leather Moc-Toe Shoe - Black (id `ecco-525654-01001`) — <https://d2gdyivv6mfknv.flashedgecdn.net/af763a4b-d432-4663-bc2e-b106008d2b57/525654-01001-o_eCom.png?v=1&format=auto&width=2560&quality=80>

## Blind rubric notes (abridged, full text in eval workspace review-notes.md)
- Formal/work/business judge rejections (hoodie, bomber, sweatshirt-as-layer, denim, camp polo)
  agree with the eye in 7/8 sampled cases; one knit-layer formal case the eye rates kinder (sheet 008).
- Residual weakness the text evidence cannot see: derby + striped interlock-knit track pants
  (sheet 010) — photo-visible side stripe, no text cue ("interlock trousers" not athletic).
- Residual weakness: rugged lace-up boots pass in business contexts (sheets 007, 025) —
  product names ("Lace-up boots", "Outdoor boots") carry no lug-sole evidence; only the photo does.
- No black/white/beige monopoly observed (navy, brown, cream, olive, burgundy, grey, pink, patterns).

## Supply / data-quality limitations
- Entire checked-in feeds carry fabricated or dead image URLs (alden, a_p_c, alfred_sargent,
  plus ~100 more dead hosts across shoe catalogs); only ~13 image hosts verified live.
- Narrow formal/business pools force compromises (e.g. sweatpants in date looks);
  the engine prefers fewer suitable alternatives over fabricated diversity (no placeholders emitted).
