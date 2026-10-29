# Same-selection Personal Stylist rotation

## Source

- Branch: `stylist-four-slot-recovery`
- Starting HEAD: `d9b04caf4fb184e8e12248df3efc6796c29b579f`
- Implementation HEAD: `7db1a09` (delivery HEAD also includes this evidence report)
- Edge: v158, active. All 16 downloaded TypeScript files match checkout byte-for-byte.
- Metro: port 8082, `/Users/v.zahid/AIWardrobe`; only one listener.

## Root cause

Reproduced before editing using the real Expo Go screen: New clothing, Old Money, Date Night, no anchor. G2 repeated Chairman and Saint shoes from G1; G3 repeated Saint and Chairman from G2. Exact final displayed decks are in `stylist-rotation-evidence/reproduced-displayed-decks.json`.

The history was a user-wide flattened set. Once the shoe slot became exhausted, the Edge restored historical shoes and ranked them without knowing which belonged to the immediately previous deck. Older alternatives were not preferred. This was not demonstrated to be a stale AsyncStorage-read race: the existing flattened-history read already waited for pending writes.

The patch retains bounded deck records under a stable user/source/style/occasion scope. Each tap has its own request ID. Final accepted decks update in-memory history before another generation can start; persistence runs asynchronously. Previous keys, older keys and deck order travel unchanged to Edge. Selection prioritizes fresh products, then products outside the previous deck, then oldest last-use. Historical catalog trimming uses the same priority. A final backend overlap audit and a client guard prevent repair from adding overlap. Existing quality, weather, budget, provider, classification and 3-by-4 contracts are unchanged.

An initial v157 test hit Edge CPU limits because history scoring was repeated for every candidate combination. v158 caches product classification once per request; its real five-generation flow passed.

## Rotation scope and request IDs

| Generation | Scope | Request ID | Previous keys | Older keys |
|---|---|---|---:|---:|
| G1 | `db0a1cee-6161-45d8-a4c9-d6b195f791a1|shop|old_money|date` | `mvtj1yrx-1` | 0 | 0 |
| G2 | `db0a1cee-6161-45d8-a4c9-d6b195f791a1|shop|old_money|date` | `mvtj2lbm-2` | 36 | 0 |
| G3 | `db0a1cee-6161-45d8-a4c9-d6b195f791a1|shop|old_money|date` | `mvtj34fs-3` | 36 | 36 |
| G4 | `db0a1cee-6161-45d8-a4c9-d6b195f791a1|shop|old_money|date` | `mvtj3kaj-4` | 36 | 72 |
| G5 | `db0a1cee-6161-45d8-a4c9-d6b195f791a1|shop|old_money|date` | `mvtj3yk9-5` | 36 | 105 |

Scopes identical: **yes**. Request IDs unique: **yes**. G2 receives G1 as previous; G3 receives G2 as previous and G1 as older history.

## Exact final displayed products

Captured from the screen’s final accepted `shownOutfits` at `recordDisplayedDeck`, after repair, quality, budget and deck selection. The Maestro flow taps Generate five times without changing any selection. Weather remains unavailable and no explicit budget or anchor is added. The JSON evidence includes physical identity keys for every product.

### Generation 1

| Outfit | Outerwear | Top | Bottom | Shoes |
|---|---|---|---|---|
| 1 | Hawke Wool Crew-Neck Sweater (`shop_allsaints-usm036kf-3860`) | Court Regular-Fit Double-Faced Shirt (`shop_allsaints-usm128sf-2824`) | Classic Comfort Cotton Twill Trousers (`shop_armani_exchange-xm002089_af11882_u0009`) | Saint / Cognac (`shop_thursday-8086847258714`) |
| 2 | Wulf Merino Wool Cardigan (`shop_allsaints-usm061kf-3860`) | Gleason Slim-Fit Denim Shirt (`shop_allsaints-usm065sf-10131`) | Ware Straight-Fit Double-Pleat Pants (`shop_allsaints-usm037tf-4074`) | Statesman / Brandy (`shop_thursday-7887606677594`) |
| 3 | Marko Wool Jacket (`shop_allsaints-usm017kd-5`) | Aspen Long-Sleeve Polo Shirt (`shop_allsaints-usm095kb-10725`) | Oxford Lyocell Trousers (`shop_armani_exchange-xm002298_af22715_uc001`) | Chairman / Black (`shop_thursday-6859648467034`) |

### Generation 2

| Outfit | Outerwear | Top | Bottom | Shoes |
|---|---|---|---|---|
| 1 | Chaiyo Suede Trucker Jacket (`shop_allsaints-usm054lf-10262`) | Mercerized Cotton Regular Fit Polo (`shop_armani_exchange-6dzfhl_zj2zz_17ab`) | Tower Slim-Fit Cotton Chinos (`shop_allsaints-usm039tf-8834`) | Journeyman / Black (`shop_thursday-7719429898330`) |
| 2 | Enzo Funnel-Neck Cashmere Sweater (`shop_allsaints-usm092kf-5`) | Brook Regular-Fit Check Western Shirt (`shop_allsaints-usm130sf-668`) | Cotton Trousers - Black (`shop_arket-1268001004`) | Executive / Black (`shop_thursday-253390159881`) |
| 3 | Mode Merino Wool Ramskull Sweater (`shop_allsaints-usm002kz-10657`) | Judd Regular-Fit Striped Shirt (`shop_allsaints-usm134sf-10186`) | Blunt Slim-Fit Tailored Pants (`shop_allsaints-usm033tf-2732`) | President StormKing® / Bull Brown (`shop_thursday-8140117377114`) |

Physical overlap with earlier displayed decks:

| Earlier deck | Outerwear | Top | Bottom | Shoes |
|---|---:|---:|---:|---:|
| G1 | 0 | 0 | 0 | 0 |

### Generation 3

| Outfit | Outerwear | Top | Bottom | Shoes |
|---|---|---|---|---|
| 1 | Enzo Cashmere Sweater (`shop_allsaints-usm093kf-3860`) | Gleason Denim Shirt (`shop_allsaints-usm159sf-2160`) | Aston Straight-Fit Double-Pleat Chinos (`shop_allsaints-usm050tf-197`) | Duke / Black Coffee (`shop_thursday-15456871514202`) |
| 2 | Stamped Cotton Cardigan (`shop_allsaints-usm120kf-5`) | Duke Regular-Fit Striped Shirt (`shop_allsaints-usm111sf-10175`) | Interlock trousers (`shop_armani_exchange-xm002374_af10365_uc001`) | Statesman / Brandy (`shop_thursday-7887606677594`) |
| 3 | Carlow Slim-Fit Wool-Blend Jacket (`shop_allsaints-usm079of-3860`) | Statten Long-Sleeve Ramskull Polo Shirt (`shop_allsaints-usm038ka-6936`) | Moor Straight-Fit Melange Pants (`shop_allsaints-usm087tf-3860`) | Renegade / Black (`shop_thursday-6794356228186`) |

Physical overlap with earlier displayed decks:

| Earlier deck | Outerwear | Top | Bottom | Shoes |
|---|---:|---:|---:|---:|
| G1 | 0 | 0 | 0 | 1 |
| G2 | 0 | 0 | 0 | 0 |

### Generation 4

| Outfit | Outerwear | Top | Bottom | Shoes |
|---|---|---|---|---|
| 1 | Aspen Waffle-Knit Wool-Blend Sweater (`shop_allsaints-usm089kb-10725`) | Bond Regular-Fit Tencel Shirt (`shop_allsaints-usm108sf-2732`) | Light Aqua Linen Seaside Trousers (`shop_aurelien-10046869209418`) | President StormKing® / Bull Brown (`shop_thursday-8140117377114`) |
| 2 | Coltra Double-Breasted Wool Trench Coat (`shop_allsaints-usm133of-10715`) | Mode Merino Wool Long-Sleeve Polo Shirt (`shop_allsaints-usm085kb-5`) | Light Beige Pleated Metropolitan Trousers (`shop_aurelien-15501547110730`) | Chairman / Black (`shop_thursday-6859648467034`) |
| 3 | Single breasted jacket in a wool and viscose blend Icon (`shop_armani_exchange-xm000345_af10897_uc001`) | Foley Regular-Fit Military Shirt (`shop_allsaints-usm124sf-10175`) | Navy Cotton Wool Regent Trousers (`shop_aurelien-16011499340106`) | Saint / Cognac (`shop_thursday-8086847258714`) |

Physical overlap with earlier displayed decks:

| Earlier deck | Outerwear | Top | Bottom | Shoes |
|---|---:|---:|---:|---:|
| G1 | 0 | 0 | 0 | 2 |
| G2 | 0 | 0 | 0 | 1 |
| G3 | 0 | 0 | 0 | 0 |

### Generation 5

| Outfit | Outerwear | Top | Bottom | Shoes |
|---|---|---|---|---|
| 1 | Arlo Wool Crew-Neck Breton Sweater (`shop_allsaints-usm064kf-2153`) | Button-up shirt in linen with classic collar (`shop_armani_exchange-xm001221_af14619_u0001`) | Beige Linen Seaside Trousers (`shop_aurelien-7649901707481`) | Journeyman / Black (`shop_thursday-7719429898330`) |
| 2 | Miller Leather Jacket (`shop_allsaints-usm009la-5`) | Bower Regular-Fit Check Shirt (`shop_allsaints-usm132sf-4938`) | Navy Pleated Metropolitan Trousers (`shop_aurelien-10228846690634`) | Statesman / Brandy (`shop_thursday-7887606677594`) |
| 3 | Cora slim fit leather jacket (`shop_allsaints-usm034lf-4732`) | Linen and cotton knitted polo shirt (`shop_armani_exchange-xm002151_af22062_m1297`) | Ecru Linen Seaside Trousers (`shop_aurelien-10046869373258`) | Executive / Black (`shop_thursday-253390159881`) |

Physical overlap with earlier displayed decks:

| Earlier deck | Outerwear | Top | Bottom | Shoes |
|---|---:|---:|---:|---:|
| G1 | 0 | 0 | 0 | 1 |
| G2 | 0 | 0 | 0 | 2 |
| G3 | 0 | 0 | 0 | 1 |
| G4 | 0 | 0 | 0 | 0 |

## Thin inventory

- Live Old Money / Date Night inventory supplies eight admitted physical shoe products across these generations. G1 and G2 use six distinct shoes; G3 uses the two remaining fresh shoes and one older shoe. No consecutive shoe overlap occurs. Older shoe reuse starts only as fresh inventory runs out.
- Controlled hosted test: five valid shoes and nine products in each other slot. G2 uses both remaining fresh shoes and one historical shoe. G3 uses the two shoes absent from G2 and only one unavoidable previous shoe. Other slots stay fresh. Whole-outfit repeat: **no**.
- A separate hosted rich-inventory test runs five same-intent generations using 15 physical products per slot and verifies zero overlap against all earlier generations. The live Old Money catalog is thinner in shoes; the report does not claim impossible zero older-history overlap there.

## Validation

| Requirement | Result |
|---|---|
| Exactly three visible cards each generation | PASS |
| Exactly four items per card | PASS |
| Existing quality and four-slot checks | PASS |
| Consecutive physical overlap per slot | 0 / 0 / 0 / 0 for all four pairs |
| Shoes rotate | PASS |
| Same selections across five real Generate presses | PASS |
| Stable scope / unique request IDs | PASS / PASS |
| Five-generation rich inventory fixture | PASS |
| Thin-shoe inventory fixture | PASS |
| Regression tests | 130 suites, 1,602 tests PASS |
| App TypeScript / Edge Deno checks | PASS / PASS |

Evidence limitations: quality is checked against the unchanged recovery rules and catalog metadata. The repeat-flow screenshot filenames are overwritten each iteration, so included PNGs show the last generation only; the JSON capture contains all five final displayed decks. Expo Go required a restart after its existing native-module reload error; the complete final flow then passed. No history was cleared on generation errors or retries.
