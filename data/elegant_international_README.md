# International elegant menswear

The expanded selection contains 246 products from 68 brands, combining the
existing visually reviewed luxury selection with tailored clothing and smart
casual pieces from the project's affordable, midrange, and premium feeds.

The image rule is enforced during generation: the reviewed luxury catalog must
carry its visual no-mannequin review, and every supplemental item must come
from a source feed explicitly labelled product-only, plain-background, or
no-model. It is a curated selection, not an exhaustive list of every menswear
store worldwide.

Additional product-only feeds from the companion `website-downloader`
workspace can be synced with:

```sh
npm run shop:sync-product-only-feeds
```

The sync validates every row has an HTTPS product and image URL plus explicit
product-only provenance before copying BOSS, Calvin Klein, GANT, Massimo
Dutti, Tommy Hilfiger, and Zara feeds into `data/`.

Generate the catalog and provenance manifest:

```sh
node scripts/expand-elegant-menswear.js
```

Validate without changing files:

```sh
node scripts/expand-elegant-menswear.js --check
```

Publish to the live catalog with `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` available in the environment:

```sh
node scripts/expand-elegant-menswear.js --apply
```

The app reads Supabase `shop_catalog`; generating local JSON alone does not
change what users see. The importer matches product URLs to preserve live IDs,
then upserts in batches. It does not delete other catalog products. A failed
batch can leave earlier batches imported; rerunning resumes via upserts.

Product prices, currencies, links, and image URLs are retained from existing
project feeds. They have not all been freshly verified. Refresh them before
relying on current prices or stock. Brand budget tiers are editorial positioning;
they do not imply currency conversion, local availability, or shipping coverage.
The manifest records the source feed and tier for each selected product.
