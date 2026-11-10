# Stylist 6 — Live Supabase catalog audit, 2026-11-10

**Read-only verified from live Supabase connector in ChatGPT**  
Project: `fyqpifmrsftsfqibhwhy`  
No database mutations, migrations, Storage uploads, paid model calls or production deployments performed.

## Current live catalog inventory

Direct SQL on `public.shop_catalog`:

| Metric | Verified result |
|---|---:|
| Total catalog rows | 7,912 |
| Active rows with positive price and HTTPS image/store URLs | 7,020 |
| Distinct base IDs among active rows (remove `-vN` suffix) | 7,017 |
| Active UZS-quoted SKUs | 3,006 |
| Active USD-quoted SKUs | 2,037 |
| Active EUR-quoted SKUs | 1,401 |
| Active GBP-quoted SKUs | 576 |

**Important:** HTTPS-looking links are not equivalent to reachable photos or an in-stock offer. The pilot separately measured substantial CDN 403s and dead imagery. `is_active` is catalog eligibility, not a live stock guarantee.

The saved offline pilot uses `data/shop_catalog_*.json` containing 40,989 rows and 38,388 admitted SKU rows, with only 8,659 distinct base IDs overall reported by Codex. **Do not compare variant-inflated 38,388 with live 7,020 as if they were unique products.** Exact SKU intersection, content drift and price changes remain unmeasured; agent should run ID-level comparison after a read-only live export.

## UZS catalogue — main observed subtype counts / minimum prices

These are *minimum recorded product prices per subtype* among active UZS rows joined to `catalog_garment_intelligence`, not outfit validation results.

| Wearing role and subtype | Active SKUs | Minimum UZS |
|---|---:|---:|
| top / t-shirt | 317 | 219,900 |
| top / polo | 251 | 238,141 |
| top / shirt | 364 | 349,900 |
| top / fine-knit | 8 | 1,186,196 |
| top / unknown | 161 | 279,900 |
| outerwear / jacket | 232 | 258,000 |
| outerwear / blazer | 119 | 849,900 |
| outerwear / cardigan | 66 | 635,043 |
| outerwear / pullover | 268 | 396,902 |
| bottom / trouser | 287 | 396,902 |
| bottom / tailored-trouser | 113 | 429,900 |
| shoes / loafer | 23 | 1,299,900 |
| shoes / minimal-sneaker | 11 | 649,900 |
| shoes / sneaker | 21 | 549,900 |

**Budget consequence:** A 1,500,000 UZS *four-piece* outfit with `shoes/loafer` cannot meet the cap using the UZS-quoted active catalog, since **1,299,900 + 219,900 = 1,519,800 UZS** even for the cheapest `shoes/loafer` and cheapest `top/t-shirt` alone — before any trousers or layer. The minimum `top/fine-knit` plus loafer is still higher. This is an inventory/price constraint under those assumptions, not proof that *all* fashionable outfits below 1.5M UZS are impossible. A different footwear subtype or wardrobe-owned item can change feasibility; actual context and hard gates must still be applied.

An absolute theoretical four-slot lower sum using the cheapest *category-level* UZS top+t Jacket+bottom+generic sneaker = **1,424,702 UZS** (219,900+258,000+396,902+549,900), but says **nothing** about visual compatibility, sizes, inventory, user preference, season or image accessibility. Never present this as a valid generated outfit.

## Source queries (executed read-only)

```sql
select count(*) as all_catalog,
  count(*) filter(where is_active) as active_catalog,
  count(*) filter(where is_active and price > 0
    and image_url like 'https://%' and source_url like 'https://%') as usable_active
from public.shop_catalog;

select currency, count(*) as active_count,
  count(distinct regexp_replace(id, '-v[0-9]+$', '', 'i')) as distinct_base
from public.shop_catalog
where is_active = true and price > 0
group by currency order by active_count desc;

select gi.slot,gi.subtype,count(*) as active_count, min(c.price) as min_price
from public.catalog_garment_intelligence gi
join public.shop_catalog c on c.id=gi.product_id
where c.is_active and c.price > 0 and c.currency='UZS'
  and gi.slot in ('top','outerwear','bottom','shoes')
group by gi.slot,gi.subtype
order by gi.slot,active_count desc;
```

## Blocker reductions and next acceptance test

- **Solved in ChatGPT session:** canonical 46-photo ZIP physically exists and was revalidated offline. Full ZIP SHA256 `3f78b1ce3912705ab2a3b74cb0f65b53a85235baa571c110237ec4cc1b7f3e8e`; 46/46 image SHA256 valid, 12,972,713 uncompressed image bytes, no duplicates or model usage. Mac still needs download from the ChatGPT conversation; no local transfer was implied.
- **Solved here:** current live Supabase access and currency/subtype statistics. This is NOT yet the requested ID-level drift comparison nor live-universe 12-scenario run.
- **Google model check:** Google AI Gemini API officially lists `gemini-3.6-flash` with image input and JSON structured output. However newer `gemini-3.8-flash` is listed. Google Cloud documents 2026-11-19 retirement for 3.6 Flash **on Cloud Vertex/Agent Platform**, not necessarily the Developer API; choose and verify exact supported model endpoint before any paid API call.
- **Still pending:** actual rights scope for individual imagery, private upload with owner authorization, actual hosted model API key + spend approval, product-image reachability/resolved lawful cache, independent three-person blind votes, and real positive staging credentials.

**Release verdict: NO-GO.** Do not deploy a fake visual critic or claim stylistic improvement from contracts alone.
