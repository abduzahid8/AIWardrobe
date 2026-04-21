#!/usr/bin/env node
/**
 * scripts/sync-massimo-dutti.js
 *
 * Pulls live product data from the Massimo Dutti website via the Apify
 * actor "datasaurus/massimodutti", then upserts every item into the
 * Supabase `shop_catalog` table.
 *
 * ── Prerequisites ──────────────────────────────────────────────────────────
 *  1. Apply supabase/migrations/005_shop_catalog.sql
 *  2. Apply supabase/migrations/006_shop_catalog_feed_sync.sql
 *  3. Fill .env (or export to shell):
 *       APIFY_API_TOKEN      – from console.apify.com → Settings → API tokens
 *       SUPABASE_URL         – your project URL
 *       SUPABASE_SERVICE_KEY – service_role key (never expose to client)
 *
 * ── Usage ─────────────────────────────────────────────────────────────────
 *   node scripts/sync-massimo-dutti.js
 *   node scripts/sync-massimo-dutti.js --dry-run      # preview without writing
 *   node scripts/sync-massimo-dutti.js --max 50       # cap scraped items
 *
 * ── What it scrapes ────────────────────────────────────────────────────────
 *  Starts from the category URLs in START_URLS below.
 *  Change them or add more to pull different sections of the site.
 */

'use strict';

// ── Load .env if present ──────────────────────────────────────────────────
try {
    require('dotenv').config();
} catch (_) {
    /* dotenv optional — env vars may already be set in the shell */
}

const https = require('https');
const { createClient } = require('@supabase/supabase-js');

// ── Config ────────────────────────────────────────────────────────────────

const APIFY_TOKEN   = process.env.APIFY_API_TOKEN;
const SUPABASE_URL  = process.env.SUPABASE_URL  || process.env.EXPO_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY  = process.env.SUPABASE_SERVICE_KEY;
const ACTOR_ID      = 'datasaurus~massimodutti'; // same as datasaurus/massimodutti
const SOURCE        = 'apify-massimodutti';

/** Men-only category pages to scrape. */
const START_URLS = [
    // ── Jackets & Coats ────────────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/jackets-n1447?celement=2028553' },
    { url: 'https://www.massimodutti.com/gb/men/coats-n1447?celement=2028554' },
    { url: 'https://www.massimodutti.com/gb/men/puffer-jackets-n1447?celement=2028556' },
    // ── Suits & Blazers ────────────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/suits-n1447?celement=2028557' },
    { url: 'https://www.massimodutti.com/gb/men/blazers-n1447?celement=2028558' },
    // ── Shirts ─────────────────────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/shirts-n1447?celement=2028542' },
    { url: 'https://www.massimodutti.com/gb/men/casual-shirts-n1447?celement=2028544' },
    // ── Knitwear & Sweatshirts ─────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/knitwear-n1447?celement=2028543' },
    { url: 'https://www.massimodutti.com/gb/men/sweatshirts-n1447?celement=2028546' },
    // ── Trousers & Jeans ───────────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/trousers-n1447?celement=2028547' },
    { url: 'https://www.massimodutti.com/gb/men/jeans-n1447?celement=2028548' },
    { url: 'https://www.massimodutti.com/gb/men/chinos-n1447?celement=2028549' },
    // ── T-shirts & Polos ───────────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/t-shirts-n1447?celement=2028540' },
    { url: 'https://www.massimodutti.com/gb/men/polo-shirts-n1447?celement=2028541' },
    // ── Shoes ──────────────────────────────────────────────────────────────
    { url: 'https://www.massimodutti.com/gb/men/shoes-n1447?celement=2028563' },
];

/** Items to fetch per start URL. 100 × 15 categories = up to 1,500 raw products. */
const maxIdx = process.argv.indexOf('--max');
const MAX_RESULTS_PER_URL = maxIdx !== -1 ? parseInt(process.argv[maxIdx + 1], 10) : 100;
const DRY_RUN             = process.argv.includes('--dry-run');
const KEEP_STALE          = process.argv.includes('--keep-stale');

// ── Category → garment_type mapping ──────────────────────────────────────

function categoryToGarmentType(raw = '') {
    const s = raw.toLowerCase();
    if (/dress|jumpsuit|gown/.test(s))             return { garment_type: 'dresses',    category: 'dresses'   };
    if (/shoe|boot|sneaker|loafer|heel|flat/.test(s)) return { garment_type: 'shoes',   category: 'shoes'     };
    if (/pant|trouser|jean|short|skirt|legging/.test(s)) return { garment_type: 'lower_body', category: 'bottoms' };
    if (/coat|jacket|blazer|outerwear|parka|puffer/.test(s)) return { garment_type: 'upper_body', category: 'outerwear' };
    if (/suit/.test(s))                            return { garment_type: 'outfit',     category: 'outfits'   };
    return { garment_type: 'upper_body', category: 'tops' };
}

// ── Apify helpers ─────────────────────────────────────────────────────────

function apiFetch(path, opts = {}) {
    return new Promise((resolve, reject) => {
        const url    = `https://api.apify.com/v2${path}`;
        const parsed = new URL(url);
        const body   = opts.body ? JSON.stringify(opts.body) : undefined;

        const req = https.request(
            {
                hostname: parsed.hostname,
                path:     parsed.pathname + parsed.search,
                method:   opts.method || 'GET',
                headers: {
                    Authorization: `Bearer ${APIFY_TOKEN}`,
                    'Content-Type': 'application/json',
                    ...(body ? { 'Content-Length': Buffer.byteLength(body) } : {}),
                },
            },
            (res) => {
                let data = '';
                res.on('data', (c) => (data += c));
                res.on('end', () => {
                    try {
                        resolve({ status: res.statusCode, body: JSON.parse(data) });
                    } catch {
                        resolve({ status: res.statusCode, body: data });
                    }
                });
            },
        );
        req.on('error', reject);
        if (body) req.write(body);
        req.end();
    });
}

async function startRun() {
    console.log('▶  Starting Apify actor run…');
    const res = await apiFetch(`/acts/${ACTOR_ID}/runs`, {
        method: 'POST',
        body: {
            startUrls:                   START_URLS,
            maxResultsPerStartUrl:        MAX_RESULTS_PER_URL,
            deduplicateAcrossAllStartUrls: true,
        },
    });
    if (res.status !== 201) {
        throw new Error(`Failed to start actor run: ${JSON.stringify(res.body)}`);
    }
    const runId = res.body?.data?.id;
    if (!runId) throw new Error('No runId in response');
    console.log(`   Run ID: ${runId}`);
    return runId;
}

async function waitForRun(runId) {
    const POLL_MS = 5000;
    process.stdout.write('   Waiting for actor to finish');
    for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        const res = await apiFetch(`/actor-runs/${runId}`);
        const status = res.body?.data?.status;
        process.stdout.write('.');
        if (status === 'SUCCEEDED') { process.stdout.write(' done.\n'); return; }
        if (['FAILED', 'ABORTED', 'TIMED-OUT'].includes(status)) {
            process.stdout.write('\n');
            throw new Error(`Actor run ended with status: ${status}`);
        }
    }
}

async function fetchDataset(runId) {
    console.log('   Fetching dataset items…');
    const res = await apiFetch(`/actor-runs/${runId}/dataset/items?clean=true&format=json&limit=9999`);
    if (res.status !== 200) throw new Error(`Dataset fetch failed: ${JSON.stringify(res.body)}`);
    return Array.isArray(res.body) ? res.body : (res.body?.items ?? []);
}

// ── Data mapping ──────────────────────────────────────────────────────────

function pick(obj, ...keys) {
    for (const k of keys) {
        const v = obj[k];
        if (v !== undefined && v !== null && v !== '') return String(v).trim();
    }
    return '';
}

/**
 * Gender filter — returns true for confirmed WOMEN'S items.
 *
 * The actor ignores the gender of start URLs and mixes the full catalog.
 * Reliable signal: `skuDimensions` inside `colorsSizesImagesJSON[].sizes[]`
 * contains dimensionName "CHEST/BUST" (women) vs "CHEST" alone (men).
 * Also checks item name for common women's garment keywords.
 */
function isWomensItem(raw) {
    // Check skuDimensions for BUST indicator
    const colorData = raw.colorsSizesImagesJSON;
    if (Array.isArray(colorData)) {
        for (const color of colorData) {
            const sizes = color.sizes || [];
            for (const size of sizes) {
                const dims = size.skuDimensions || [];
                for (const dim of dims) {
                    const name = (dim.dimensionName || '').toUpperCase();
                    if (name.includes('BUST') || name.includes('HIP')) return true;
                }
            }
        }
    }
    // Fallback: women's garment name keywords
    const n = (raw.name || raw.nameEn || '').toLowerCase();
    const womensWords = [
        'blouse','skirt','heel','jumpsuit','legging','cardigan crop',
        'floral','feminine','v-neck dress','midi dress','maxi dress',
        'bodysuit','corset','bralette',
    ];
    return womensWords.some((w) => n.includes(w));
}

function isSupportedMensClothing(raw) {
    const text = [
        raw.name,
        raw.nameEn,
        raw.productType,
        raw.category,
        raw.categoryName,
        raw.productPage,
        raw.productUrl,
        raw.url,
        raw.link,
    ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

    const excludedKeywords = [
        'bra', 'brief', 'briefs', 'thong', 'bralette', 'bandeau', 'corset',
        'lingerie', 'lace', 'floral lace', 'semi-sheer', 'triangle top',
        'sunglasses', 'towel', 'espadrille', 'espadrilles',
        'sandals', 'slippers', 'belt', 'wallet',
        'bag', 'cap', 'hat', 'scarf', 'swim', 'bikini', 'underwire',
        'cowl neck', 'gold buttons',
        'studio', 'crop', 'cropped', 'feminine', 'ruffle', 'ruffled',
        'sheer', 'slim fit dress', 'kimono', 'beachwear',
    ];

    if (excludedKeywords.some((keyword) => text.includes(keyword))) {
        return false;
    }

    if (text.includes('/men/')) {
        return true;
    }

    const includedKeywords = [
        'jacket', 'coat', 'blazer', 'shirt', 't-shirt', 'tshirt', 'tee',
        'polo', 'sweater', 'knit', 'cardigan', 'trouser', 'trousers',
        'pants', 'jeans', 'chino', 'suit', 'waistcoat', 'bomber',
        'parka', 'puffer', 'overshirt', 'sweatshirt', 'hoodie',
        'jumper', 'shoe', 'shoes', 'boot', 'boots', 'loafer', 'loafers',
        'sneaker', 'sneakers', 'trainer', 'trainers', 'derby', 'oxford',
        'moccasin', 'moc toe',
    ];

    return includedKeywords.some((keyword) => text.includes(keyword));
}

/**
 * Pick best product image from this actor's data.
 *
 * The actor exposes `colorsSizesImagesJSON[].xmedia[]` with all shots.
 * Massimo Dutti image suffix guide (from inspecting CDN URLs):
 *   -o3  → main ghost/packshot front (best)
 *   -o1  → editorial model front (avoid)
 *   -o2  → editorial model side  (avoid)
 *   -c   → close-up / swatch
 *   -r   → rear view
 *   -t   → texture / detail
 *
 * Strategy: prefer -o3 packshot, then any -o suffix, then mainImage fallback.
 */
function pickImage(raw) {
    const allUrls = [];

    // Pull xmedia from first available color variant
    const colorData = raw.colorsSizesImagesJSON;
    if (Array.isArray(colorData) && colorData.length > 0) {
        const xmedia = colorData[0].xmedia || [];
        for (const u of xmedia) {
            if (typeof u === 'string' && u.startsWith('http')) allUrls.push(u);
        }
    }

    // Score: prefer ghost/packshot shots, penalise model shots
    if (allUrls.length > 0) {
        const score = (url) => {
            if (/-o3\./.test(url)) return 100;   // main packshot (ghost mannequin)
            if (/-o4\./.test(url)) return 80;
            if (/-o5\./.test(url)) return 70;
            if (/-o\d/.test(url))  return 50;    // other editorial — lower priority
            if (/-o1\./.test(url)) return 20;    // usually on-model front
            if (/-o2\./.test(url)) return 15;
            return 40;
        };
        allUrls.sort((a, b) => score(b) - score(a));
        return allUrls[0];
    }

    // Direct fallback
    return pick(raw, 'mainImage', 'imageUrl', 'image_url', 'image', 'thumbnail');
}

function parsePrice(raw) {
    if (!raw) return 0;
    const n = parseFloat(String(raw).replace(/[^\d.]/g, ''));
    if (!Number.isFinite(n)) return 0;
    // Massimo Dutti actor returns prices in minor currency units (pence/cents)
    // e.g. 16900 → £169.00. Divide when no decimal and value > 1000.
    if (Number.isInteger(n) && n > 1000) return n / 100;
    return n;
}

function mapItem(raw, index, syncTimestamp) {
    // Drop women's items
    if (isWomensItem(raw)) return null;
    if (!isSupportedMensClothing(raw)) return null;

    const name = pick(raw, 'name', 'nameEn', 'title', 'productName');
    const imageUrl = pickImage(raw);
    if (!name || !imageUrl) return null;

    // Use productType + name for category detection
    const rawCat = pick(raw, 'productType', 'category', 'categoryName');
    const catInput = (rawCat && !['clothing', 'all', 'other', 'tops'].includes(rawCat.toLowerCase()))
        ? rawCat
        : `${rawCat} ${name}`;
    const { garment_type, category } = categoryToGarmentType(catInput);

    const price    = parsePrice(pick(raw, 'price', 'currentPrice', 'salePrice'));
    const currency = 'GBP';

    const externalId = String(raw.id || raw.productId || raw.itemId || '');
    const sourceUrl  = pick(raw, 'productPage', 'url', 'productUrl', 'link', 'pageUrl');

    const id  = externalId
        ? `md-apify-${externalId}`
        : `md-apify-${name.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 80)}`;

    return {
        id,
        brand:        'Massimo Dutti',
        name,
        price,
        currency,
        category,
        garment_type,
        description:  pick(raw, 'longDescription', 'description', 'shortDescription') || '',
        image_url:    imageUrl,
        source_url:   sourceUrl,
        source:       SOURCE,
        external_id:  externalId || null,
        is_active:    true,
        sort_order:   index * 10,
        last_seen_at: syncTimestamp,
        updated_at:   syncTimestamp,
    };
}

// ── Supabase upsert ───────────────────────────────────────────────────────

function createSupabaseAdminClient() {
    if (!SUPABASE_URL || !SUPABASE_KEY) {
        console.error('\n❌  Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env first.');
        process.exit(1);
    }

    return createClient(SUPABASE_URL, SUPABASE_KEY);
}

async function upsertRows(client, rows) {
    if (DRY_RUN) {
        console.log(`\n[dry-run] Would upsert ${rows.length} rows. First 3:`);
        console.log(JSON.stringify(rows.slice(0, 3), null, 2));
        return;
    }

    const BATCH     = 100;
    let   upserted  = 0;

    for (let i = 0; i < rows.length; i += BATCH) {
        const batch = rows.slice(i, i + BATCH);
        const { error } = await client.from('shop_catalog').upsert(batch, { onConflict: 'id' });
        if (error) {
            console.error(`\n❌  Batch ${Math.floor(i / BATCH) + 1} failed:`, error.message);
            process.exit(1);
        }
        upserted += batch.length;
        process.stdout.write(`\r   Upserted ${upserted} / ${rows.length}`);
    }
    process.stdout.write('\n');
}

async function deactivateStaleRows(client, syncTimestamp) {
    if (DRY_RUN || KEEP_STALE) {
        const mode = DRY_RUN ? 'dry-run' : 'keep-stale';
        console.log(`   Skipping stale deactivation (${mode}).`);
        return;
    }

    console.log('   Deactivating stale products…');
    const { data, error } = await client
        .from('shop_catalog')
        .update({
            is_active: false,
            updated_at: new Date().toISOString(),
        })
        .eq('source', SOURCE)
        .eq('is_active', true)
        .lt('last_seen_at', syncTimestamp)
        .select('id');

    if (error) {
        console.error(`\n⚠  Failed to deactivate stale rows: ${error.message}`);
        return;
    }

    console.log(`   Deactivated stale rows : ${data?.length ?? 0}`);
}

// ── Main ──────────────────────────────────────────────────────────────────

async function main() {
    if (!APIFY_TOKEN) {
        console.error('❌  APIFY_API_TOKEN is not set. Add it to .env or export it.');
        process.exit(1);
    }

    console.log(`\n🛍  Massimo Dutti Men → shop_catalog sync`);
    console.log(`   max results per URL : ${MAX_RESULTS_PER_URL}`);
    console.log(`   start URLs          : ${START_URLS.length}`);
    console.log(`   dry run             : ${DRY_RUN}\n`);

    const syncTimestamp = new Date().toISOString();
    const runId  = await startRun();
    await waitForRun(runId);
    const items  = await fetchDataset(runId);

    console.log(`   Raw items from Apify: ${items.length}`);

    const rows = [];
    let skippedWomens = 0;
    for (let i = 0; i < items.length; i++) {
        const row = mapItem(items[i], i, syncTimestamp);
        if (row) rows.push(row);
        else skippedWomens++;
    }

    console.log(`   Mapped rows (men's)  : ${rows.length}`);
    console.log(`   Skipped (women's)    : ${skippedWomens}`);

    if (rows.length === 0) {
        console.log('\n⚠  Nothing to upsert. The actor may have returned empty results.');
        console.log('   Tip: try --dry-run and check the Apify dataset in the console.');
        return;
    }

    const client = DRY_RUN ? null : createSupabaseAdminClient();
    await upsertRows(client, rows);
    if (client) await deactivateStaleRows(client, syncTimestamp);
    console.log(`\n✅  Done — ${rows.length} Massimo Dutti items synced to shop_catalog.\n`);
}

main().catch((err) => {
    console.error('\n❌', err.message || err);
    process.exit(1);
});
