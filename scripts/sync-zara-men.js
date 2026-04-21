#!/usr/bin/env node
/**
 * scripts/sync-zara-men.js
 *
 * Pulls live men's product data from Zara category pages via the Apify actor
 * "shahidirfan/zara-product-scraper" and upserts the catalog into Supabase
 * `shop_catalog`.
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
 *   node scripts/sync-zara-men.js
 *   node scripts/sync-zara-men.js --dry-run             # preview without writing
 *   node scripts/sync-zara-men.js --max 20              # cap items per category
 *   node scripts/sync-zara-men.js --keep-stale          # do not deactivate stale Zara rows
 *   node scripts/sync-zara-men.js --keep-legacy-sources # keep old Massimo rows active
 */

'use strict';

try {
    require('dotenv').config();
} catch (_) {
    /* dotenv optional — env vars may already be set in the shell */
}

const https = require('https');
const { createClient } = require('@supabase/supabase-js');

const APIFY_TOKEN  = process.env.APIFY_API_TOKEN;
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY;
const ACTOR_ID     = 'shahidirfan~zara-product-scraper';
const SOURCE       = 'apify-zara-men';

const START_CATEGORIES = [
    { label: 'Basic T-Shirts',     url: 'https://www.zara.com/us/en/man-tshirts-basics-l856.html' },
    { label: 'Shirts',             url: 'https://www.zara.com/us/en/man-shirts-l737.html' },
    { label: 'Casual Shirts',      url: 'https://www.zara.com/us/en/man-shirts-casual-l743.html' },
    { label: 'Formal Shirts',      url: 'https://www.zara.com/us/en/man-shirts-formal-l751.html' },
    { label: 'Linen Shirts',       url: 'https://www.zara.com/us/en/man-shirts-linen-l754.html' },
    { label: 'Plain Polos',        url: 'https://www.zara.com/us/en/man-polos-plain-l1660.html' },
    { label: 'Trousers',           url: 'https://www.zara.com/us/en/man-trousers-l838.html' },
    { label: 'Jeans',              url: 'https://www.zara.com/us/en/man-jeans-l659.html' },
    { label: 'Knitwear',           url: 'https://www.zara.com/us/en/man-knitwear-l681.html' },
    { label: 'Bomber Jackets',     url: 'https://www.zara.com/us/en/man-jackets-bombers-l645.html' },
    { label: 'Shoes',              url: 'https://www.zara.com/us/en/man-shoes-l769.html' },
    { label: 'Blazers',            url: 'https://www.zara.com/us/en/man-blazers-double-breasted-l2506.html' },
];

const LEGACY_SOURCES_TO_DISABLE = ['apify-massimodutti'];

/** Items to fetch per category. 80 × 15 categories = up to 1,200 raw products. */
const maxIdx = process.argv.indexOf('--max');
const MAX_RESULTS_PER_CATEGORY = maxIdx !== -1 ? parseInt(process.argv[maxIdx + 1], 10) : 80;
const DRY_RUN                  = process.argv.includes('--dry-run');
const KEEP_STALE               = process.argv.includes('--keep-stale');
const KEEP_LEGACY_SOURCES      = process.argv.includes('--keep-legacy-sources');

const EXCLUDED_KEYWORDS = [
    'perfume', 'fragrance', 'body spray', 'cologne',
    'belt', 'wallet', 'bag', 'backpack', 'cap', 'hat', 'scarf',
    'sunglasses', 'jewelry', 'bracelet', 'necklace', 'ring', 'watch',
    'slipper', 'sandals', 'flip flop', 'sock',
    '2 pack', '2-pack', 'two pack', 'two-pack',
    '3 pack', '3-pack', 'three pack', 'three-pack',
    '4 pack', '4-pack', 'four pack', 'four-pack',
    'illustration', 'illustrated', 'graphic', 'print', 'printed',
    'patch', 'logo', 'slogan', 'text patch', 'contrast print',
    'tank top', 'tank', 'sleeveless', 'mesh',
    'cargo', 'distressed', 'ripped', 'destroyed',
    'washed', 'faded', 'oversized', 'boxy fit', 'crochet', 'openwork',
    'willy chavarria', 'dylan', 'jaws', 'gremlins',
    'jurassic', 'space jam', 'back to the future',
];

const SPORTY_EXCEPTIONS = ['sport coat', 'sport blazer'];

const SPORTY_PATTERNS = [
    /\bsports?(wear)?\b/,
    /\bathletic\b/,
    /\bathleisure\b/,
    /\btraining\b/,
    /\brunning\b/,
    /\brunner\b/,
    /\bgym\b/,
    /\btrack\b/,
    /\btracksuit\b/,
    /\bjogger(s)?\b/,
    /\bjogging\b/,
    /\btechnical\b/,
    /\bwater repellent\b/,
    /\bwindproof\b/,
    /\bperformance\b/,
    /\bhiking\b/,
    /\btrail\b/,
    /\boutdoor\b/,
    /\bworkout\b/,
    /\bactivewear\b/,
    /\bfootball\b/,
    /\bsoccer\b/,
    /\bbasketball\b/,
    /\btennis\b/,
    /\bpadel\b/,
    /\bcompression\b/,
    /\bbase layer\b/,
    /\btrainer(s)?\b/,
];

const INCLUDED_KEYWORDS = [
    't-shirt', 'tshirt', 'tee', 'shirt', 'polo',
    'trouser', 'trousers', 'pants', 'jeans', 'chino', 'cargo',
    'knit', 'knitwear', 'sweater', 'jumper', 'hoodie', 'sweatshirt',
    'jacket', 'coat', 'outerwear', 'blazer', 'overshirt', 'bomber',
    'shoe', 'shoes', 'boot', 'boots', 'loafer', 'loafers',
    'sneaker', 'sneakers', 'derby', 'oxford', 'moccasin',
];

const SIMILARITY_REPLACEMENTS = [
    [/\bt[\s-]?shirts?\b/g, ' tee '],
    [/\btees?\b/g, ' tee '],
    [/\bslim[\s-]?fit\b/g, ' slimfit '],
    [/\brelaxed[\s-]?fit\b/g, ' relaxedfit '],
    [/\bloose[\s-]?fit\b/g, ' loosefit '],
    [/\bv[\s-]?neck\b/g, ' vneck '],
    [/\bcrew[\s-]?neck\b/g, ' crewneck '],
    [/\bmock[\s-]?neck\b/g, ' mockneck '],
    [/\bbutton[\s-]?down\b/g, ' buttondown '],
    [/\blong[\s-]?sleeve(?:d)?\b/g, ' longsleeve '],
    [/\bshort[\s-]?sleeve(?:d)?\b/g, ' shortsleeve '],
    [/\bheavy[\s-]?weight\b/g, ' heavyweight '],
    [/\bmedium[\s-]?weight\b/g, ' mediumweight '],
    [/\blight[\s-]?weight\b/g, ' lightweight '],
    [/\bwide[\s-]?leg\b/g, ' wideleg '],
    [/\bstraight[\s-]?leg\b/g, ' straightleg '],
    [/\bdouble[\s-]?breasted\b/g, ' doublebreasted '],
    [/\bsingle[\s-]?breasted\b/g, ' singlebreasted '],
    [/\bregular[\s-]?fit\b/g, ' regularfit '],
];

const COLOR_WORDS = new Set([
    'black', 'white', 'offwhite', 'ecru', 'cream', 'beige', 'stone', 'sand',
    'taupe', 'camel', 'khaki', 'olive', 'green', 'sage', 'mint',
    'blue', 'navy', 'indigo', 'brown', 'chocolate', 'grey', 'gray',
    'silver', 'charcoal', 'red', 'burgundy', 'maroon', 'pink',
    'purple', 'lilac', 'yellow', 'mustard', 'orange', 'rust',
]);

const SIMILARITY_STOP_WORDS = new Set([
    'man', 'men', 'zara',
    'regularfit', 'classic', 'basic', 'essential', 'premium',
    'soft', 'touch', 'washed', 'faded', 'lightweight', 'heavyweight',
    'mediumweight', 'slimfit', 'relaxedfit', 'loosefit',
    'textured', 'structured', 'comfort', 'interlock', 'blend', 'stretch',
    'fit',
    'detail', 'details', 'piece', 'collection', 'edition',
    'cotton', 'viscose', 'polyester', 'polyamide', 'elastane',
    'lyocell', 'modal', 'jersey', 'percent',
    'with', 'and', 'the', 'for', 'from',
]);

const SIMILARITY_TOKEN_MAP = new Map([
    ['pant', 'trouser'],
    ['trouser', 'trouser'],
    ['jean', 'jean'],
    ['sweater', 'knit'],
    ['jumper', 'knit'],
    ['knitwear', 'knit'],
    ['trainer', 'sneaker'],
]);

const SILHOUETTE_TOKENS = new Set([
    'tee', 'shirt', 'polo', 'trouser', 'jean', 'chino', 'cargo',
    'knit', 'hoodie', 'sweatshirt', 'cardigan', 'waistcoat',
    'jacket', 'coat', 'blazer', 'overshirt', 'bomber', 'parka', 'puffer',
    'loafer', 'sneaker', 'boot', 'derby', 'oxford', 'moccasin',
]);

function apiFetch(path, opts = {}) {
    return new Promise((resolve, reject) => {
        const parsed = new URL(`https://api.apify.com/v2${path}`);
        const body   = opts.body ? JSON.stringify(opts.body) : undefined;

        const req = https.request(
            {
                hostname: parsed.hostname,
                path: parsed.pathname + parsed.search,
                method: opts.method || 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...(body ? { 'Content-Length': Buffer.byteLength(body) } : {}),
                },
            },
            (res) => {
                let data = '';
                res.on('data', (chunk) => (data += chunk));
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

function pick(obj, ...keys) {
    for (const key of keys) {
        const value = obj[key];
        if (value !== undefined && value !== null && value !== '') return String(value).trim();
    }
    return '';
}

function slug(value, maxLen = 120) {
    return String(value || '')
        .toLowerCase()
        .replace(/https?:\/\//g, '')
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '')
        .slice(0, maxLen);
}

function normalizeText(value) {
    return String(value || '')
        .toLowerCase()
        .replace(/https?:\/\//g, '')
        .replace(/[^a-z0-9]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function extractProductSlug(sourceUrl) {
    if (!sourceUrl) return '';

    try {
        const { pathname } = new URL(sourceUrl);
        const segment = pathname.split('/').filter(Boolean).pop() || '';
        return segment
            .replace(/\.html?$/i, '')
            .replace(/-p\d+$/i, '')
            .toLowerCase();
    } catch {
        return '';
    }
}

function buildProductKey(row = {}) {
    const sourceSlug = extractProductSlug(row.source_url || row.sourceUrl);
    const category = normalizeText(row.category);

    if (sourceSlug) return `${category}:${sourceSlug}`;

    const name = normalizeText(row.name);
    return `${category}:${name}`;
}

function buildFilterText(raw, categoryLabel, sourceUrl) {
    const tagText = Array.isArray(raw.tags) ? raw.tags.join(' ') : '';

    return normalizeText(
        [
            raw.name,
            raw.description,
            raw.category_name,
            raw.family_name,
            raw.subfamily_name,
            raw.type,
            raw.kind,
            tagText,
            categoryLabel,
            sourceUrl,
        ]
            .filter(Boolean)
            .join(' '),
    );
}

function isSportyItem(text) {
    if (SPORTY_EXCEPTIONS.some((phrase) => text.includes(phrase))) {
        return false;
    }

    return SPORTY_PATTERNS.some((pattern) => pattern.test(text));
}

function normalizeSimilaritySource(value) {
    let text = String(value || '').toLowerCase();

    for (const [pattern, replacement] of SIMILARITY_REPLACEMENTS) {
        text = text.replace(pattern, replacement);
    }

    return text;
}

function singularizeToken(token) {
    if (token.endsWith('ies') && token.length > 4) {
        return `${token.slice(0, -3)}y`;
    }

    if (/(sses|shes|ches|xes|zes)$/.test(token)) {
        return token.slice(0, -2);
    }

    if (token.endsWith('s') && token.length > 3 && !token.endsWith('ss')) {
        return token.slice(0, -1);
    }

    return token;
}

function getSimilarityTokens(row = {}) {
    const source = typeof row === 'string'
        ? row
        : [row.name]
            .filter(Boolean)
            .join(' ');
    const normalized = normalizeText(normalizeSimilaritySource(source));
    const tokens = [];
    const seen = new Set();

    for (const rawToken of normalized.split(' ')) {
        if (!rawToken || /^\d+$/.test(rawToken)) continue;

        const singular = singularizeToken(rawToken);
        const token = SIMILARITY_TOKEN_MAP.get(singular) || singular;

        if (token.length < 2) continue;
        if (COLOR_WORDS.has(token) || SIMILARITY_STOP_WORDS.has(token)) continue;
        if (seen.has(token)) continue;

        seen.add(token);
        tokens.push(token);
    }

    return tokens;
}

function buildSimilarityKey(row = {}) {
    const category = normalizeText(row.category);
    const tokens = getSimilarityTokens(row).filter((token) => token !== category);
    return `${category}:${tokens.join(' ') || normalizeText(row.name)}`;
}

function getPrimarySilhouette(row = {}) {
    const tokens = getSimilarityTokens(row);
    return tokens.find((token) => SILHOUETTE_TOKENS.has(token)) || normalizeText(row.category);
}

function areRowsTooSimilar(existingRow, candidateRow) {
    if (!existingRow || !candidateRow) return false;

    if (buildProductKey(existingRow) === buildProductKey(candidateRow)) {
        return true;
    }

    if (normalizeText(existingRow.category) !== normalizeText(candidateRow.category)) {
        return false;
    }

    if (buildSimilarityKey(existingRow) === buildSimilarityKey(candidateRow)) {
        return true;
    }

    if (getPrimarySilhouette(existingRow) !== getPrimarySilhouette(candidateRow)) {
        return false;
    }

    const existingTokens = getSimilarityTokens(existingRow);
    const candidateTokens = getSimilarityTokens(candidateRow);
    if (existingTokens.length === 0 || candidateTokens.length === 0) {
        return false;
    }

    const existingSet = new Set(existingTokens);
    const candidateSet = new Set(candidateTokens);
    const sharedCount = existingTokens.filter((token) => candidateSet.has(token)).length;
    if (sharedCount === 0) {
        return false;
    }

    const smallerSetSize = Math.min(existingSet.size, candidateSet.size);
    const unionSize = new Set([...existingSet, ...candidateSet]).size;
    const overlap = sharedCount / smallerSetSize;
    const jaccard = sharedCount / unionSize;

    return overlap >= 0.85 || (smallerSetSize <= 3 && overlap >= 0.67) || jaccard >= 0.75;
}

function selectCatalogRow(selectedRowsByCategory, candidateRow) {
    const categoryKey = normalizeText(candidateRow.category);
    const bucket = selectedRowsByCategory.get(categoryKey) || [];

    for (let index = 0; index < bucket.length; index++) {
        const existingRow = bucket[index];
        if (!areRowsTooSimilar(existingRow, candidateRow)) {
            continue;
        }

        if (shouldReplaceRow(existingRow, candidateRow)) {
            bucket[index] = candidateRow;
            selectedRowsByCategory.set(categoryKey, bucket);
            return 'replaced';
        }

        return 'discarded';
    }

    bucket.push(candidateRow);
    selectedRowsByCategory.set(categoryKey, bucket);
    return 'added';
}

function shouldReplaceRow(existingRow, candidateRow) {
    if (!existingRow) return true;
    if (candidateRow.sort_order !== existingRow.sort_order) {
        return candidateRow.sort_order < existingRow.sort_order;
    }

    return String(candidateRow.id) < String(existingRow.id);
}

function categoryToGarmentType(raw = '') {
    const s = raw.toLowerCase();
    // IMPORTANT: check "shirt" / "polo" / "tee" BEFORE the shoes regex,
    // otherwise titles like "STRIPED OXFORD SHIRT" match the bare word
    // "oxford" in the shoes regex and get misclassified as shoes.
    if (/\b(shirt|t-shirt|tee|blouse|polo|polo shirt|henley|tank|vest top|sweatshirt|sweater|cardigan|pullover|knitwear|jumper|hoodie)\b/.test(s)) {
        if (/blazer|jacket|coat|outerwear|parka|puffer|bomber|windbreaker|overshirt/.test(s)) {
            return { garment_type: 'upper_body', category: 'outerwear' };
        }
        return { garment_type: 'upper_body', category: 'tops' };
    }
    if (/\b(shoe|shoes|boot|boots|sneaker|sneakers|loafer|loafers|derby|derbies|moccasin|moccasins|trainer|trainers|brogue|brogues|espadrille|espadrilles|sandal|sandals)\b/.test(s)) {
        return { garment_type: 'shoes', category: 'shoes' };
    }
    // Treat bare "oxford" as a shoe only when not accompanied by shirt (handled above).
    if (/\boxford\b/.test(s) && !/shirt/.test(s)) {
        return { garment_type: 'shoes', category: 'shoes' };
    }
    if (/pant|trouser|jean|short|chino|cargo/.test(s)) {
        return { garment_type: 'lower_body', category: 'bottoms' };
    }
    if (/blazer|jacket|coat|outerwear|parka|puffer|bomber|windbreaker|overshirt/.test(s)) {
        return { garment_type: 'upper_body', category: 'outerwear' };
    }
    if (/suit/.test(s)) {
        return { garment_type: 'outfit', category: 'outfits' };
    }
    return { garment_type: 'upper_body', category: 'tops' };
}

function isSupportedMensClothing(raw, categoryLabel, sourceUrl) {
    const text = buildFilterText(raw, categoryLabel, sourceUrl);

    if (raw.extra_info?.hideProductInfo || raw.extra_info?.isAddToCartInGridDisabled) {
        return false;
    }

    if (EXCLUDED_KEYWORDS.some((keyword) => text.includes(keyword))) {
        return false;
    }

    if (isSportyItem(text)) {
        return false;
    }

    return INCLUDED_KEYWORDS.some((keyword) => text.includes(keyword));
}

function pickImage(raw) {
    const isRenderableImageUrl = (value) =>
        typeof value === 'string' &&
        value.startsWith('http') &&
        !value.includes('.m3u8') &&
        !value.includes('/master.m3u8');

    const imageScore = (value) => {
        if (!isRenderableImageUrl(value)) return -1000;

        const url = value.toLowerCase();
        let score = 0;

        if (/([/-])e1([/.])/.test(url) || url.includes('/e1/')) score += 500;
        else if (/([/-])e2([/.])/.test(url) || url.includes('/e2/')) score += 450;
        else if (/([/-])f1([/.])/.test(url) || url.includes('/f1/')) score += 350;
        else if (/([/-])f2([/.])/.test(url) || url.includes('/f2/')) score += 300;
        else if (/([/-])p([/.])/.test(url) || url.includes('-p/')) score += 260;
        else if (/([/-])a1([/.])/.test(url) || url.includes('/a1/')) score += 80;
        else if (/([/-])a2([/.])/.test(url) || url.includes('/a2/')) score += 60;
        else if (/([/-])a3([/.])/.test(url) || url.includes('/a3/')) score += 40;
        else score += 100;

        if (url.includes('/a1/') || url.includes('/a2/') || url.includes('/a3/')) score -= 50;
        if (url.includes('master.m3u8')) score -= 1000;

        return score;
    };

    const direct = pick(raw, 'image_url');
    const candidates = [];
    if (isRenderableImageUrl(direct)) candidates.push(direct);

    if (Array.isArray(raw.image_urls)) {
        for (const url of raw.image_urls) {
            if (isRenderableImageUrl(url)) candidates.push(url);
        }
    }

    if (candidates.length === 0) return '';

    candidates.sort((a, b) => imageScore(b) - imageScore(a));
    return candidates[0];
}

function parsePrice(rawPrice, rawMinor) {
    const price = Number(rawPrice);
    if (Number.isFinite(price) && price > 0) return price;

    const minor = Number(rawMinor);
    if (Number.isFinite(minor) && minor > 0) return minor / 100;

    return 0;
}

async function scrapeCategory(category) {
    console.log(`\n▶  Scraping Zara Men — ${category.label}`);
    const path = `/acts/${ACTOR_ID}/run-sync-get-dataset-items?token=${encodeURIComponent(APIFY_TOKEN)}&clean=true&format=json`;
    const res = await apiFetch(path, {
        method: 'POST',
        body: {
            startUrl: category.url,
            results_wanted: MAX_RESULTS_PER_CATEGORY,
        },
    });

    if (!res.status || res.status < 200 || res.status >= 300) {
        const preview =
            typeof res.body === 'string'
                ? res.body.slice(0, 500)
                : JSON.stringify(res.body).slice(0, 500);
        throw new Error(`Zara scrape failed for ${category.label} (status ${res.status}): ${preview}`);
    }

    const items = Array.isArray(res.body) ? res.body : [];
    console.log(`   Raw items collected  : ${items.length}`);
    return items;
}

function mapItem(raw, categoryIndex, itemIndex, categoryLabel, sourceUrl, syncTimestamp) {
    if (!isSupportedMensClothing(raw, categoryLabel, sourceUrl)) return null;

    const name = pick(raw, 'name');
    const imageUrl = pickImage(raw);
    if (!name || !imageUrl) return null;

    const garmentInput = [
        raw.category_name,
        raw.family_name,
        raw.subfamily_name,
        raw.type,
        raw.kind,
        categoryLabel,
        sourceUrl,
        name,
    ]
        .filter(Boolean)
        .join(' ');

    const { garment_type, category } = categoryToGarmentType(garmentInput);
    const price = parsePrice(raw.price, raw.price_minor);
    const currency = pick(raw, 'currency') || 'USD';
    const externalId = pick(raw, 'seo_product_id', 'product_reference', 'display_reference', 'product_id');
    const productUrl = pick(raw, 'product_url') || sourceUrl;
    const id = externalId
        ? `zara-apify-${slug(externalId, 80)}`
        : `zara-apify-${slug(name + imageUrl, 100)}`;

    return {
        id,
        brand: 'ZARA',
        name,
        price,
        currency,
        category,
        garment_type,
        description: pick(raw, 'description') || '',
        image_url: imageUrl,
        source_url: productUrl,
        source: SOURCE,
        external_id: externalId || null,
        is_active: true,
        sort_order: categoryIndex * 1000 + itemIndex,
        last_seen_at: syncTimestamp,
        updated_at: syncTimestamp,
    };
}

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

    const batchSize = 100;
    let upserted = 0;

    for (let i = 0; i < rows.length; i += batchSize) {
        const batch = rows.slice(i, i + batchSize);
        const { error } = await client.from('shop_catalog').upsert(batch, { onConflict: 'id' });
        if (error) {
            console.error(`\n❌  Batch ${Math.floor(i / batchSize) + 1} failed:`, error.message);
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
        console.log(`   Skipping stale Zara deactivation (${mode}).`);
        return;
    }

    console.log('   Deactivating stale Zara rows…');
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
        console.error(`\n⚠  Failed to deactivate stale Zara rows: ${error.message}`);
        return;
    }

    console.log(`   Deactivated stale Zara : ${data?.length ?? 0}`);
}

async function deactivateDuplicateRows(client, selectedRows) {
    if (DRY_RUN) {
        console.log('   Skipping duplicate Zara cleanup (dry-run).');
        return;
    }

    if (!Array.isArray(selectedRows) || selectedRows.length === 0) {
        console.log('   Skipping duplicate Zara cleanup (no selected rows).');
        return;
    }

    console.log('   Deactivating duplicate/similar Zara rows…');

    const selectedIds = new Set(selectedRows.map((row) => row.id));
    const selectedRowsByCategory = new Map();
    for (const row of selectedRows) {
        const categoryKey = normalizeText(row.category);
        const bucket = selectedRowsByCategory.get(categoryKey) || [];
        bucket.push(row);
        selectedRowsByCategory.set(categoryKey, bucket);
    }
    const { data, error } = await client
        .from('shop_catalog')
        .select('id, name, category, source_url')
        .eq('source', SOURCE)
        .eq('is_active', true);

    if (error) {
        console.error(`\n⚠  Failed to fetch Zara duplicates: ${error.message}`);
        return;
    }

    const duplicateIds = (data || [])
        .filter((row) => {
            if (selectedIds.has(row.id)) {
                return false;
            }

            const bucket = selectedRowsByCategory.get(normalizeText(row.category)) || [];
            return bucket.some((selectedRow) => areRowsTooSimilar(selectedRow, row));
        })
        .map((row) => row.id);

    if (duplicateIds.length === 0) {
        console.log('   Deactivated duplicates : 0');
        return;
    }

    const { data: deactivated, error: deactivateError } = await client
        .from('shop_catalog')
        .update({
            is_active: false,
            updated_at: new Date().toISOString(),
        })
        .in('id', duplicateIds)
        .select('id');

    if (deactivateError) {
        console.error(`\n⚠  Failed to deactivate Zara duplicates: ${deactivateError.message}`);
        return;
    }

    console.log(`   Deactivated duplicates : ${deactivated?.length ?? 0}`);
}

async function deactivateLegacySources(client) {
    if (DRY_RUN || KEEP_LEGACY_SOURCES || LEGACY_SOURCES_TO_DISABLE.length === 0) {
        const mode = DRY_RUN ? 'dry-run' : 'keep-legacy-sources';
        console.log(`   Skipping legacy source cleanup (${mode}).`);
        return;
    }

    console.log(`   Deactivating legacy sources: ${LEGACY_SOURCES_TO_DISABLE.join(', ')}`);
    const { data, error } = await client
        .from('shop_catalog')
        .update({
            is_active: false,
            updated_at: new Date().toISOString(),
        })
        .in('source', LEGACY_SOURCES_TO_DISABLE)
        .eq('is_active', true)
        .select('id');

    if (error) {
        console.error(`\n⚠  Failed to deactivate legacy sources: ${error.message}`);
        return;
    }

    console.log(`   Deactivated legacy rows: ${data?.length ?? 0}`);
}

async function main() {
    if (!APIFY_TOKEN) {
        console.error('❌  APIFY_API_TOKEN is not set. Add it to .env or export it.');
        process.exit(1);
    }

    console.log(`\n🛍  Zara Men → shop_catalog sync`);
    console.log(`   max results per cat. : ${MAX_RESULTS_PER_CATEGORY}`);
    console.log(`   category URLs        : ${START_CATEGORIES.length}`);
    console.log(`   dry run              : ${DRY_RUN}\n`);

    const syncTimestamp = new Date().toISOString();
    const client = DRY_RUN ? null : createSupabaseAdminClient();
    const selectedRowsByCategory = new Map();
    let rawItemCount = 0;
    let skippedUnsupported = 0;
    let skippedSimilar = 0;
    const failedCategories = [];

    for (let categoryIndex = 0; categoryIndex < START_CATEGORIES.length; categoryIndex++) {
        const category = START_CATEGORIES[categoryIndex];
        let items = [];
        try {
            items = await scrapeCategory(category);
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            failedCategories.push(`${category.label}: ${message}`);
            console.warn(`   Skipping category     : ${category.label}`);
            continue;
        }

        rawItemCount += items.length;
        for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
            const row = mapItem(
                items[itemIndex],
                categoryIndex,
                itemIndex,
                category.label,
                category.url,
                syncTimestamp,
            );

            if (!row) {
                skippedUnsupported++;
                continue;
            }

            const selectionResult = selectCatalogRow(selectedRowsByCategory, row);
            if (selectionResult === 'discarded') {
                skippedSimilar++;
            }
        }
    }

    const rows = Array.from(selectedRowsByCategory.values())
        .flat()
        .sort((a, b) => a.sort_order - b.sort_order);

    console.log(`\n   Raw items from Zara  : ${rawItemCount}`);
    console.log(`   Deduped rows         : ${rows.length}`);
    console.log(`   Skipped unsupported  : ${skippedUnsupported}`);
    console.log(`   Skipped similar      : ${skippedSimilar}`);
    console.log(`   Failed categories    : ${failedCategories.length}`);

    if (rows.length === 0) {
        console.log('\n⚠  Nothing to upsert. Zara may have returned empty results.');
        console.log('   Tip: try --dry-run and validate one category URL first.');
        return;
    }

    await upsertRows(client, rows);

    if (client) {
        await deactivateDuplicateRows(client, rows);
        await deactivateStaleRows(client, syncTimestamp);
        await deactivateLegacySources(client);
    }

    if (failedCategories.length > 0) {
        console.log('   Failed category list :');
        failedCategories.forEach((entry) => console.log(`   - ${entry}`));
    }

    console.log(`\n✅  Done — ${rows.length} Zara Men items synced to shop_catalog.\n`);
}

if (require.main === module) {
    main().catch((err) => {
        console.error('\n❌', err.message || err);
        process.exit(1);
    });
}

module.exports = {
    areRowsTooSimilar,
    buildProductKey,
    buildSimilarityKey,
    categoryToGarmentType,
    extractProductSlug,
    getSimilarityTokens,
    isSupportedMensClothing,
    normalizeText,
    shouldReplaceRow,
};
