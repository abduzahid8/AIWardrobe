/**
 * Import shop catalog rows from CSV or Google Shopping–style XML into Supabase `shop_catalog`.
 *
 * Env:
 *   SUPABASE_URL
 *   SUPABASE_SERVICE_KEY   (service_role — never commit)
 *
 * Usage:
 *   npx tsx scripts/import-feed.ts --file ./feed.csv --source awin-zara
 *   npx tsx scripts/import-feed.ts --file ./products.xml --source google-feed --dry-run
 *
 * CSV headers (case-insensitive; extra columns ignored):
 *   id, external_id, brand, name, price, currency, category, garment_type,
 *   description, image_url, source_url, sort_order
 *
 * If `id` is omitted, id = slug(source + external_id) or slug(source + name + image_url).
 *
 * garment_type must be one of: upper_body | lower_body | dresses | shoes | outfit
 * category must be one of: tops | bottoms | shoes | dresses | outerwear | outfits
 */

import * as fs from 'fs';
import * as path from 'path';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { parse } from 'csv-parse/sync';
import { XMLParser } from 'fast-xml-parser';

const GARMENT_TYPES = new Set(['upper_body', 'lower_body', 'dresses', 'shoes', 'outfit']);
const CATEGORIES = new Set(['tops', 'bottoms', 'shoes', 'dresses', 'outerwear', 'outfits']);

type Row = {
    id: string;
    brand: string;
    name: string;
    price: number;
    currency: string;
    category: string;
    garment_type: string;
    description: string;
    image_url: string;
    source_url: string;
    sort_order: number;
    source: string;
    external_id: string | null;
    is_active: boolean;
    last_seen_at: string;
    updated_at: string;
};

function arg(name: string): string | undefined {
    const i = process.argv.indexOf(name);
    if (i === -1 || !process.argv[i + 1]) return undefined;
    return process.argv[i + 1];
}

function hasFlag(name: string): boolean {
    return process.argv.includes(name);
}

function normalizeHeader(h: string): string {
    return h.trim().toLowerCase().replace(/\s+/g, '_');
}

function slug(s: string, maxLen = 120): string {
    const t = s
        .toLowerCase()
        .replace(/https?:\/\//g, '')
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '')
        .slice(0, maxLen);
    return t || 'item';
}

function stableId(source: string, externalId: string | undefined, name: string, imageUrl: string): string {
    if (externalId && externalId.trim()) return `${slug(source, 40)}-${slug(externalId.trim(), 80)}`;
    return `${slug(source, 40)}-${slug(name + imageUrl, 100)}`;
}

function parsePrice(raw: string | undefined): number {
    if (!raw) return 0;
    const n = parseFloat(String(raw).replace(/[^\d.]/g, ''));
    return Number.isFinite(n) ? n : 0;
}

function mapCategoryToGarmentType(category: string): string {
    const c = category.toLowerCase();
    if (c === 'bottoms') return 'lower_body';
    if (c === 'tops' || c === 'outerwear') return 'upper_body';
    if (c === 'dresses') return 'dresses';
    if (c === 'shoes') return 'shoes';
    if (c === 'outfits') return 'outfit';
    return 'upper_body';
}

function normalizeRow(
    raw: Record<string, string>,
    source: string,
    defaultSortBase: number
): Row | null {
    const get = (k: string) => {
        const key = Object.keys(raw).find((x) => normalizeHeader(x) === k);
        return key ? String(raw[key] ?? '').trim() : '';
    };

    let id = get('id');
    const external_id = get('external_id') || null;
    const brand = get('brand') || 'Unknown';
    const name = get('name') || get('title');
    const image_url = get('image_url') || get('image_link') || get('image');
    if (!name || !image_url) return null;

    let category = (get('category') || 'tops').toLowerCase();
    if (!CATEGORIES.has(category)) category = 'tops';

    let garment_type = (get('garment_type') || '').toLowerCase();
    if (!garment_type || !GARMENT_TYPES.has(garment_type)) {
        garment_type = mapCategoryToGarmentType(category);
    }

    const price = parsePrice(get('price'));
    const currency = (get('currency') || 'USD').toUpperCase().slice(0, 3);
    const description = get('description') || '';
    const source_url = get('source_url') || get('link') || get('product_url') || '';
    const sortOrderRaw = get('sort_order');
    const sort_order = sortOrderRaw ? parseInt(sortOrderRaw, 10) || defaultSortBase : defaultSortBase;

    if (!id) id = stableId(source, external_id ?? undefined, name, image_url);

    const now = new Date().toISOString();
    return {
        id,
        brand,
        name,
        price,
        currency,
        category,
        garment_type,
        description,
        image_url,
        source_url,
        sort_order,
        source,
        external_id: external_id && external_id.length ? external_id : null,
        is_active: true,
        last_seen_at: now,
        updated_at: now,
    };
}

function parseCsv(filePath: string, source: string): Row[] {
    const text = fs.readFileSync(filePath, 'utf8');
    const records = parse(text, {
        columns: (header: string[]) => header.map((h) => normalizeHeader(h)),
        skip_empty_lines: true,
        trim: true,
        relax_column_count: true,
    }) as Record<string, string>[];

    const rows: Row[] = [];
    let i = 0;
    for (const rec of records) {
        const row = normalizeRow(rec as unknown as Record<string, string>, source, i * 10);
        if (row) rows.push(row);
        i++;
    }
    return rows;
}

/** Flatten Google Shopping / RSS item: last wins for duplicate local names (e.g. g:id → id). */
function flattenXmlProduct(obj: unknown): Record<string, string> {
    const out: Record<string, string> = {};
    const visit = (o: unknown) => {
        if (o === null || o === undefined) return;
        if (typeof o === 'string' || typeof o === 'number') return;
        if (Array.isArray(o)) {
            for (const el of o) visit(el);
            return;
        }
        if (typeof o !== 'object') return;
        for (const [k, v] of Object.entries(o as Record<string, unknown>)) {
            const key = (k.includes(':') ? k.split(':').pop()! : k).toLowerCase();
            if (typeof v === 'string' || typeof v === 'number') {
                out[key] = String(v);
            } else if (Array.isArray(v) && v[0] != null && (typeof v[0] === 'string' || typeof v[0] === 'number')) {
                out[key] = String(v[0]);
            } else if (v && typeof v === 'object') {
                visit(v);
            }
        }
    };
    visit(obj);
    return out;
}

function categoryFromProductType(raw: string): string {
    const s = raw.toLowerCase();
    if (/shoe|boot|sneaker|loafer|heel/.test(s)) return 'shoes';
    if (/dress|gown|jumpsuit/.test(s)) return 'dresses';
    if (/pant|trouser|jean|short|skirt/.test(s)) return 'bottoms';
    if (/coat|jacket|blazer|outerwear|parka/.test(s)) return 'outerwear';
    if (/suit/.test(s)) return 'outfits';
    return 'tops';
}

function parseGoogleShoppingXml(filePath: string, source: string): Row[] {
    const xml = fs.readFileSync(filePath, 'utf8');
    const parser = new XMLParser({
        ignoreAttributes: false,
        attributeNamePrefix: '@_',
        isArray: (name) => name === 'item' || name === 'entry',
    });
    const doc = parser.parse(xml);

    let items: unknown[] = [];
    if (doc?.rss?.channel?.item) {
        items = Array.isArray(doc.rss.channel.item) ? doc.rss.channel.item : [doc.rss.channel.item];
    } else if (doc?.feed?.entry) {
        items = Array.isArray(doc.feed.entry) ? doc.feed.entry : [doc.feed.entry];
    } else if (Array.isArray(doc?.products?.product)) {
        items = doc.products.product;
    } else if (doc?.products?.product) {
        items = [doc.products.product];
    }

    const rows: Row[] = [];
    let i = 0;
    for (const item of items) {
        const f = flattenXmlProduct(item);
        const productType = f.product_type || f.type || '';
        const raw: Record<string, string> = {
            external_id: f.id || '',
            name: f.title || '',
            description: f.description || '',
            brand: f.brand || '',
            price: f.price || '',
            currency: '',
            image_url: f.image_link || f.image || '',
            source_url: f.link || '',
            category: productType ? categoryFromProductType(productType) : 'tops',
        };
        if (raw.price.includes(' ')) {
            const parts = raw.price.split(/\s+/);
            raw.price = parts[0];
            if (parts[1]) raw.currency = parts[1];
        }
        const row = normalizeRow(raw, source, i * 10);
        if (row) rows.push(row);
        i++;
    }
    return rows;
}

async function upsertRows(client: SupabaseClient | null, rows: Row[], dryRun: boolean): Promise<void> {
    if (dryRun) {
        console.log(`[dry-run] Would upsert ${rows.length} rows (showing first 3):`);
        console.log(JSON.stringify(rows.slice(0, 3), null, 2));
        return;
    }

    if (!client) {
        console.error('Missing Supabase client.');
        process.exit(1);
    }

    const batchSize = 100;
    for (let i = 0; i < rows.length; i += batchSize) {
        const batch = rows.slice(i, i + batchSize);
        const { error } = await client.from('shop_catalog').upsert(batch, { onConflict: 'id' });
        if (error) {
            console.error(`Upsert batch ${i / batchSize + 1} failed:`, error.message);
            process.exit(1);
        }
        console.log(`  Upserted ${Math.min(i + batchSize, rows.length)} / ${rows.length}`);
    }
}

async function main(): Promise<void> {
    const file = arg('--file');
    const source = arg('--source');
    const dryRun = hasFlag('--dry-run');

    if (!file || !source) {
        console.error(
            'Usage: npx tsx scripts/import-feed.ts --file <path.csv|path.xml> --source <label> [--dry-run]'
        );
        process.exit(1);
    }

    const abs = path.resolve(process.cwd(), file);
    if (!fs.existsSync(abs)) {
        console.error('File not found:', abs);
        process.exit(1);
    }

    const ext = path.extname(abs).toLowerCase();
    let rows: Row[];
    if (ext === '.csv' || ext === '.tsv') {
        rows = parseCsv(abs, source);
    } else if (ext === '.xml') {
        rows = parseGoogleShoppingXml(abs, source);
    } else {
        console.error('Unsupported extension. Use .csv, .tsv, or .xml');
        process.exit(1);
    }

    console.log(`Parsed ${rows.length} rows from ${path.basename(abs)} (source=${source})`);

    const url = process.env.SUPABASE_URL;
    const key = process.env.SUPABASE_SERVICE_KEY;
    if (!dryRun && (!url || !key)) {
        console.error('Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or use --dry-run).');
        process.exit(1);
    }

    const client = !dryRun && url && key ? createClient(url, key) : null;
    await upsertRows(client, rows, dryRun);
    console.log(rows.length ? 'Done.' : 'Nothing to import.');
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
