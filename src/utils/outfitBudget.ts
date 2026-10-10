/**
 * Outfit budget — parse a spend cap from user text and score shop spend.
 *
 * Owned wardrobe pieces cost 0. Only catalog/shop items with a numeric
 * `price` count toward the cap.
 */

export interface ParsedBudget {
    amount: number;
    currency: string;
}

const CURRENCY_FROM_TOKEN: Record<string, string> = {
    $: 'USD',
    usd: 'USD',
    dollar: 'USD',
    dollars: 'USD',
    '€': 'EUR',
    eur: 'EUR',
    euro: 'EUR',
    euros: 'EUR',
    '£': 'GBP',
    gbp: 'GBP',
    pound: 'GBP',
    pounds: 'GBP',
};

const UNIT_FOLLOW = /^(?:\s*(?:°\s*)?(?:c|f|cm|kg|lbs?|items?|pieces?|degrees?)\b)/i;

function tokenToCurrency(token?: string | null): string {
    if (!token) return 'USD';
    return CURRENCY_FROM_TOKEN[token.trim().toLowerCase()] || CURRENCY_FROM_TOKEN[token.trim()] || 'USD';
}

function parseAmount(raw: string): number | null {
    const n = Number(String(raw).replace(',', '.'));
    if (!Number.isFinite(n) || n < 1 || n > 100_000) return null;
    return Math.round(n * 100) / 100;
}

/**
 * Extract a max spend from free text: "under $150", "250$", "$250",
 * "budget 80 euros", "Work: 250$". Ignores weather/size units ("under 20 C").
 */
export function parseBudgetFromText(text?: string | null): ParsedBudget | null {
    if (!text) return null;
    const src = String(text);
    const found: ParsedBudget[] = [];

    const consider = (amountRaw: string, currRaw: string, after: string) => {
        if (UNIT_FOLLOW.test(after)) return;
        const amount = parseAmount(amountRaw);
        if (amount == null) return;
        found.push({ amount, currency: tokenToCurrency(currRaw) });
    };

    const prefix = /([$€£])\s*(\d+(?:[.,]\d{1,2})?)/g;
    let m: RegExpExecArray | null;
    while ((m = prefix.exec(src)) !== null) {
        consider(m[2], m[1], src.slice(m.index + m[0].length));
    }

    const suffix = /(\d+(?:[.,]\d{1,2})?)\s*([$€£]|usd|eur|gbp|dollars?|euros?|pounds?)/gi;
    while ((m = suffix.exec(src)) !== null) {
        consider(m[1], m[2], src.slice(m.index + m[0].length));
    }

    const keyword = /\b(?:budget(?:\s*(?:of|is|:))?|under|below|within|up\s*to|at\s*most|no\s+more\s+than|less\s+than|max(?:imum)?(?:\s+of)?)\s*([$€£]|usd|eur|gbp)?\s*(\d+(?:[.,]\d{1,2})?)/gi;
    while ((m = keyword.exec(src)) !== null) {
        consider(m[2], m[1] || '', src.slice(m.index + m[0].length));
    }

    if (found.length === 0) return null;
    return found.reduce((best, cur) => (cur.amount < best.amount ? cur : best));
}

/**
 * Remove budget tokens from free text while keeping style/occasion words:
 * "Work: work, old money, $250" -> "Work: work, old money".
 *
 * The deployed generate-outfits edge parses a budget from the prompt and
 * compares it against raw UZS prices with no currency conversion, so a
 * "$250" token makes it drop the whole shop pool and fail with
 * "No clothing items found". Budgets are enforced client-side (fitsCap +
 * the normalizer's clamp), so the edge must simply never see one.
 */
export function stripBudgetFromText(text?: string | null): string {
    if (!text) return '';
    let out = String(text);
    // Keyword forms first — they own their own optional currency + amount
    // ("under $250", "budget of 250 euros", "up to 300").
    out = out.replace(
        /\b(?:budget(?:\s*(?:of|is|:))?|under|below|within|up\s+to|at\s+most|no\s+more\s+than|less\s+than|max(?:imum)?(?:\s+of)?)\s*([$€£]|usd|eur|gbp)?\s*\d+(?:[.,]\d{1,2})?(?:\s*(?:usd|eur|gbp|dollars?|euros?|pounds?))?/gi,
        ' ',
    );
    // Standalone prefix forms: "$250", "€30".
    out = out.replace(/([$€£])\s*\d+(?:[.,]\d{1,2})?/g, ' ');
    // Suffix forms: "250$", "250 USD", "250 dollars".
    out = out.replace(/\d+(?:[.,]\d{1,2})?\s*([$€£]|usd|eur|gbp|dollars?|euros?|pounds?)/gi, ' ');
    return out
        .replace(/\s{2,}/g, ' ')
        .replace(/\s+([,.;:!?])/g, '$1')
        .replace(/^[,.;:\s]+/, '')
        .trim();
}

export function isCatalogShopId(id?: string | null): boolean {
    const s = String(id || '');
    return /^(shop_|md-|zara-|classic-m-)/i.test(s);
}

/**
 * Units per 1 USD — used ONLY to express a text budget in the currency the
 * catalog actually lists. shop_catalog rows are UZS (median ~1.2M so'm), so
 * comparing a parsed "$250" against a UZS price rejected EVERY item: budget
 * scenarios fell back to placeholder Pants/Shoes tiles and dropped every
 * shop look. Override with EXPO_PUBLIC_BUDGET_USD_UZS when rates drift.
 */
const UNITS_PER_USD: Record<string, number> = {
    USD: 1,
    EUR: 1.08,
    GBP: 0.79,
    UZS: (() => {
        const v = Number(process.env.EXPO_PUBLIC_BUDGET_USD_UZS);
        return Number.isFinite(v) && v > 0 ? v : 12600;
    })(),
};

/** First non-empty item currency, upper-cased (e.g. 'UZS'). */
export function catalogCurrencyOf(
    items: Array<{ currency?: string | null } | null | undefined> | null | undefined,
): string | null {
    for (const it of items || []) {
        const c = String(it?.currency || '').trim().toUpperCase();
        if (c) return c;
    }
    return null;
}

/**
 * Express `budget` in `targetCurrency` so a cap can be compared with catalog
 * prices. Same-currency / unknown-currency pairs are returned untouched, so
 * calling it again with an already-converted budget is a no-op (clients and
 * the edge both call it — no double conversion).
 */
export function budgetInCurrency(
    budget: ParsedBudget | null | undefined,
    targetCurrency?: string | null,
): ParsedBudget | null {
    if (!budget) return null;
    const to = String(targetCurrency || '').trim().toUpperCase();
    const from = String(budget.currency || 'USD').trim().toUpperCase();
    if (!to || to === from) return { ...budget, currency: from };
    const toPerUsd = UNITS_PER_USD[to];
    const fromPerUsd = UNITS_PER_USD[from];
    if (!toPerUsd || !fromPerUsd) return { ...budget, currency: from };
    const converted = (budget.amount * toPerUsd) / fromPerUsd;
    return {
        amount: to === 'UZS' ? Math.round(converted) : Math.round(converted * 100) / 100,
        currency: to,
    };
}

export function isShopPricedItem(item: { isShopItem?: boolean; isWardrobe?: boolean; id?: string } | null | undefined): boolean {
    if (!item) return false;
    if (item.isWardrobe === true && !item.isShopItem) return false;
    return Boolean(item.isShopItem || item.isWardrobe === false || isCatalogShopId(item.id));
}

/** Shop/catalog spend for one piece. Owned / unpriced items are 0. */
export function itemShopPrice(item: { price?: number | null; isShopItem?: boolean; isWardrobe?: boolean; id?: string } | null | undefined): number {
    if (!item || !isShopPricedItem(item)) return 0;
    const n = Number(item.price);
    return Number.isFinite(n) && n > 0 ? n : 0;
}

/** Resolve a listed price from the item, then from a catalog by matching ids. */
export function resolveShopPrice(
    item: { price?: number | null; isShopItem?: boolean; isWardrobe?: boolean; id?: string } | null | undefined,
    catalog?: Array<{ id?: string; price?: number | null; isShopItem?: boolean; isWardrobe?: boolean }>,
): number {
    const direct = itemShopPrice(item);
    if (direct > 0) return direct;
    const id = String(item?.id || '');
    if (!id || !catalog?.length) return 0;
    const bare = id.replace(/^shop_/i, '');
    const alts = new Set([id, bare, `shop_${bare}`]);
    for (const row of catalog) {
        if (!alts.has(String(row?.id || ''))) continue;
        const listed = Number(row.price);
        if (Number.isFinite(listed) && listed > 0) return listed;
        const priced = itemShopPrice({ ...row, isShopItem: true });
        if (priced > 0) return priced;
    }
    return 0;
}

export function outfitShopSpend(
    items: Array<{ price?: number | null; isShopItem?: boolean; isWardrobe?: boolean; id?: string } | null | undefined>,
    catalog?: Array<{ id?: string; price?: number | null; isShopItem?: boolean; isWardrobe?: boolean }>,
): number {
    return (items || []).reduce((sum, item) => sum + resolveShopPrice(item, catalog), 0);
}

export function formatMoney(amount: number, currency = 'USD'): string {
    const n = Number.isFinite(amount) ? amount : 0;
    const rounded = Number.isInteger(n) ? String(n) : n.toFixed(2);
    if (currency === 'EUR') return `€${rounded}`;
    if (currency === 'GBP') return `£${rounded}`;
    if (currency === 'USD') return `$${rounded}`;
    if (currency === 'UZS') {
        // Full so'm figures need grouping to be readable on a tile.
        return `${rounded.replace(/\B(?=(\d{3})+(?!\d))/g, ' ')} UZS`;
    }
    return `${rounded} ${currency}`;
}

export function budgetInstruction(budget: ParsedBudget | null): string {
    if (!budget) {
        return 'When recommending shop/catalog pieces, name their listed prices. If the user gives a budget, the shop items in a look must sum to that amount or less (owned wardrobe = 0).';
    }
    return `HARD BUDGET: ${formatMoney(budget.amount, budget.currency)}. Owned wardrobe items cost 0. Sum ONLY shop/catalog prices. Never recommend a look whose shop total exceeds this cap. Prefer cheaper shop substitutes over dropping a required slot. If nothing in a slot fits, say so and keep the rest under budget.`;
}

function itemLabel(item: {
    name?: string;
    brand?: string;
    primaryColor?: string;
    color?: string;
    subCategory?: string;
    category?: string;
}): string {
    const color = item.primaryColor || item.color || '';
    const name = item.name || item.subCategory || item.category || 'item';
    return [item.brand, color, name].filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
}

/** Compact clothing context for the chat stylist (owned + priced shop). */
export function buildAgentClothingContext(opts: {
    owned?: Array<Record<string, any>>;
    shop?: Array<Record<string, any>>;
    userMessage?: string;
    maxOwned?: number;
    maxShop?: number;
}): string {
    const pricedShop = (opts.shop || []).filter((i) => typeof i?.price === 'number' && i.price > 0);
    // Compare the parsed cap in the catalog's own currency (UZS rows vs a
    // "$250" prompt) so cheap pieces are actually promoted.
    const budget = budgetInCurrency(parseBudgetFromText(opts.userMessage), catalogCurrencyOf(pricedShop));
    const maxOwned = opts.maxOwned ?? 12;
    const maxShop = opts.maxShop ?? 20;
    const owned = (opts.owned || []).slice(0, maxOwned);
    let shop = pricedShop;
    if (budget) {
        const under = shop.filter((i) => Number(i.price) <= budget.amount);
        shop = (under.length > 0 ? under : shop).slice().sort((a, b) => Number(a.price) - Number(b.price));
    }
    shop = shop.slice(0, maxShop);

    const ownedBit = owned.length
        ? `Owned wardrobe (cost 0): ${owned.map((i) => itemLabel(i)).filter(Boolean).join('; ')}`
        : '';
    const shopBit = shop.length
        ? `Shop catalog with prices: ${shop.map((i) => {
            const curr = String(i.currency || budget?.currency || 'USD').toUpperCase();
            return `${itemLabel(i)} ${formatMoney(Number(i.price), curr)}`;
        }).join('; ')}`
        : '';
    return [ownedBit, shopBit, budgetInstruction(budget)].filter(Boolean).join('\n');
}
