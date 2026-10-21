import { budgetCandidatesFromText, stripBudgetText } from '../../backend/supabase/functions/_shared/outfitBudgetText';
export { budgetCandidatesFromText } from '../../backend/supabase/functions/_shared/outfitBudgetText';
import { type BudgetFxContext, fitsBudgetAmount, referenceAmount } from '../../backend/supabase/functions/_shared/budgetFx';
/**
 * Outfit budget — parse a spend cap from user text and score shop spend.
 *
 * Owned wardrobe pieces cost 0. Only catalog/shop items with a numeric
 * `price` count toward the cap.
 */

export interface ParsedBudget {
    amount: number;
    currency: string;
    fx?: BudgetFxContext;
}

/** Preserve the spending cap when budget wording is stripped from the prompt. */
export function withOutfitBudget<T extends object>(preferences: T, budget?: ParsedBudget | null): T & { maxShopBudget?: ParsedBudget; budgetRatesId?: string } {
    return budget ? { ...preferences, maxShopBudget: { amount: budget.amount, currency: budget.currency }, ...(budget.fx ? { budgetRatesId: budget.fx.snapshot.id } : {}) } : { ...preferences };
}

/** Mixed-currency limits require a verified reference snapshot before comparison. */
export function parseBudgetFromText(text?: string | null, fx?: BudgetFxContext): ParsedBudget | null {
    const found = budgetCandidatesFromText(text);
    if (!found.length || (!fx && new Set(found.map(b => b.currency)).size > 1)) return null;
    const best = found.reduce((best, cur) => {
        const comparable = convertShopAmount(cur.amount, cur.currency, best.currency, fx);
        return comparable < best.amount ? cur : best;
    });
    return { ...best, ...(fx ? {fx} : {}) };
}

/**
 * Remove budget tokens from free text while keeping style/occasion words:
 * "Work: work, old money, $250" -> "Work: work, old money".
 *
 * The spending cap is sent separately in preferences for server enforcement.
 */
export function stripBudgetFromText(text?: string | null): string { return stripBudgetText(text); }

export function isCatalogShopId(id?: string | null): boolean {
    const s = String(id || '');
    return /^(shop_|md-|zara-|classic-m-)/i.test(s);
}

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
 * prices using the pinned reference snapshot. Missing conversion data returns null.
 */
export function budgetInCurrency(
    budget: ParsedBudget | null | undefined,
    targetCurrency?: string | null,
): ParsedBudget | null {
    if (!budget) return null;
    const to = String(targetCurrency || '').trim().toUpperCase();
    const from = String(budget.currency || 'USD').trim().toUpperCase();
    if (!to || to === from) return { ...budget, currency: from };
    const converted = convertShopAmount(budget.amount, from, to, budget.fx);
    return Number.isFinite(converted) ? { ...budget, amount: converted, currency: to } : null;
}

export function isShopPricedItem(item: { isShopItem?: boolean; isWardrobe?: boolean; id?: string } | null | undefined): boolean {
    if (!item) return false;
    if (item.isWardrobe === true && !item.isShopItem) return false;
    return Boolean(item.isShopItem || item.isWardrobe === false || isCatalogShopId(item.id));
}

/** Convert configured monetary units; an unknown pair cannot pass a spending cap. */
export function convertShopAmount(amount: number, from: string | null | undefined, to: string, fx?: BudgetFxContext): number {
    const source = String(from || '').trim().toUpperCase();
    const target = String(to).trim().toUpperCase();
    if (source === target) return Number.isFinite(amount) && amount >= 0 ? amount : Infinity;
    if (!fx) return Number.POSITIVE_INFINITY;
    return referenceAmount(amount, source, target, fx.snapshot, fx.checkedAt);
}

/** Owned pieces cost zero; an unverified shop price cannot pass an explicit cap. */
export function itemShopPrice(item: { price?: number | null; isShopItem?: boolean; isWardrobe?: boolean; id?: string; currency?: string | null } | null | undefined, targetCurrency?: string | null, fx?: BudgetFxContext): number {
    if (!item || !isShopPricedItem(item)) return 0;
    const n = Number(item.price);
    return Number.isFinite(n) && n > 0 ? (targetCurrency ? convertShopAmount(n, item.currency, targetCurrency, fx) : n) : (targetCurrency ? Number.POSITIVE_INFINITY : 0);
}

export interface ListedShopQuote { price: number; currency: string | null }
type QuoteItem = { id?: string; price?: number | null; currency?: string | null; isShopItem?: boolean; isWardrobe?: boolean };

/** Keep a price and its currency from the same snapshot or matching catalog row. */
export function resolveShopQuote(item: QuoteItem | null | undefined, catalog?: QuoteItem[]): ListedShopQuote | null {
    if (!item || (item.isWardrobe === true && !item.isShopItem)) return null;
    const currencyOf = (value: unknown): string | null => {
        const currency = typeof value === 'string' ? value.trim().toUpperCase() : '';
        return /^[A-Z]{3}$/.test(currency) ? currency : null;
    };
    const direct = itemShopPrice(item), currency = currencyOf(item.currency);
    if (direct > 0 && currency) return { price: direct, currency };
    const id = String(item.id || ''), bare = id.replace(/^shop_/i, '');
    const ids = new Set([id, bare, `shop_${bare}`]);
    if (id) for (const row of catalog || []) {
        if (!ids.has(String(row?.id || ''))) continue;
        if (row.isWardrobe === true && !row.isShopItem) return null;
        const price = Number(row.price), rowCurrency = currencyOf(row.currency);
        if (Number.isFinite(price) && price > 0 && rowCurrency) return { price, currency: rowCurrency };
    }
    // An unlabelled amount is retained as unknown, never relabelled USD or
    // paired with a currency taken from a different product.
    return direct > 0 ? { price: direct, currency: null } : null;
}

/** Resolve a listed price from the item, then from a catalog by matching ids. */
export function resolveShopPrice(
    item: QuoteItem | null | undefined,
    catalog?: QuoteItem[],
    targetCurrency?: string | null,
    fx?: BudgetFxContext,
): number {
    if (!item || (item.isWardrobe === true && !item.isShopItem)) return 0;
    const id = String(item.id || ''), bare = id.replace(/^shop_/i, '');
    const ids = new Set([id, bare, `shop_${bare}`]);
    const row = id ? catalog?.find(row => ids.has(String(row?.id || ''))) : null;
    if (row?.isWardrobe === true && !row.isShopItem && !isShopPricedItem(item)) return 0;
    if (targetCurrency) {
        const quote = resolveShopQuote(item, catalog);
        return quote ? convertShopAmount(quote.price, quote.currency, targetCurrency, fx) : (isShopPricedItem(item) || (row && !(row.isWardrobe === true && !row.isShopItem)) ? Number.POSITIVE_INFINITY : 0);
    }
    const direct = itemShopPrice(item);
    if (direct > 0) return direct;
    const listed = Number(row?.price);
    return Number.isFinite(listed) && listed > 0 ? listed : 0;
}

export function outfitShopSpend(
    items: Array<QuoteItem | null | undefined>,
    catalog?: QuoteItem[],
    targetCurrency?: string | null,
    fx?: BudgetFxContext,
): number {
    const currency = targetCurrency || catalogCurrencyOf(catalog) || catalogCurrencyOf(items);
    return (items || []).reduce((sum, item) => sum + resolveShopPrice(item, catalog, currency, fx), 0);
}

export function formatMoney(amount: number, currency = 'USD'): string {
    if (!Number.isFinite(amount)) return '—';
    const n = amount;
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
    budgetFx?: BudgetFxContext;
}): string {
    const pricedShop = (opts.shop || []).filter((i) => typeof i?.price === 'number' && i.price > 0);
    // Compare the parsed cap in the catalog's own currency (UZS rows vs a
    // "$250" prompt) so cheap pieces are actually promoted.
    const budget = parseBudgetFromText(opts.userMessage, opts.budgetFx);
    const maxOwned = opts.maxOwned ?? 12;
    const maxShop = opts.maxShop ?? 20;
    const owned = (opts.owned || []).slice(0, maxOwned);
    let shop = pricedShop;
    if (budget) {
        const under = shop.filter((i) => fitsBudgetAmount(itemShopPrice({ ...i, isShopItem: true }, budget.currency, budget.fx), budget.amount));
        shop = under.slice().sort((a, b) => itemShopPrice({ ...a, isShopItem: true }, budget.currency, budget.fx) - itemShopPrice({ ...b, isShopItem: true }, budget.currency, budget.fx));
    }
    shop = shop.slice(0, maxShop);

    const ownedBit = owned.length
        ? `Owned wardrobe (cost 0): ${owned.map((i) => itemLabel(i)).filter(Boolean).join('; ')}`
        : '';
    const shopBit = shop.length
        ? `Shop catalog with prices: ${shop.map((i) => {
            const curr = String(i.currency || '').trim().toUpperCase();
            const listed = /^[A-Z]{3}$/.test(curr) ? formatMoney(Number(i.price), curr) : 'price currency unavailable';
            const reference = budget?.fx && curr !== budget.currency
                ? ` (reference cost ${formatMoney(itemShopPrice({...i, isShopItem: true}, budget.currency, budget.fx), budget.currency)}; CBU ${budget.fx.snapshot.asOf})` : '';
            return `${itemLabel(i)} ${listed}${reference}`;
        }).join('; ')}`
        : '';
    return [ownedBit, shopBit, budgetInstruction(budget)].filter(Boolean).join('\n');
}
