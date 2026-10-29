export function isDressGarment(item: { name?: string; type?: string; macroCategory?: string } | null | undefined): boolean {
    if (!item) return false;
    const name = `${item.name || ''} ${item.type || ''}`.toLowerCase();
    // “Dress” describes formality in footwear names; it does not make
    // dress penny loafers a garment replacing both top and bottom.
    if (/\b(shoes?|loafers?|boots?|sneakers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|mules?)\b/.test(name)) return false;
    const slot = String(item.macroCategory || '').toLowerCase();
    if (slot && slot !== 'top' && slot !== 'dress') return false;
    if (slot === 'dress') return true;
    return /\bdress(?:es)?\b|плать[ея]/.test(name)
        && !/\bdress[ -](shirts?|shoes?|pants?|trousers?)\b/.test(name);
}

/** Known non-garment catalog products cannot occupy any clothing slot. */
export function isNonOutfitItem(item: { name?: unknown; type?: unknown } | null | undefined): boolean {
    const name = typeof item?.name === 'string' ? item.name.toLowerCase() : '';
    const type = typeof item?.type === 'string' ? item.type.toLowerCase() : '';
    const accessory = /\b(watch(?:es)?|towels?|(?:suit|garment)[ -]+carriers?|key[ -]?rings?|keychains?|(?:hand|tote)?bags?|backpacks?|wallets?|card[ -]?holders?|purses?|belts?|scar(?:f|ves)|hats?|caps?|beanies?|jewell?ery|sunglasses|gloves?|cufflinks?|pocket[ -]?squares?|necklaces?|earrings?|bracelets?|rings?|umbrellas?|socks?|cases?|pouches?|perfumes?|fragrances?)\b/.test(`${name} ${type}`);
    // A towel-knit shirt or watch-print shirt is still a garment. Only
    // product names without an actual clothing noun are rejected here.
    const garment = /\b(t-?shirts?|tees?|shirts?|polos?|blouses?|dress(?:es)?|pants?|trousers?|jeans|shorts?|skirts?|chinos?|jackets?|coats?|blazers?|cardigans?|sweaters?|hoodies?|vests?|shoes?|loafers?|boots?|sneakers?|sandals?)\b/.test(name);
    return accessory && !garment;
}

/** Shared wire contract. Validation never repairs or substitutes a product. */
export function outfitStructureReason(items: unknown, requireLayer = false): string | null {
    if (!Array.isArray(items) || !items.length) return 'empty_outfit';
    const ids = new Set<string>();
    const slots = new Map<string, number>();
    let dresses = 0;
    for (const item of items) {
        if (!item || typeof item !== 'object') return 'invalid_item';
        if (isNonOutfitItem(item)) return 'non_outfit_item';
        if (typeof item.id !== 'string' || !item.id.trim() || /^(placeholder_|item_|local_)/.test(item.id)) return 'invalid_item_id';
        if (ids.has(item.id)) return 'duplicate_item';
        ids.add(item.id);
        const slot = item.macroCategory;
        if (!['top', 'bottom', 'shoes', 'outerwear', 'dress'].includes(slot)) return 'invalid_slot';
        const dress = (slot === 'dress' || slot === 'top') && isDressGarment(item);
        if (dress) dresses++;
        const normalized = slot === 'dress' ? 'top' : slot;
        slots.set(normalized, (slots.get(normalized) || 0) + 1);
    }
    if ([...slots.values()].some(count => count !== 1)) return 'duplicate_slot';
    if (!slots.has('shoes') || !slots.has('top')) return 'missing_slot';
    if (dresses ? slots.has('bottom') : !slots.has('bottom')) return dresses ? 'dress_with_bottom' : 'missing_slot';
    if (requireLayer && !slots.has('outerwear')) return 'missing_layer';
    return null;
}

/** Only identifiable shirts/T-shirts and pants can fill those categories. */
export function isFourSlotCandidate(item: {macroCategory?: string; name?: string; type?: string}): boolean {
    const text = `${item.name || ''} ${item.type || ''}`.toLowerCase();
    const name = String(item.name || '').toLowerCase();
    if (isNonOutfitItem(item) || isDressGarment(item)) return false;
    if (item.macroCategory === 'top') {
        // A stale `type: shirt` field must not override a product title that
        // identifies a plain knit garment. Knit polos and knit shirts remain
        // valid because their names identify one of the allowed base-top types.
        if (/\bknit(?:ted|wear)?\b/.test(name) && !/\b(?:polos?|shirts?|t-?shirts?|tees?)\b/.test(name)) return false;
        if (/\b(sweaters?|pullovers?|jumpers?|cardigans?|hoodies?|jackets?|blazers?|coats?|vests?|sweatshirts?)\b|свитер|джемпер|кардиган|толстовк|куртк|пиджак/.test(text)) return false;
        return /\b(t[ -]?shirts?|tees?|shirts?|polos?)\b|рубашк|футболк|поло/.test(text);
    }
    if (item.macroCategory === 'bottom') {
        // Product/color names can contain Jeans; a named shirt is never pants.
        if (/\b(t[ -]?shirts?|tees?|shirts?|polos?|blouses?|sweaters?|pullovers?|cardigans?|jackets?|blazers?|coats?)\b|рубашк|футболк|поло|свитер|куртк|пиджак/.test(text)) return false;
        if (/\b(shorts?|bermudas?|skirts?|trunks?|swimwear)\b|шорт|юбк/.test(text)) return false;
        return /\b(pants?|trousers?|jeans|chinos?|joggers?|slacks?)\b|брюк|джинс|чинос/.test(text);
    }
    return item.macroCategory === 'outerwear' || item.macroCategory === 'shoes';
}

/** App generation contract: exactly one layer, base shirt, pants and shoes. */
export function fourSlotOutfitReason(items: unknown, _requireLayer = true): string | null {
    const structural = outfitStructureReason(items, true);
    if (structural) return structural;
    const parts = items as Array<{macroCategory: string; name?: string; type?: string}>;
    if (parts.length !== 4 || parts.some(isDressGarment)) return 'requires_four_categories';
    if (!parts.every(isFourSlotCandidate)) return 'invalid_required_category';
    const top = parts.find(item => item.macroCategory === 'top')!;
    const bottom = parts.find(item => item.macroCategory === 'bottom')!;
    const topName = `${top.name || ''} ${top.type || ''}`.toLowerCase();
    if (/\b(sweaters?|pullovers?|jumpers?|cardigans?|hoodies?|jackets?|blazers?|coats?|vests?)\b|свитер|джемпер|кардиган|толстовк|куртк|пиджак/.test(topName)) return 'base_top_must_be_shirt';
    const bottomName = `${bottom.name || ''} ${bottom.type || ''}`.toLowerCase();
    if (/\b(shorts?|bermudas?|skirts?|trunks?|swimwear)\b|шорт|юбк/.test(bottomName)) return 'bottom_must_be_pants';
    return null;
}
