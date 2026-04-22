export interface CatalogOrderItem {
    id: string;
    name: string;
    garmentType?: string | null;
}

const RECENT_SIMILARITY_WINDOW = 2;

const SIMILARITY_REPLACEMENTS: Array<[RegExp, string]> = [
    [/\bt[\s-]?shirts?\b/g, ' tee '],
    [/\btees?\b/g, ' tee '],
    [/\bslim[\s-]?fit\b/g, ' slimfit '],
    [/\brelaxed[\s-]?fit\b/g, ' relaxedfit '],
    [/\bloose[\s-]?fit\b/g, ' loosefit '],
    [/\bregular[\s-]?fit\b/g, ' regularfit '],
    [/\bv[\s-]?neck\b/g, ' vneck '],
    [/\bcrew[\s-]?neck\b/g, ' crewneck '],
    [/\bmock[\s-]?neck\b/g, ' mockneck '],
    [/\bbutton[\s-]?down\b/g, ' buttondown '],
    [/\blong[\s-]?sleeve(?:d)?\b/g, ' longsleeve '],
    [/\bshort[\s-]?sleeve(?:d)?\b/g, ' shortsleeve '],
    [/\bheavy[\s-]?weight\b/g, ' heavyweight '],
    [/\bmedium[\s-]?weight\b/g, ' mediumweight '],
    [/\blight[\s-]?weight\b/g, ' lightweight '],
];

const COLOR_WORDS = new Set([
    'black', 'white', 'offwhite', 'ecru', 'cream', 'beige', 'stone', 'sand',
    'taupe', 'camel', 'khaki', 'olive', 'green', 'sage', 'mint',
    'blue', 'navy', 'indigo', 'brown', 'chocolate', 'grey', 'gray',
    'silver', 'charcoal', 'red', 'burgundy', 'maroon', 'pink',
    'purple', 'lilac', 'yellow', 'mustard', 'orange', 'rust',
]);

const LOW_SIGNAL_WORDS = new Set([
    'basic', 'classic', 'essential', 'premium',
    'soft', 'touch', 'washed', 'faded', 'heavyweight', 'mediumweight', 'lightweight',
    'regularfit', 'relaxedfit', 'slimfit', 'loosefit',
    'textured', 'structured', 'comfort', 'interlock', 'stretch',
    'cotton', 'viscose', 'polyester', 'polyamide', 'elastane',
    'lyocell', 'modal', 'jersey',
    'fit', 'detail', 'details', 'edition',
    'men', 'man', 'zara',
]);

const TOKEN_ALIASES = new Map([
    ['tshirt', 'tee'],
    ['tee', 'tee'],
    ['tees', 'tee'],
    ['trainer', 'sneaker'],
    ['trainers', 'sneaker'],
    ['sweater', 'knit'],
    ['jumper', 'knit'],
]);

function normalizeText(value: string): string {
    return value
        .toLowerCase()
        .replace(/https?:\/\//g, '')
        .replace(/[^a-z0-9]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function normalizeSimilaritySource(value: string): string {
    let text = value.toLowerCase();

    for (const [pattern, replacement] of SIMILARITY_REPLACEMENTS) {
        text = text.replace(pattern, replacement);
    }

    return text;
}

function singularizeToken(token: string): string {
    if (token.endsWith('ies') && token.length > 4) {
        return `${token.slice(0, -3)}y`;
    }

    if (token.endsWith('s') && token.length > 3 && !token.endsWith('ss')) {
        return token.slice(0, -1);
    }

    return token;
}

function getSimilarityTokens(item: CatalogOrderItem): string[] {
    const normalized = normalizeText(normalizeSimilaritySource(item.name));
    const seen = new Set<string>();
    const tokens: string[] = [];

    for (const rawToken of normalized.split(' ')) {
        if (!rawToken || /^\d+$/.test(rawToken)) continue;

        const singular = singularizeToken(rawToken);
        const token = TOKEN_ALIASES.get(singular) ?? singular;

        if (token.length < 2) continue;
        if (COLOR_WORDS.has(token) || LOW_SIGNAL_WORDS.has(token)) continue;
        if (seen.has(token)) continue;

        seen.add(token);
        tokens.push(token);
    }

    return tokens;
}

export function buildCatalogSimilarityKey(item: CatalogOrderItem): string {
    const garmentType = String(item.garmentType || 'item').trim().toLowerCase() || 'item';
    const tokens = getSimilarityTokens(item);
    return `${garmentType}:${tokens.join(' ') || garmentType}`;
}

export function spreadSimilarCatalogItems<T extends CatalogOrderItem>(
    items: T[],
    recentItems: CatalogOrderItem[] = [],
): T[] {
    if (items.length <= 1) return items;

    const recentKeys = recentItems
        .slice(-RECENT_SIMILARITY_WINDOW)
        .map(buildCatalogSimilarityKey);
    const remaining = items.map((item) => ({
        item,
        similarityKey: buildCatalogSimilarityKey(item),
    }));
    const ordered: typeof remaining = [];

    while (remaining.length > 0) {
        let nextIndex = remaining.findIndex(
            (candidate) => !recentKeys.includes(candidate.similarityKey),
        );

        if (nextIndex === -1) {
            const lastKey = recentKeys[recentKeys.length - 1];
            nextIndex = remaining.findIndex(
                (candidate) => candidate.similarityKey !== lastKey,
            );
        }

        if (nextIndex === -1) {
            nextIndex = 0;
        }

        const [nextItem] = remaining.splice(nextIndex, 1);
        ordered.push(nextItem);
        recentKeys.push(nextItem.similarityKey);

        if (recentKeys.length > RECENT_SIMILARITY_WINDOW) {
            recentKeys.shift();
        }
    }

    return ordered.map(({ item }) => item);
}
