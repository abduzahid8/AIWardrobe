/**
 * Gate F currency honesty for scripts/import-feed.ts.
 *
 * The old `(get('currency') || 'USD')` default invented a currency for every
 * row whose feed omitted the column, mislabeling e.g. UZS amounts as USD.
 * Currency must now be feed-provided (per-row column) or run-declared
 * (--currency); anything unresolvable is skipped and counted, never guessed.
 */
import { normalizeRow } from '../../scripts/import-feed';

const base = {
    name: 'Wool Blend Overcoat',
    image_url: 'https://example.com/images/overcoat.jpg',
    price: '849900',
};

const skips = (): Record<string, number> => ({});

describe('import-feed currency resolution', () => {
    it('trusts a valid per-row feed currency', () => {
        const row = normalizeRow({ ...base, currency: 'UZS' }, 'feed-x', 0);
        expect(row).not.toBeNull();
        expect(row!.currency).toBe('UZS');
        // Lowercase feed values normalize to the ISO code, not a new guess.
        const lower = normalizeRow({ ...base, currency: 'usd' }, 'feed-x', 0);
        expect(lower!.currency).toBe('USD');
    });

    it('skips and counts rows with no resolvable currency instead of defaulting', () => {
        const counted = skips();
        expect(normalizeRow({ ...base }, 'feed-x', 0, undefined, counted)).toBeNull();
        expect(counted.currency_unknown).toBe(1);
        // A garbled feed value is equally unresolvable.
        const garbled = skips();
        expect(normalizeRow({ ...base, currency: 'us' }, 'feed-x', 0, undefined, garbled)).toBeNull();
        expect(garbled.currency_unknown).toBe(1);
        // A truncated-by-slice guess (e.g. 'UZSX' → 'UZS') is still a guess.
        const sliced = skips();
        expect(normalizeRow({ ...base, currency: 'UZSX' }, 'feed-x', 0, undefined, sliced)).toBeNull();
        expect(sliced.currency_unknown).toBe(1);
    });

    it('uses the run-declared --currency when the feed omits the column', () => {
        const row = normalizeRow({ ...base }, 'feed-x', 0, 'UZS');
        expect(row).not.toBeNull();
        expect(row!.currency).toBe('UZS');
        // A per-row currency still outranks the run declaration.
        const mixed = normalizeRow({ ...base, currency: 'GBP' }, 'feed-x', 0, 'UZS');
        expect(mixed!.currency).toBe('GBP');
        // A garbled feed value falls back to the declaration, not to a guess.
        const garbled = normalizeRow({ ...base, currency: 'us' }, 'feed-x', 0, 'UZS');
        expect(garbled!.currency).toBe('UZS');
    });

    it('skips when neither the feed nor the run declares a usable currency', () => {
        expect(normalizeRow({ ...base }, 'feed-x', 0)).toBeNull();
        // An invalid declaration is rejected upstream by the CLI; the pure
        // function treats it as absent and skips.
        expect(normalizeRow({ ...base }, 'feed-x', 0, 'dollars')).toBeNull();
    });
});
