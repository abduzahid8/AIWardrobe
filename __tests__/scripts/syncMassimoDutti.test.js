const {
    feedLocaleOf,
    mapItem,
} = require('../../scripts/sync-massimo-dutti.js');

const syncedAt = '2026-11-13T00:00:00.000Z';

const mdRaw = (over = {}) => ({
    name: 'Wool Blend Overcoat',
    productType: 'coats',
    price: 16900,
    image: 'https://cdn.massimodutti.com/photos/o1/2026/1/15/p/0/40/22/040220001_1_1_1.jpg',
    productPage: 'https://www.massimodutti.com/gb/men/wool-blend-overcoat-p04022001.html',
    id: '04022001',
    ...over,
});

describe('sync-massimo-dutti feed locale guard', () => {
    it('resolves the storefront locale from URLs', () => {
        expect(feedLocaleOf('https://www.massimodutti.com/gb/men/coats-n1447?celement=1')).toBe('gb');
        expect(feedLocaleOf('https://www.massimodutti.com/gb/men/overcoat-p04022001.html')).toBe('gb');
        expect(feedLocaleOf('https://www.massimodutti.com/de/men/coats-n1447')).toBe('de');
        expect(feedLocaleOf('https://www.massimodutti.com/en-gb/men/coats-n1447')).toBe('en-gb');
        expect(feedLocaleOf('https://www.massimodutti.com/men/coats-n1447')).toBeNull();
        expect(feedLocaleOf('not a url')).toBeNull();
        expect(feedLocaleOf(undefined)).toBeNull();
    });

    it('maps a /gb item to an honest GBP row', () => {
        const row = mapItem(mdRaw(), 0, syncedAt);
        expect(row).not.toBeNull();
        expect(row.currency).toBe('GBP');
        // Actor prices arrive in minor units (pence): 16900 → £169.00.
        expect(row.price).toBe(169);
        expect(row.last_seen_at).toBe(syncedAt);
    });

    it('skips items whose own product URL is another locale, never mislabels GBP', () => {
        expect(mapItem(mdRaw({
            productPage: 'https://www.massimodutti.com/de/men/wool-blend-overcoat-p04022001.html',
        }), 0, syncedAt)).toBeNull();
        expect(mapItem(mdRaw({
            productPage: 'https://www.massimodutti.com/en-gb/men/wool-blend-overcoat-p04022001.html',
        }), 0, syncedAt)).toBeNull();
        // An item without any URL still maps: the run-level guard has already
        // vetted the configured feed URLs it came from.
        expect(mapItem(mdRaw({ productPage: '' }), 0, syncedAt)).not.toBeNull();
    });
});
