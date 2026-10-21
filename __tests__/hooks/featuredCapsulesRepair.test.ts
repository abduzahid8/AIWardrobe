import { repairDeletedCapsuleUrl } from '../../hooks/useFeaturedCapsules';

const SUPABASE = 'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/shop-catalog';

describe('repairDeletedCapsuleUrl (Oct 2026 deleted admin/ objects)', () => {
    it('remaps each known-deleted capsule object to its verified-good replacement', () => {
        expect(repairDeletedCapsuleUrl(`${SUPABASE}/admin/capsule-1781186443576.jpg`))
            .toBe(`${SUPABASE}/items/shop-inspo-1.png`);
        expect(repairDeletedCapsuleUrl(`${SUPABASE}/admin/capsule-1781186470011.jpg`))
            .toBe(`${SUPABASE}/items/shop-inspo-3.png`);
        expect(repairDeletedCapsuleUrl(`${SUPABASE}/admin/capsule-1781186613082.png`))
            .toBe(`${SUPABASE}/items/shop-inspo-5.png`);
    });

    it('leaves healthy URLs untouched (self-retiring once the DB is repaired)', () => {
        const healthy = `${SUPABASE}/items/shop-inspo-1.png`;
        expect(repairDeletedCapsuleUrl(healthy)).toBe(healthy);
        const unsplash = 'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=600&q=80';
        expect(repairDeletedCapsuleUrl(unsplash)).toBe(unsplash);
    });

    it('passes through empty / non-storage values safely', () => {
        expect(repairDeletedCapsuleUrl('')).toBe('');
        expect(repairDeletedCapsuleUrl(undefined as unknown as string)).toBe(undefined);
        expect(repairDeletedCapsuleUrl(null as unknown as string)).toBe(null);
    });
});
