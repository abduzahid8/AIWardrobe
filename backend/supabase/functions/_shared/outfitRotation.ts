import {countDistinctProducts, diversitySlotOf, productIdentityKeys, type DiversitySlot} from './outfitDiversity.ts';

/** A shared id, normalized name OR normalized photo is the same physical product. */
export function wasRecentlyShown(item: any, history: ReadonlySet<string>): boolean {
    const id = String(item?.id || '');
    return history.has(id) || history.has(id.replace(/^shop_/, ''))
        || productIdentityKeys({...item, imageUrl: item?.imageUrl || item?.image_url || item?.image}).some(key => history.has(key));
}

/** Apply only to eligible products. Recycle a slot only after its fresh supply runs short. */
export function rotationPool<T extends {id?: unknown}>(
    items: T[], history: ReadonlySet<string>, target: number, lockedIds: ReadonlySet<string> = new Set(),
): {items: T[]; recycledSlots: DiversitySlot[]} {
    const recycledSlots: DiversitySlot[] = [];
    const kept: T[] = [];
    for (const slot of ['outerwear','top','bottom','shoes'] as DiversitySlot[]) {
        const pool = items.filter(item => diversitySlotOf(item) === slot);
        const fresh = pool.filter(item => !wasRecentlyShown(item, history) || lockedIds.has(String(item.id)));
        const seen = pool.filter(item => wasRecentlyShown(item, history) && !lockedIds.has(String(item.id)));
        kept.push(...fresh);
        if (countDistinctProducts(fresh) < target && seen.length) {
            recycledSlots.push(slot);
            kept.push(...seen);
        }
    }
    return {items: kept, recycledSlots};
}

/** Among already valid decks, consume unused physical products before recycling. */
export function freshProductCount(items: any[], history: ReadonlySet<string>, anchorId?: string): number {
    return countDistinctProducts(items.filter(item => String(item.id) !== anchorId && !wasRecentlyShown(item, history)));
}
