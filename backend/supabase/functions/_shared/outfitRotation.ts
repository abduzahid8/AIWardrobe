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

/** Lexicographic deck priority: fresh first, avoid previous next, oldest last-use next. */
export function rotationDeckPriority(items:any[],history:ReadonlySet<string>,previous:ReadonlySet<string>,decks:{productKeys:string[]}[],anchorId?:string):number {
    const products=items.filter(item=>String(item.id)!==anchorId);
    const fresh=freshProductCount(products,history);
    const previousCount=products.filter(item=>wasRecentlyShown(item,previous)).length;
    const age=products.reduce((sum,item)=>{
        let last=-1;decks.forEach((deck,index)=>{if(wasRecentlyShown(item,new Set(deck.productKeys)))last=index;});
        return sum+(last<0?0:decks.length-last);
    },0);
    return fresh*100000 + (products.length-previousCount)*1000 + age;
}

/** Cache history classification once per physical row during a bounded deck search. */
export function createRotationPriority(history:ReadonlySet<string>,previous:ReadonlySet<string>,decks:{productKeys:string[]}[],anchorId?:string) {
    const deckSets=decks.map(deck=>new Set(deck.productKeys));
    const cache=new Map<string,{keys:string[];score:number}>();
    return (items:any[]):number=>{
        let total=0;const used=new Set<string>();
        for(const item of items){
            const id=String(item.id);if(id===anchorId)continue;
            let entry=cache.get(id);
            if(!entry){
                const keys=productIdentityKeys(item);let last=-1;
                deckSets.forEach((set,index)=>{if(wasRecentlyShown(item,set))last=index;});
                entry={keys,score:(!wasRecentlyShown(item,history)?100000:0)+(!wasRecentlyShown(item,previous)?1000:0)+(last<0?0:decks.length-last)};
                cache.set(id,entry);
            }
            if(entry.keys.some(key=>used.has(key)))continue;
            entry.keys.forEach(key=>used.add(key));total+=entry.score;
        }
        return total;
    };
}
