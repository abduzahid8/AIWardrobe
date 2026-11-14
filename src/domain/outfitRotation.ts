import {countDistinctProducts, diversitySlotOf, productIdentityKeys, normalizeProductHistory, type DiversitySlot} from './outfitDiversity.ts';

export interface DisplayedRotationHistory {
    previousDeckProductKeys: string[];
    olderRecentProductKeys: string[];
    rotationDecks: {productKeys: string[]}[];
    previousShoeProductKeys?: string[];
    recentShoeProductKeys?: string[];
    shoeRotationDecks?: {productKeys: string[]}[];
}
export function displayedRotation(history?: DisplayedRotationHistory, anchorId?: string) {
    const recent = new Set([...(history?.previousDeckProductKeys || []), ...(history?.olderRecentProductKeys || []), ...(history?.recentShoeProductKeys || [])]);
    const previous = new Set(history?.previousDeckProductKeys || []);
    return {recent, priority: createShoeRotationPriority(recent, previous, history?.rotationDecks || [], anchorId,
        history?.shoeRotationDecks ? {previous: new Set(history.previousShoeProductKeys || []), decks: history.shoeRotationDecks} : undefined)};
}

/** A shared id, normalized name OR normalized photo is the same physical product. */
export function wasRecentlyShown(item: any, history: ReadonlySet<string>): boolean {
    const id = String(item?.id || '');
    return history.has(id) || history.has(id.replace(/^shop_/, '')) || history.has(`id:${id.replace(/^shop_/,'')}`) || history.has(`id:shop_${id.replace(/^shop_/,'')}`)
        || productIdentityKeys({...item, imageUrl: item?.imageUrl || item?.image_url || item?.image}).some(key => history.has(key));
}

/** Apply only to eligible products. Recycle a slot only after its fresh supply runs short.
 * `pruned` carries the recently-shown products dropped while fresh supply still met the
 * target — the caller needs them to prove feasibility when the fresh-only pool starves. */
export function rotationPool<T extends {id?: unknown}>(
    items: T[], history: ReadonlySet<string>, target: number, lockedIds: ReadonlySet<string> = new Set(),
): {items: T[]; recycledSlots: DiversitySlot[]; pruned: T[]} {
    const recycledSlots: DiversitySlot[] = [];
    const pruned: T[] = [];
    const kept: T[] = [];
    for (const slot of ['outerwear','top','bottom','shoes'] as DiversitySlot[]) {
        const pool = items.filter(item => diversitySlotOf(item) === slot);
        const fresh = pool.filter(item => !wasRecentlyShown(item, history) || lockedIds.has(String(item.id)));
        const seen = pool.filter(item => wasRecentlyShown(item, history) && !lockedIds.has(String(item.id)));
        kept.push(...fresh);
        if (countDistinctProducts(fresh) < target && seen.length) {
            recycledSlots.push(slot);
            kept.push(...seen);
        } else {
            pruned.push(...seen);
        }
    }
    return {items: kept, recycledSlots, pruned};
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
        let last=-1;decks.forEach((deck,index)=>{if(wasRecentlyShown(item,normalizeProductHistory(deck.productKeys)))last=index;});
        return sum+(last<0?0:decks.length-last);
    },0);
    return fresh*100000 + (products.length-previousCount)*1000 + age;
}

/** Cache history classification once per physical row during a bounded deck search. */
export function createRotationPriority(history:ReadonlySet<string>,previous:ReadonlySet<string>,decks:{productKeys:string[]}[],anchorId?:string) {
    const deckSets=decks.map(deck=>normalizeProductHistory(deck.productKeys));
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

export function compareRotationOrder(a: readonly number[], b: readonly number[]): number {
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
        const delta = (a[i] || 0) - (b[i] || 0);
        if (delta) return delta;
    }
    return 0;
}

/** Hard-quality-approved products only. Shoes cannot be pinned by a quality tie-break. */
export function createShoeRotationPriority(history: ReadonlySet<string>, previous: ReadonlySet<string>, decks: {productKeys: string[]}[], anchorId?: string,
    shoeHistory?: {previous:ReadonlySet<string>; decks:{productKeys:string[]}[]}) {
    const sets = decks.map(deck => normalizeProductHistory(deck.productKeys));
    const shoeSets = shoeHistory?.decks.map(deck=>normalizeProductHistory(deck.productKeys)) || sets;
    const legacy = createRotationPriority(history, previous, decks, anchorId);
    const cache = new Map<string, {keys: string[]; previous: boolean; last: number; count: number; fresh:boolean}>();
    return (items: any[]): number[] => {
        let shoeOverlap = 0, otherOverlap = 0, freshClothing=0;
        const ages = Array(shoeSets.length + 1).fill(0), counts = Array(shoeSets.length + 1).fill(0);
        const used = new Set<string>(), clothing: any[] = [];
        for (const item of items) {
            const id = String(item.id);
            if (id === anchorId) continue;
            const isShoe=diversitySlotOf(item)==='shoes';
            const cacheKey=`${id}|${isShoe?'shoe':'clothing'}`;
            let entry = cache.get(cacheKey);
            if (!entry) {
                let last = -1, count = 0;
                (isShoe ? shoeSets : sets).forEach((set, i) => { if (wasRecentlyShown(item, set)) {last = i; count++;} });
                entry = {keys: productIdentityKeys(item), previous: wasRecentlyShown(item, isShoe ? shoeHistory?.previous || previous : previous), last, count, fresh:!wasRecentlyShown(item,history)};
                cache.set(cacheKey, entry);
            }
            if (entry.keys.some(key => used.has(key))) continue;
            entry.keys.forEach(key => used.add(key));
            if (diversitySlotOf(item) === 'shoes') {
                shoeOverlap += Number(entry.previous);
                ages[entry.last + 1]++;
                counts[entry.count]++;
            } else {
                otherOverlap += Number(entry.previous);
                freshClothing+=Number(entry.fresh);
                clothing.push(item);
            }
        }
        // Prefer the oldest last-use buckets, then the fewest appearances.
        // Explicit vector comparison avoids weights that quality can outweigh.
        return [-shoeOverlap, freshClothing, -otherOverlap, ...ages, ...counts, legacy(clothing)];
    };
}

/** Exhaustively assign the small admitted shoe pool to already validated looks.
 * This closes the gap where a bounded complete-look shortlist omits a shoe.
 * Clothing is immutable; every substitution must pass the caller's full gate. */
export function assignRotatingShoes<O extends {items: any[]}>(
    looks: O[], shoes: any[], accepts: (items: any[]) => boolean,
    priority: (items: any[]) => readonly number[], quality: (items: any[]) => number,
    anchorId?: string, varietyOfDeck: (deck:O[])=>number = ()=>0,
): {deck: O[]; feasibleMinimumOverlap: number | null} {
    const anchor = shoes.find(shoe => String(shoe.id) === anchorId);
    const choices = looks.map(look => (anchor ? [anchor] : shoes).map(shoe => {
        const items = look.items.map(item => diversitySlotOf(item) === 'shoes' ? shoe : item);
        return accepts(items) ? {look: {...look, items}, keys: productIdentityKeys(shoe), score: quality(items)} : null;
    }).filter((choice): choice is NonNullable<typeof choice> => choice !== null));
    const requireUnique = !anchor && countDistinctProducts(shoes) >= looks.length;
    let best: O[] = [], bestOrder: readonly number[] = [], bestQuality = -Infinity, minimum: number | null = null;
    // CPU-safe bound for Edge execution: the exhaustive walk visits the full
    // product of per-look choice counts. Beyond this bound a deterministic
    // greedy pass (fewest choices first, best rotation order then quality,
    // index order as the final tie-break) assigns the same hard-constrained
    // shoes without the combinatorial blowup.
    const MAX_SHOE_ASSIGNMENT_COMBINATIONS = 20000;
    const totalCombinations = choices.reduce((product, list) => product * Math.max(1, list.length), 1);
    const greedyAssign = (): void => {
        const order = choices
            .map((list, index) => ({ list, index }))
            .sort((a, b) => a.list.length - b.list.length || a.index - b.index);
        const picked: (O | null)[] = new Array(choices.length).fill(null);
        const used = new Set<string>();
        let pickedScore = 0;
        for (const { list, index } of order) {
            let winner: (typeof list)[number] | null = null;
            for (const choice of list) {
                if (requireUnique && choice.keys.some(key => used.has(key))) continue;
                if (!winner) { winner = choice; continue; }
                const byOrder = compareRotationOrder(
                    priority([...picked.filter(Boolean).map(look => (look as O).items).flat(), ...choice.look.items]),
                    priority([...picked.filter(Boolean).map(look => (look as O).items).flat(), ...winner.look.items]),
                );
                const partial=picked.filter(Boolean) as O[];
                const gap=choice.score-winner.score;
                const betterQuality=Math.abs(gap)>0.08 ? gap>0 : comparableScore([...partial,choice.look as O],choice.score)>comparableScore([...partial,winner.look as O],winner.score);
                if (byOrder > 0 || (byOrder === 0 && betterQuality)) winner = choice;
            }
            if (!winner) { best = []; return; }
            picked[index] = winner.look as O;
            pickedScore += winner.score;
            winner.keys.forEach(key => used.add(key));
        }
        best = picked as O[];
        if (best.length) {
            const finalOrder = priority(best.flatMap(look => look.items));
            minimum = -finalOrder[0];
            bestOrder = finalOrder;
            bestQuality = pickedScore;
        }
    };
    const comparableScore=(deck:O[],score:number)=>score*2+varietyOfDeck(deck);
    const walk = (index: number, selected: O[], used: Set<string>, score: number) => {
        if (index === choices.length) {
            const order = priority(selected.flatMap(look => look.items));
            minimum = Math.min(minimum ?? Infinity, -order[0]);
            const comparison = best.length ? compareRotationOrder(order, bestOrder) : 1;
            const qualityGap=score-bestQuality;
            const betterQuality=Math.abs(qualityGap)>0.08*selected.length ? qualityGap>0
                : comparableScore(selected,score)>comparableScore(best,bestQuality);
            if (comparison > 0 || (comparison === 0 && betterQuality)) {best = [...selected]; bestOrder = order; bestQuality = score;}
            return;
        }
        for (const choice of choices[index]) {
            if (requireUnique && choice.keys.some(key => used.has(key))) continue;
            walk(index + 1, [...selected, choice.look as O], new Set([...used, ...choice.keys]), score + choice.score);
        }
    };
    if (looks.length) {
        if (totalCombinations > MAX_SHOE_ASSIGNMENT_COMBINATIONS) greedyAssign();
        else walk(0, [], new Set(), 0);
    }
    return {deck: best, feasibleMinimumOverlap: minimum};
}
