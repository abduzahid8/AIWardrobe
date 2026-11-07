/** @jest-environment node */
import AsyncStorage from '@react-native-async-storage/async-storage';
import {assembleDiverseDeck, brandKey, createDeckVarietyScorer, deckNeedsVarietyReserve, garmentNearDuplicateKey, normalizeProductHistory, productIdentityKeys, spreadSuitableProducts} from '../../src/domain/outfitDiversity';
import {assignRotatingShoes, rotationPool, wasRecentlyShown} from '../../src/domain/outfitRotation';
import {getRecentShownIds, getRotationHistory, recordDisplayedDeck, recordShownOutfits, rotationScope} from '../../src/services/recentOutfitIds';
import {outfitVisualSignature, recentLookSimilarity} from '../../src/domain/outfitVisualSignature';

const slots=['top','outerwear','bottom','shoes'];
const formula=['White cotton shirt','Navy wool blazer','Grey tailored trousers','Brown leather loafers'];
const alternate=['Cream cotton polo','Beige cashmere cardigan','Olive cotton chinos','White leather sneakers'];
function look(id:string,brand:string,score=0.94,names=formula) {
    return {id,score,items:slots.map((macroCategory,i)=>({id:`${id}-${i}`,macroCategory,brand,color:names[i].split(' ')[0],name:`${names[i]} model ${id}`,imageUrl:`https://cdn.test/${id}-${i}.jpg`}))};
}
function select(candidates:ReturnType<typeof look>[]) {
    return assembleDiverseDeck({candidates,itemsOf:o=>o.items,qualityOf:o=>o.score,usedKeys:new Set<string>(),target:3,maxSearchSteps:512}).deck;
}

test('brand novelty never displaces stronger outfits; identity diversity survives shoe assignment',()=>{
    const mono=[look('a1','Brand A'),look('a2','Brand A'),look('a3','Brand A')];
    const candidates=[...mono,look('b','Brand B',0.93),look('c','Brand C',0.92),look('weak','Brand D',0.50)];
    expect(deckNeedsVarietyReserve(mono.map(o=>o.items))).toBe(true);
    const chosen=select(candidates);
    expect(chosen).toHaveLength(3);
    expect(chosen.map(o=>o.id).sort()).toEqual(['a1','a2','a3']);
    expect(chosen.some(o=>o.id==='weak')).toBe(false);
    expect(select(mono)).toHaveLength(3); // Scarcity never rejects suitable repeated brands.
    const ordered=spreadSuitableProducts(candidates.flatMap(o=>o.items.filter(i=>i.macroCategory==='top')),i=>i,()=>0.94);
    expect(new Set(ordered.slice(0,3).map(brandKey)).size).toBe(3);
    const scoreVariety=createDeckVarietyScorer();
    const assigned=assignRotatingShoes(mono,candidates.filter(o=>o.id!=='weak').flatMap(o=>o.items.filter(i=>i.macroCategory==='shoes')),()=>true,()=>[0],()=>0.94,undefined,deck=>-scoreVariety(deck.map(o=>o.items)));
    expect(new Set(assigned.deck.map(o=>o.items.find(i=>i.macroCategory==='shoes')!.id)).size).toBe(3);
});

test('display history reaches the next occasion, persists and excludes product variants, raw IDs and old image aliases',async()=>{
    const disk=new Map<string,string>();
    (AsyncStorage.getItem as jest.Mock).mockImplementation(async(key:string)=>disk.get(key)||null);
    (AsyncStorage.setItem as jest.Mock).mockImplementation(async(key:string,value:string)=>{disk.set(key,value);});
    const owner='repetition-regression-user',scope=rotationScope(owner,'shop','classic','Work');
    const shown=look('shown','Brand A');
    const original=shown.items[0] as any;
    original.shopUrl='https://brand.test/products/cotton-shirt?variant=white';
    original.imageUrl='https://db.test/storage/v1/render/image/public/catalog/shirt.jpg?width=400';
    recordShownOutfits([shown],owner);
    await recordDisplayedDeck([shown],scope,'request-one');
    const history=await getRotationHistory(rotationScope(owner,'shop','casual','Date'));
    expect(history.previousDeckProductKeys).toContain('id:shown-0');
    expect(history.recentLookSignatures).toContain(outfitVisualSignature(shown.items));
    expect([...disk.keys()]).toContain(`stylistRotationDecks::products:${owner}`);
    const recent=normalizeProductHistory([...history.previousDeckProductKeys,...history.olderRecentProductKeys]);
    const variant={...original,id:'variant',name:'Renamed listing',imageUrl:'https://other.test/new.jpg',shopUrl:'https://brand.test/products/cotton-shirt?variant=cream'};
    const alias={...original,id:'alias',name:'Another title',shopUrl:undefined,imageUrl:'https://db.test/storage/v1/object/public/catalog/shirt.jpg'};
    expect(wasRecentlyShown(variant,recent)).toBe(true);
    expect(wasRecentlyShown(alias,normalizeProductHistory(['i:db.test/storage/v1/render/image/public/catalog/shirt.jpg']))).toBe(true);
    expect(wasRecentlyShown({...original,id:'shop_raw-id'},normalizeProductHistory(['id:raw-id']))).toBe(true);
    const fresh=[look('fresh1','B'),look('fresh2','C'),look('fresh3','D')].flatMap(o=>o.items);
    const rotated=rotationPool([...shown.items,variant,alias,...fresh],recent,3);
    expect(rotated.recycledSlots).toEqual([]);
    expect(rotated.items.map(i=>i.id).sort()).toEqual(fresh.map(i=>i.id).sort());
    recordShownOutfits([look('next','B')],owner);
    recordShownOutfits([shown],owner);
    const ids=await getRecentShownIds(owner);
    expect(ids.at(-1)).toBe('shown-3'); // A repeat refreshes last-use instead of aging out.
});

test('unique SKU substitutions trigger reserve and lose to a different coherent formula within the quality margin',()=>{
    const mono=[look('near1','A',0.95),look('near2','B',0.94),look('near3','C',0.93)];
    const different=look('different','D',0.92,alternate),weak=look('weak-visual','E',0.50,alternate);
    expect(new Set(mono.flatMap(o=>o.items.map(i=>productIdentityKeys(i)[0]))).size).toBe(12);
    expect(garmentNearDuplicateKey(mono[0].items[0])).toBe(garmentNearDuplicateKey(mono[1].items[0]));
    expect(deckNeedsVarietyReserve(mono.map(o=>o.items))).toBe(true);
    const chosen=select([...mono,different,weak]);
    expect(chosen.map(o=>o.id)).toContain('different');
    expect(chosen.map(o=>o.id)).not.toContain('weak-visual');
    const recent=mono.map(o=>outfitVisualSignature(o.items));
    expect(recentLookSimilarity(different.items,recent)).toBeLessThan(recentLookSimilarity(mono[0].items,recent));
});
