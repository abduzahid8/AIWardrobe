import { outfitStylingDistance } from './outfitVisualSignature.ts';
import {productIdentityKeys, garmentNearDuplicateKey} from './outfitDiversity.ts';

/** Enumerate bounded, diverse proposals. The caller supplies the full validator. */
export function outfitChoices<T extends {id: string; macroCategory?: string; name?: string; type?: string}>(
  items: T[], layered: boolean, accepts: (items: T[]) => boolean, isDress: (item: T) => boolean,
  requiredIds: string[] = [], maximum = 24, searchLimit = 6000, minimumChangedItems = 2,
): {id: string; items: T[]}[] {
  const slot = (name: string) => items.filter(item => item.macroCategory === name);
  const tops = slot('top').filter(item => !isDress(item));
  const dresses = items.filter(item => isDress(item));
  const groups = [layered ? [tops,slot('outerwear'),slot('bottom'),slot('shoes')] : [tops,slot('bottom'),slot('shoes')],
    layered ? [dresses,slot('outerwear'),slot('shoes')] : [dresses,slot('shoes')]];
  const result: {id:string;items:T[]}[] = [];
  // An anchor removes one varying slot. Keep a bounded reserve before
  // trimming, so early valid combinations cannot hide the only third look.
  const reserve = requiredIds.length && maximum >= 3 ? Math.min(maximum * 12, 256) : maximum;
  let examined = 0;
  const gcd=(a:number,b:number):number=>b ? gcd(b,a%b) : a;
  // A locked garment fixes its slot before enumeration. Filtering after
  // sampling wastes the bounded search on outfits that cannot use the anchor.
  const configurations=groups.map(group=>group.map(pool=>{
    const locked=pool.filter(item=>requiredIds.includes(item.id));
    return locked.length?locked:pool;
  })).filter(group=>group.every(pool=>pool.length)).map(group=>{
    const total=group.reduce((n,pool)=>n*pool.length,1);
    let stride=Math.max(1,Math.floor(total*0.618));
    while(gcd(stride,total)!==1)stride++;
    return {group,total,stride};
  });
  const largest=Math.max(0,...configurations.map(config=>config.total));
  for(let i=0;i<largest && examined<searchLimit && result.length<reserve;i++) {
    for(const {group,total,stride} of configurations) {
      if(i>=total || examined>=searchLimit || result.length>=reserve)continue;
      examined++;
      let n=(i*stride)%total;
      const parts=group.map(pool=>{const item=pool[n%pool.length];n=Math.floor(n/pool.length);return item;});
      const ids=new Set(parts.map(item=>item.id));
      if(ids.size!==parts.length || requiredIds.some(id=>!ids.has(id)))continue;
      if(result.some(look=>look.items.filter(item=>ids.has(item.id)).length>Math.min(parts.length,look.items.length)-minimumChangedItems))continue;
      if(!accepts(parts))continue;
      result.push({id:`look${String(result.length+1).padStart(3,'0')}`,items:parts});
    }
  }
  if(reserve>maximum) {
    const keys=result.map(look=>new Set(look.items.filter(item=>!requiredIds.includes(item.id)).flatMap(productIdentityKeys)));
    const overlaps=(a:number,b:number)=>[...keys[a]].some(key=>keys[b].has(key));
    for(let a=0;a<result.length;a++)for(let b=a+1;b<result.length;b++) {
      if(overlaps(a,b))continue;
      for(let c=b+1;c<result.length;c++)if(!overlaps(a,c)&&!overlaps(b,c)) {
        const seed=new Set([a,b,c]);
        return [result[a],result[b],result[c],...result.filter((_,i)=>!seed.has(i))].slice(0,maximum);
      }
    }
  }
  return result.slice(0,maximum);
}

/** Only an AI-selected complete option may be expanded; unknown/partial IDs fail. */
export function resolveOutfitChoices(selections: any[] | null, choices: {id:string;items:any[]}[]): any[] | null {
  if(!selections?.length)return null;
  const byId=new Map(choices.map(choice=>[choice.id,choice]));
  const seen=new Set<string>(),result=[];
  for(const selection of selections){
    const id=selection.items?.length===1 ? selection.items[0]?.id : null;
    if(typeof id!=='string'||seen.has(id)||!byId.has(id))return null;
    seen.add(id);
    const {id: _modelId, ...metadata}=selection;
    result.push({...metadata,items:byId.get(id)!.items});
  }
  return result;
}

/** Compact ranked-choice protocol; only IDs are decoded, never product metadata. */
export function parseRankedOutfitChoices(raw: string): any[] | null {
  try {
    const parsed=JSON.parse(raw.trim().replace(/^```(?:json)?\s*/i,'').replace(/\s*```$/,''));
    const ids=parsed?.rankedLookIds;
    if(!Array.isArray(ids)||!ids.length||ids.length>100||ids.some(id=>typeof id!=='string'||!id.trim()))return null;
    return ids.map(id=>({items:[{id:id.trim()}]}));
  }catch{return null;}
}

/** Expand a feasible seed deck with diverse alternatives from the existing
 * validated reserve. No new enumeration or provider calls; at most 24 x 9
 * cached comparisons. Shared products are allowed BETWEEN shortlist options,
 * since the final deck selector enforces product uniqueness after ranking. */
export function curatedOutfitShortlist<T extends {id: string; items: any[]}>(
  candidates: T[], seed: T[], qualityOf: (look: T) => number,
  requiredIds: string[] = [], maximum = 9,
): T[] {
  const cap = Math.min(9, Math.max(1, maximum));
  const locked = new Set(requiredIds);
  const reserve = candidates.slice(0, 24).map((look, order) => ({look, order, quality: qualityOf(look),
    products: look.items.filter(item => !locked.has(String(item.id))).flatMap(productIdentityKeys),
    families: look.items.filter(item => !locked.has(String(item.id))).map(garmentNearDuplicateKey).filter(Boolean)}));
  const chosen = seed.filter((look, index) => candidates.includes(look) && seed.indexOf(look) === index).slice(0, cap);
  const products = new Set<string>(), families = new Set<string>();
  const remember = (look: T) => {
    const row = reserve.find(row => row.look === look);
    row?.products.forEach(key => products.add(key));
    row?.families.forEach(key => families.add(key));
  };
  chosen.forEach(remember);
  while (chosen.length < cap) {
    const available = reserve.filter(row => !chosen.includes(row.look));
    if (!available.length) break;
    const novelty = (row: typeof available[number]) =>
      (chosen.length ? Math.min(...chosen.map(look=>outfitStylingDistance(row.look.items,look.items,requiredIds[0]))) * 2 : 0)
      + row.products.filter(key => !products.has(key)).length / Math.max(1, row.products.length)
      + row.families.filter(key => !families.has(key)).length / Math.max(1, row.families.length);
    const bestQuality = Math.max(...available.map(row => row.quality));
    const comparable = available.filter(row => row.quality >= bestQuality - 0.08)
      .map(row => ({...row, novelty: novelty(row)}));
    comparable.sort((a, b) => b.novelty - a.novelty || b.quality - a.quality || a.order - b.order);
    chosen.push(comparable[0].look);
    remember(comparable[0].look);
  }
  return chosen;
}
