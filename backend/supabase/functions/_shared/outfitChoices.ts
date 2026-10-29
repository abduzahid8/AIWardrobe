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
  let examined = 0;
  const gcd=(a:number,b:number):number=>b ? gcd(b,a%b) : a;
  const configurations=groups.filter(group=>group.every(pool=>pool.length)).map(group=>{
    const total=group.reduce((n,pool)=>n*pool.length,1);
    let stride=Math.max(1,Math.floor(total*0.618));
    while(gcd(stride,total)!==1)stride++;
    return {group,total,stride};
  });
  const largest=Math.max(0,...configurations.map(config=>config.total));
  for(let i=0;i<largest && examined<searchLimit && result.length<maximum;i++) {
    for(const {group,total,stride} of configurations) {
      if(i>=total || examined>=searchLimit || result.length>=maximum)continue;
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
  return result;
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
