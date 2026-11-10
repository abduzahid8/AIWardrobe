/**
 * Experimental Stylist 6: creative briefs -> real-SKU composition ->
 * hosted whole-look visual critique -> joint deck selection.
 *
 * This module is deliberately NOT an HTTP endpoint. Auth, subscription,
 * quota, identity aliases and FX source must be supplied by the hardened
 * backend adapter. Do not call it directly from mobile or replace production.
 */
import {createHash} from 'node:crypto';
export const STYLIST6_VERSION='stylist6-visual-first-prototype-1';
const slots=['top','outerwear','bottom','shoes'];
const text=x=>String(x??'').trim().toLowerCase().replace(/[\s-]+/g,'_');
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const isHttps=x=>{try{const u=new URL(x);return u.protocol==='https:'&&!!u.hostname&&!/^(localhost|127\.|10\.|192\.168\.|169\.254\.|0\.)/i.test(u.hostname)}catch{return false}};
const assert=(yes,reason)=>{if(!yes)throw Error('stylist6_'+reason)};
function parseBriefs(raw){
  assert(Array.isArray(raw)&&raw.length>=6&&raw.length<=12,'creative_briefs_required');
  const seen=new Set(),formats=new Set();
  return raw.map(x=>{
    assert(x&&typeof x==='object'&&!Array.isArray(x)&&typeof x.id==='string'&&typeof x.name==='string','brief_schema');
    assert(!('products'in x)&&!('productIds'in x)&&!('items'in x),'brief_cannot_choose_skus');
    const id=text(x.id);assert(id.length>=3&&!seen.has(id),'brief_id');seen.add(id);
    assert(['required','optional','absent'].includes(x.layerPolicy),'layer_policy');
    const roles={};
    for(const slot of slots){
      const vals=x.roles?.[slot]??[];
      assert(Array.isArray(vals)&&vals.length<=10&&vals.every(y=>typeof y==='string'),'role_schema');
      roles[slot]=[...new Set(vals.map(text).filter(Boolean))];
    }
    assert(roles.top.length&&roles.bottom.length&&roles.shoes.length,'required_roles');
    if(x.layerPolicy==='required')assert(roles.outerwear.length,'required_layer');
    const palette=Array.isArray(x.palette)?x.palette.map(text).slice(0,8):[];
    const silhouette=text(x.silhouette);
    formats.add(JSON.stringify([roles,x.layerPolicy,palette,silhouette]));
    return {id,name:x.name.slice(0,140),roles,layerPolicy:x.layerPolicy,palette,silhouette};
  }).map((b,i,a)=>{if(i===a.length-1)assert(formats.size>=4,'creative_briefs_not_diverse');return b});
}
function safeCatalogItem(raw,mode){
  if(!raw||!raw.id)return null;
  const item={...raw};
  item.id=String(item.id);item.macroCategory=text(item.macroCategory);
  item.subtype=text(item.subtype||item.visual?.subtype);
  item.color=text(item.color||item.visual?.color);
  item.texture=text(item.texture||item.visual?.texture);
  if(!slots.includes(item.macroCategory)||!isHttps(item.imageUrl||item.image_url))return null;
  if(mode==='catalog'&&item.isShopItem!==true)return null;
  if(mode==='wardrobe'&&item.isShopItem===true)return null;
  if(item.isShopItem===true&&(!isHttps(item.shopUrl||item.sourceUrl)||!finite(item.price)||item.price<=0||!/^[A-Z]{3}$/.test(item.currency||'')))return null;
  return Object.freeze(item);
}
const score=(item,brief)=>(
  (brief.roles[item.macroCategory].includes(item.subtype)?1:0)
  +(brief.palette.includes(item.color)?0.15:0)
  +(item.visualEvidenceVerified===true?0.05:0)
);
function getRole(items,brief,slot,cap){
  const ranked=items.filter(i=>i.macroCategory===slot&&brief.roles[slot].includes(i.subtype))
    .sort((a,b)=>score(b,brief)-score(a,brief)||a.id.localeCompare(b.id));
  const chosen=[],seen=new Set();
  for(const i of ranked){
    const kind=i.subtype+'|'+i.color+'|'+i.texture;
    if(seen.has(kind))continue;
    chosen.push(i);seen.add(kind);
    if(chosen.length===Math.min(cap,brief.roles[slot].length*2))break;
  }
  for(const i of ranked){if(chosen.length>=cap)break;if(!chosen.includes(i))chosen.push(i)}
  return chosen;
}
function hashLook(items){return createHash('sha256').update(items.map(i=>i.id).sort().join('|')).digest('hex').slice(0,16)}
function makeCandidateDeck(brief,pools,max){
  const pairs=brief.layerPolicy==='absent'?pools.top.map(t=>[t]):
    pools.top.flatMap(t=>pools.outerwear.map(l=>[t,l]));
  const bottoms=pools.bottom.flatMap(b=>pools.shoes.map(s=>[b,s]));
  const candidates=[];
  for(const upper of pairs)for(const lower of bottoms){
    const items=[...upper,...lower];
    if(new Set(items.map(x=>x.id)).size!==items.length)continue;
    candidates.push({id:'s6_'+hashLook(items),briefId:brief.id,brief,items,
      seedScore:items.reduce((s,i)=>s+score(i,brief),0)});
  }
  candidates.sort((a,b)=>b.seedScore-a.seedScore||a.id.localeCompare(b.id));
  // Interleave item families, not just the first preferred item repeatedly.
  const selected=[],usage=new Map();
  for(const pass of [0,1,2,3]){
    for(const c of candidates){
      if(selected.length>=max)break;
      if(selected.some(v=>v.id===c.id))continue;
      if(c.items.some(i=>(usage.get(i.id)||0)>pass))continue;
      selected.push(c);for(const i of c.items)usage.set(i.id,(usage.get(i.id)||0)+1);
    }
  }
  return selected;
}
const similarity=(a,b)=>{
  const ai=new Set(a.items.map(i=>i.id)),bi=new Set(b.items.map(i=>i.id));
  const overlap=[...ai].filter(x=>bi.has(x)).length/new Set([...ai,...bi]).size;
  const roleMatch=slots.filter(slot=>a.items.find(i=>i.macroCategory===slot)?.subtype===b.items.find(i=>i.macroCategory===slot)?.subtype).length/4;
  return 0.65*roleMatch+0.35*overlap;
};
function validCritique(row,allowed){
  if(!row||!allowed.has(row.candidateId))return false;
  const keys=['visualCohesion','styleFit','occasionFit','formalityFit','confidence'];
  return keys.every(k=>finite(row[k])&&row[k]>=0&&row[k]<=1)&&
    Array.isArray(row.evidence)&&row.evidence.length>0&&row.evidence.every(v=>typeof v==='string');
}
export async function proposeVisualFirstDeck({
 request,catalog,creativeDirector,visualCritic,hardGate,priceToBudgetCurrency,options={}
}){
  assert(request&&['catalog','wardrobe','mixed'].includes(request.sourceMode),'request');
  assert(Array.isArray(catalog)&&typeof creativeDirector==='function','input');
  assert(typeof hardGate==='function'&&typeof priceToBudgetCurrency==='function','trusted_gates_required');
  assert(typeof visualCritic==='function','visual_critic_required');
  const target=Math.min(3,Math.max(1,options.target??3));
  const briefLimit=Math.min(12,Math.max(6,options.briefLimit??8));
  const perRole=Math.min(24,Math.max(4,options.perRole??12));
  const maxVisual=Math.min(72,Math.max(12,options.maxVisual??48));
  const minVisualQuality=options.minVisualQuality??0.68;
  const briefCandidates=Math.min(120,Math.max(10,options.perBrief??60));
  const items=catalog.map(i=>safeCatalogItem(i,request.sourceMode)).filter(Boolean);
  const briefs=parseBriefs(await creativeDirector(request,briefLimit));
  const all=new Map(),rejections={budget:0,hardGate:0,unpriced:0};
  for(const b of briefs){
    const roles=Object.fromEntries(slots.map(slot=>[slot,getRole(items,b,slot,perRole)]));
    if(!roles.top.length||!roles.bottom.length||!roles.shoes.length)continue;
    for(const policy of b.layerPolicy==='optional'?['required','absent']:[b.layerPolicy]){
      if(policy==='required'&&!roles.outerwear.length)continue;
      const candidates=makeCandidateDeck({...b,layerPolicy:policy},roles,briefCandidates);
      for(const c of candidates){
        if((request.requiredIds||[]).some(id=>!c.items.some(i=>i.id===id)))continue;
        const gate=await hardGate(c.items.map(i=>i),request);
        if(!gate||gate.ok!==true){rejections.hardGate++;continue}
        if(request.budget){
          const charges=c.items.map(i=>i.isShopItem===true?priceToBudgetCurrency(i,request.budget.currency,request):0);
          if(charges.some(v=>!finite(v)||v<0)){rejections.unpriced++;continue}
          if(charges.reduce((a,v)=>a+v,0)>request.budget.amount+0.000001){rejections.budget++;continue}
        }
        const prior=all.get(c.id);
        if(!prior||c.seedScore>prior.seedScore)all.set(c.id,c);
      }
    }
  }
  const perBrief=new Map(briefs.map(b=>[b.id,[]]));
  for(const c of all.values())perBrief.get(c.briefId)?.push(c);
  for(const v of perBrief.values())v.sort((a,b)=>b.seedScore-a.seedScore);
  const shortlist=[],added=new Set();
  for(let round=0;round<maxVisual;round++){
    for(const b of briefs){
      const c=perBrief.get(b.id)?.[round];
      if(c&&!added.has(c.id)){added.add(c.id);shortlist.push(c)}
      if(shortlist.length===maxVisual)break;
    }
    if(shortlist.length===maxVisual)break;
  }
  // Actual photos must be consumed by the remote critic adapter, not
  // replaced with image-derived adjectives and called 'visual analysis'.
  const critiques=await visualCritic(shortlist.map(c=>({
    id:c.id,briefId:c.briefId,concept:c.brief.name,
    products:c.items.map(i=>({id:i.id,imageUrl:i.imageUrl||i.image_url,name:i.name}))
  })),request);
  assert(Array.isArray(critiques),'critic_response');
  const accepted=new Map();
  for(const c of critiques){
    if(!validCritique(c,added)||accepted.has(c.candidateId))continue;
    const quality=0.30*c.visualCohesion+0.25*c.styleFit+0.25*c.occasionFit+0.20*c.formalityFit;
    if(c.confidence>=0.6&&quality>=minVisualQuality&&!c.reject)accepted.set(c.candidateId,{...c,quality});
  }
  const eligible=shortlist.filter(c=>accepted.has(c.id));
  const selected=[];
  while(selected.length<target){
    const remaining=eligible.filter(c=>!selected.some(s=>s.id===c.id)).map(c=>{
      const closest=selected.length?Math.max(...selected.map(s=>similarity(s,c))):0;
      const repeats=selected.some(s=>s.briefId===c.briefId);
      return {...c,utility:accepted.get(c.id).quality+0.18*(1-closest)-(repeats?0.10:0),
        novelty:1-closest};
    }).filter(c=>!selected.length||c.novelty>=(options.minNovelty??0.30))
      .sort((a,b)=>b.utility-a.utility||a.id.localeCompare(b.id));
    if(!remaining.length)break;
    selected.push(remaining[0]);
  }
  return {generationVersion:STYLIST6_VERSION,
    status:selected.length===target?'complete':selected.length?'distinct_inventory_limited':'no_verified_looks',
    outfits:selected.map(c=>({id:c.id,concept:{id:c.brief.id,title:c.brief.name},
      items:c.items,visualQuality:accepted.get(c.id).quality,evidence:accepted.get(c.id).evidence})),
    diagnostics:{briefs:briefs.length,catalogInput:catalog.length,admittedCatalogItems:items.length,
      candidateCount:all.size,visualReviewed:shortlist.length,visualApproved:accepted.size,
      selected:selected.length,rejections}
  };
}
