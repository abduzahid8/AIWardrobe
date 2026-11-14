/**
 * Experimental Stylist 6: creative briefs -> real-SKU composition ->
 * hosted whole-look visual critique -> joint deck selection.
 *
 * This module is deliberately NOT an HTTP endpoint. Auth, subscription,
 * quota, identity aliases and FX source must be supplied by the hardened
 * backend adapter. Do not call it directly from mobile or replace production.
 */
import {createHash} from 'node:crypto';
import {canonicalProductKey} from '../../src/domain/outfitDiversity.ts';
import {baseProductId} from './catalog_adapter.mjs';
import {MAX_VISUAL_CANDIDATES,validVisualVerdict} from './visual_critic_hosted.mjs';
export const STYLIST6_VERSION='stylist6-visual-first-prototype-4';
// Treat a near-tie as insufficient evidence to displace an earlier feasible pick.
const VISUAL_NEAR_TIE=0.015;
const PALETTE_MATCH_WEIGHT=0.15;
const slots=['top','outerwear','bottom','shoes'];
const text=x=>String(x??'').trim().toLowerCase().replace(/[\s-]+/g,'_');
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const isHttps=x=>{try{const u=new URL(x);return u.protocol==='https:'&&!!u.hostname&&!/^(localhost|127\.|10\.|192\.168\.|169\.254\.|0\.)/i.test(u.hostname)}catch{return false}};
const assert=(yes,reason)=>{if(!yes)throw Error('stylist6_'+reason)};
function parseBriefs(raw,fashion){
  assert(Array.isArray(raw)&&raw.length>=(fashion?1:6)&&raw.length<=12,'creative_briefs_required');
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
    // Reference-first briefs carry a stable lookbook identity; no media URL
    // or unverified user-uploaded binary is sent to a model by this adapter.
    const referenceId=typeof x.referenceId==='string' && /^rt_\d{3}$/.test(x.referenceId)
      ? x.referenceId : undefined;
    return {id,name:x.name.slice(0,140),roles,layerPolicy:x.layerPolicy,palette,silhouette,referenceId};
  }).map((b,i,a)=>{if(!fashion&&i===a.length-1)assert(formats.size>=4,'creative_briefs_not_diverse');return b});
}
function safeCatalogItem(raw,mode){
  if(!raw||!raw.id||raw.is_active===false||raw.isActive===false)return null;
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
  +(brief.palette.includes(item.color)?PALETTE_MATCH_WEIGHT:0)
  +(item.visualEvidenceVerified===true?0.05:0)
);
function getRole(items,brief,slot,cap,rank,fashion,required){
  const slotItems=items.filter(i=>i.macroCategory===slot);
  const locked=fashion?slotItems.filter(i=>required.includes(i.id)):[];
  if(locked.length)return locked;
  const matched=slotItems.filter(i=>brief.roles[slot].includes(i.subtype));
  const ranked=(fashion&&matched.length<cap?slotItems:matched)
    .map(item=>({item,rank:rank(item,brief)}))
    .sort((a,b)=>b.rank-a.rank||a.item.id.localeCompare(b.item.id))
    .map(({item})=>item);
  const chosen=[],seen=new Set();
  for(const i of ranked){
    const kind=fashion?fashion.family(i):i.subtype+'|'+i.color+'|'+i.texture;
    if(seen.has(kind))continue;
    chosen.push(i);seen.add(kind);
    if(chosen.length===Math.min(cap,brief.roles[slot].length*2))break;
  }
  for(const i of ranked){if(chosen.length>=cap)break;if(!chosen.includes(i))chosen.push(i)}
  return chosen;
}
function hashLook(items){return createHash('sha256').update(items.map(i=>i.id).sort().join('|')).digest('hex').slice(0,16)}
function makeCandidateDeck(brief,pools,max,rank,admit){
  const pairs=brief.layerPolicy==='absent'?pools.top.map(t=>[t]):
    pools.top.flatMap(t=>pools.outerwear.map(l=>[t,l]));
  const bottoms=pools.bottom.flatMap(b=>pools.shoes.map(s=>[b,s]));
  const candidates=[];
  for(const upper of pairs)for(const lower of bottoms){
    const items=[...upper,...lower];
    if(new Set(items.map(x=>x.id)).size!==items.length||!admit(items))continue;
    // Mean item rank, not the sum: a three-piece look must compete on the
    // quality of its garments, not lose a full item-score to every layered
    // look before coherence is even measured.
    candidates.push({id:'s6_'+hashLook(items),briefId:brief.id,brief,items,
      seedScore:items.reduce((s,i)=>s+rank(i,brief),0)/items.length});
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
const photoUrl=i=>i.imageUrl||i.image_url;
const sameProduct=(a,b)=>baseProductId(a.id)===baseProductId(b.id)||
  (!!photoUrl(a)&&photoUrl(a)===photoUrl(b));
const sharedProducts=(a,b)=>a.items.filter(item=>b.items.some(other=>sameProduct(item,other))).length;
// Preserve the seed winner and use the remaining review slots for alternatives
// within one palette match of the strongest unreviewed seed.
function diversifyBriefHead(candidates){
  candidates.sort((a,b)=>b.seedScore-a.seedScore);
  if(candidates.length<3)return;
  const chosen=[candidates.shift()];
  while(candidates.length&&chosen.length<MAX_VISUAL_CANDIDATES){
    const floor=candidates[0].seedScore-PALETTE_MATCH_WEIGHT-1e-9;
    let best=0,leastOverlap=Infinity;
    for(const [index,candidate] of candidates.entries()){
      if(candidate.seedScore<floor)break;
      const overlap=Math.max(...chosen.map(look=>sharedProducts(candidate,look)));
      if(overlap<leastOverlap){
        best=index;
        leastOverlap=overlap;
      }
      if(leastOverlap===0)break;
    }
    chosen.push(candidates.splice(best,1)[0]);
  }
  candidates.unshift(...chosen);
}
function novelty(a,b){
  const shared=sharedProducts(a,b);
  if(Math.max(a.items.length,b.items.length)-shared<2)return 0;
  const overlap=shared/(a.items.length+b.items.length-shared);
  const roleMatch=slots.filter(slot=>{
    const x=a.items.find(i=>i.macroCategory===slot),y=b.items.find(i=>i.macroCategory===slot);
    return x?.subtype===y?.subtype&&['color','texture'].every(field=>!x?.[field]||!y?.[field]||x[field]===y[field]);
  }).length/slots.length;
  return 1-(0.65*roleMatch+0.35*overlap);
}
/**
 * Retrieval-only shortlist for blind human review. Runs every deterministic
 * stage up to (not including) the hosted visual critic and returns
 * seed-ranked candidates explicitly marked `visualVerdict:'NOT_RUN'`.
 * Human judges are the primary evaluators in the pilot; this output must
 * never be presented as visually approved.
 */
export async function shortlistRetrievalDeck({
 request,catalog,creativeDirector,hardGate,priceToBudgetCurrency,options={}
}){
  assert(request&&['catalog','wardrobe','mixed'].includes(request.sourceMode),'request');
  assert(Array.isArray(catalog)&&typeof creativeDirector==='function','input');
  assert(!request.budget||(finite(request.budget.amount)&&request.budget.amount>=0&&
    /^[A-Z]{3}$/.test(request.budget.currency)),'budget');
  assert(typeof hardGate==='function'&&typeof priceToBudgetCurrency==='function','trusted_gates_required');
  const briefLimit=Math.min(12,Math.max(6,options.briefLimit??8));
  const perRole=Math.min(24,Math.max(4,options.perRole??12));
  const maxVisual=Math.min(72,Math.max(12,options.maxVisual??48));
  const briefCandidates=Math.min(120,Math.max(10,options.perBrief??60));
  const fashion=options.fashion,required=request.requiredIds||[];
  const rank=fashion?fashion.rankItem:score;
  let items=catalog.map(i=>safeCatalogItem(i,request.sourceMode)).filter(Boolean);
  if(fashion)items=items.map(fashion.prepare).filter(Boolean);
  // Pre-ranking exclusion: previously displayed products leave the candidate
  // pool before roles are built, so a second tap cannot resurface the same
  // SKU when suitable alternatives exist. Anchored required ids are exempt.
  // When exclusion would empty a mandatory slot pool, that slot is relaxed
  // and reported instead of failing the whole deck.
  const requiredBases = new Set(required.map((id)=>baseProductId(String(id))));
  const excludedBases = new Set();
  for (const raw of (Array.isArray(request.excludedIds) ? request.excludedIds : [])) {
    const id = String(raw || '').trim();
    if (!id || required.includes(id) || requiredBases.has(baseProductId(id))) continue;
    excludedBases.add(baseProductId(id));
  }
  const relaxedSlots = [];
  if (excludedBases.size) {
    const kept = items.filter((i)=>!excludedBases.has(baseProductId(i.id)));
    for (const slot of ['top','bottom','shoes']) {
      if (!kept.some((i)=>i.macroCategory===slot)) relaxedSlots.push(slot);
    }
    items = relaxedSlots.length
      ? items.filter((i)=>relaxedSlots.includes(i.macroCategory) || !excludedBases.has(baseProductId(i.id)))
      : kept;
  }
  const briefs=parseBriefs(await creativeDirector(request,briefLimit),fashion);
  const all=new Map(),rejections={budget:0,hardGate:0,unpriced:0,...(fashion?{coherence:0}:{})};
  const costs=new Map();
  const budgetAdmits=(parts)=>{
    if(!request.budget)return true;
    const charges=parts.map(i=>{
      if(!costs.has(i))costs.set(i,i.isShopItem===true?priceToBudgetCurrency(i,request.budget.currency,request):0);
      return costs.get(i);
    });
    if(charges.some((v,i)=>!finite(v)||v<0||(parts[i].isShopItem===true&&v===0))){rejections.unpriced++;return false}
    if(charges.reduce((a,v)=>a+v,0)>request.budget.amount+0.000001){rejections.budget++;return false}
    return true;
  };
  const admit=fashion?parts=>required.every(id=>parts.some(i=>i.id===id))&&
    !parts.some((item,index)=>parts.slice(0,index).some(other=>sameProduct(item,other)))&&budgetAdmits(parts):()=>true;
  for(const b of briefs){
    const roles=Object.fromEntries(slots.map(slot=>[slot,getRole(items,b,slot,perRole,rank,fashion,required)]));
    if(!roles.top.length||!roles.bottom.length||!roles.shoes.length)continue;
    for(const policy of b.layerPolicy==='optional'?['required','absent']:[b.layerPolicy]){
      if(policy==='required'&&!roles.outerwear.length)continue;
      const candidates=makeCandidateDeck({...b,layerPolicy:policy},roles,briefCandidates,rank,admit);
      for(const c of candidates){
        if(required.some(id=>!c.items.some(i=>i.id===id)))continue;
        const gate=await hardGate(c.items.map(i=>i),request);
        if(!gate||gate.ok!==true){rejections.hardGate++;continue}
        if(!fashion&&!budgetAdmits(c.items))continue;
        if(fashion){
          const quality=fashion.rankLook(c.items,c.brief);
          if(!quality){rejections.coherence++;continue}
          c.fashion=quality;
          c.seedScore=quality.score;
        }
        const prior=all.get(c.id);
        if(!prior||c.seedScore>prior.seedScore)all.set(c.id,c);
      }
    }
  }
  const perBrief=new Map(briefs.map(b=>[b.id,[]]));
  for(const c of all.values())perBrief.get(c.briefId)?.push(c);
  for(const candidates of perBrief.values())diversifyBriefHead(candidates);
  const shortlist=[],added=new Set();
  for(let round=0;round<maxVisual;round++){
    for(const b of briefs){
      const c=perBrief.get(b.id)?.[round];
      if(c&&!added.has(c.id)){added.add(c.id);shortlist.push(c)}
      if(shortlist.length===maxVisual)break;
    }
    if(shortlist.length===maxVisual)break;
  }
  return {generationVersion:STYLIST6_VERSION,visualVerdict:'NOT_RUN',briefs,shortlist,added,
    diagnostics:{briefs:briefs.length,catalogInput:catalog.length,admittedCatalogItems:items.length,
      candidateCount:all.size,shortlisted:shortlist.length,rejections,
      exclusions:{requested:excludedBases.size,relaxedSlots}}};
}

export async function proposeVisualFirstDeck({
 request,catalog,creativeDirector,visualCritic,hardGate,priceToBudgetCurrency,options={}
}){
  const retrieval=await shortlistRetrievalDeck({
    request,catalog,creativeDirector,hardGate,priceToBudgetCurrency,options});
  const shortlist=request.referenceId
    ? retrieval.shortlist.filter(c=>c.brief.referenceId===request.referenceId)
    : retrieval.shortlist.filter(c=>c.brief.referenceId);
  return evaluateVisualShortlist({...retrieval,shortlist},request,visualCritic,options);
}

/** Judge a bounded slice of a trusted, already feasibility-gated shortlist. */
export async function evaluateVisualShortlist(retrieval,request,visualCritic,options={}){
  assert(typeof visualCritic==='function','visual_critic_required');
  const limit=options.maxHostedCandidates??MAX_VISUAL_CANDIDATES;
  assert(Number.isInteger(limit)&&limit>=1&&limit<=MAX_VISUAL_CANDIDATES,'invalid_visual_limit');
  const shortlist=retrieval.shortlist.slice(0,limit);
  const payload=shortlist.map(c=>({
    id:c.id,referenceId:request.referenceId||c.brief.referenceId,
    products:c.items.map(i=>({id:i.id,slot:i.macroCategory,imageUrl:i.imageUrl||i.image_url}))
  }));
  let critiques=[];
  let criticFailed=false;
  if(payload.length){
    try{
      const rows=await visualCritic(payload,{style:request.style,occasion:request.occasion,brief:request.brief});
      if(Array.isArray(rows)&&rows.length<=limit)critiques=rows;
      else criticFailed=true;
    }catch{criticFailed=true}
  }
  return selectVisualDeck({...retrieval,shortlist},critiques,options,criticFailed);
}

/**
 * A visually reviewed deck needs garment-architecture contrast, not merely
 * renamed SKUs. Explicit subtype, texture and known color differences are
 * catalog evidence; unknown attributes never fabricate a positive claim.
 */
export function independentOutfitDirection(a, b, requiredIds = []) {
  const roles = ['top', 'outerwear', 'bottom', 'shoes'];
  const anchored = new Set((requiredIds || []).map(String));
  const valid = value => typeof value === 'string' && value.trim() &&
    !['unknown','neutral','none','smooth','missing'].includes(value.trim().toLowerCase());
  const nameColor = item => {
    const name = String(item?.name || '').toLowerCase();
    const match = name.match(/\b(?:navy|white|cream|grey|gray|beige|brown|black|tan|burgundy|charcoal|olive|blue|red)\b/);
    return match?.[0] === 'gray' ? 'grey' : match?.[0];
  };
  const get = (look, slot) => (look?.items || []).find(x => x.macroCategory === slot);
  let structure = 0, texture = 0, palette = 0, comparable = 0;
  for (const role of roles) {
    const x = get(a, role), y = get(b, role);
    if (role === 'outerwear' && Boolean(x) !== Boolean(y)) { structure++; comparable++; continue; }
    if (!x || !y) continue;
    const subA = text(x.subtype), subB = text(y.subtype);
    if (role !== 'shoes' && valid(subA) && valid(subB)) {
      comparable++;
      if (subA !== subB) structure++;
    }
    const texA = text(x.texture), texB = text(y.texture);
    if (role !== 'shoes' && valid(texA) && valid(texB) && texA !== texB) texture++;
    if (anchored.has(String(x.id)) && anchored.has(String(y.id))) continue;
    const colA = text(valid(x.color) ? x.color : nameColor(x));
    const colB = text(valid(y.color) ? y.color : nameColor(y));
    if (valid(colA) && valid(colB) && colA !== colB) palette++;
  }
  if (!comparable) return true; // lack of metadata is not a veto
  return structure > 0 || texture >= 2 || palette >= (anchored.size ? 2 : 3);
}

/** Replay verdicts against the exact judged candidate set without provider I/O. */
export function selectVisualDeck({generationVersion,shortlist,diagnostics},critiques,options={},criticFailed=false){
  assert(shortlist.length<=MAX_VISUAL_CANDIDATES,'invalid_visual_limit');
  const target=Math.min(3,Math.max(1,options.target??3));
  const minVisualQuality=options.minVisualQuality??0.68;
  const allowed=new Set(shortlist.map(c=>c.id));
  const seen=new Set(),duplicates=new Set(),valid=new Map(),accepted=new Map();
  for(const row of critiques){
    if(!allowed.has(row?.candidateId))continue;
    if(seen.has(row.candidateId))duplicates.add(row.candidateId);
    seen.add(row.candidateId);
    if(validVisualVerdict(row,row.candidateId))valid.set(row.candidateId,row);
  }
  for(const id of duplicates)valid.delete(id);
  for(const [id,c] of valid){
    // Reference-less verdicts carry no referenceFidelity dimension: the
    // quality re-weights to cohesion-led with the same 0-1 scale, so a
    // missing reference can never read as a perfect or failed reference fit.
    const quality=c.referenceless===true
      ? 0.45*c.visualCohesion+0.20*c.styleFit+0.20*c.occasionFit+0.15*c.formalityFit
      : 0.30*c.visualCohesion+0.30*c.referenceFidelity+
        0.15*c.styleFit+0.15*c.occasionFit+0.10*c.formalityFit;
    if(c.confidence>=0.6&&quality>=minVisualQuality&&!c.reject)accepted.set(id,{...c,quality});
  }
  const eligible=shortlist.filter(c=>accepted.has(c.id));
  const strongest=Math.max(...eligible.map(c=>accepted.get(c.id).quality));
  const first=eligible.find(c=>accepted.get(c.id).quality>=strongest-VISUAL_NEAR_TIE);
  let selected=first?[first]:[],bestUtility=-Infinity;
  const anchorSet=new Set((options.requiredIds||[]).map(String));
  const nonAnchorSlotKeys=(look,slot)=>new Set(
    (look?.items||[])
      .filter(it=>it&&it.macroCategory===slot&&!anchorSet.has(String(it.id)))
      .map(it=>canonicalProductKey(it))
      .filter(Boolean));
  const diversityAdjustment=(look,deck)=>{
    let adj=0;
    for(const slot of ['top','bottom']){
      const cur=nonAnchorSlotKeys(look,slot);
      if(!cur.size)continue;
      const prev=new Set(deck.flatMap(d=>[...nonAnchorSlotKeys(d,slot)]));
      if([...cur].some(k=>prev.has(k)))adj-=0.04;
    }
    const shoeKeys=nonAnchorSlotKeys(look,'shoes');
    if(shoeKeys.size){
      const prevShoes=new Set(deck.flatMap(d=>[...nonAnchorSlotKeys(d,'shoes')]));
      adj+=[...shoeKeys].some(k=>prevShoes.has(k))?-0.02:0.01;
    }
    return adj;
  };
  // A greedy second pick can block the only viable third. The judged pool
  // is capped at four, so evaluate the small set of feasible continuations.
  function consider(deck,utility){
    if(deck.length>selected.length||(deck.length===selected.length&&utility>bestUtility)){
      selected=deck;
      bestUtility=utility;
    }
    if(deck.length===target)return;
    for(const c of eligible){
      if(deck.includes(c))continue;
      if(options.requireDistinctArchitectures===true && deck.length>=2 &&
          deck.some(other=>!independentOutfitDirection(c,other,options.requiredIds)))continue;
      const difference=Math.min(...deck.map(other=>novelty(c,other)));
      if(difference<(options.minNovelty??0.30)||difference===0)continue;
      const repeats=deck.some(other=>other.briefId===c.briefId);
      consider([...deck,c],utility+accepted.get(c.id).quality+0.18*difference-(repeats?0.10:0)+diversityAdjustment(c,deck));
    }
  }
  if(first)consider([first],0);
  return {generationVersion,
    status:selected.length===target?'complete':selected.length?'distinct_inventory_limited':'no_verified_looks',
    outfits:selected.map(c=>({id:c.id,concept:{id:c.brief.id,title:c.brief.name,referenceId:c.brief.referenceId||null},
      items:c.items,visualQuality:accepted.get(c.id).quality,
      referenceFidelity:accepted.get(c.id).referenceFidelity??null,
      referenceMode:accepted.get(c.id).referenceMode || 'unknown',
      evidence:accepted.get(c.id).evidence})),
    diagnostics:{...diagnostics,visualVerdict:valid.size?'JUDGED':shortlist.length?'ABSTAINED':'NOT_RUN',
      visualAttempted:shortlist.length,visualValid:valid.size,visualAbstained:shortlist.length-valid.size,
      visualRejected:[...valid.values()].filter(row=>row.reject).length,
      rejectedCandidateIds:[...valid.values()].filter(row=>row.reject&&row.confidence>=0.6).map(row=>row.candidateId),
      visualApproved:accepted.size,
      criticFailed,selected:selected.length}
  };
}
