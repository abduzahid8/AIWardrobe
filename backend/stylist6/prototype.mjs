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
  // AI Creative Director's inventory-grounded roles are intentional. Do
  // not quietly replace them with unrelated garment subtypes just to fill
  // per-role retrieval capacity; fallback house briefs handle empty plans.
  const strictDirectedRole = /^ai_director_\d+$/.test(text(brief.id));
  const ranked=(fashion&&matched.length<cap&&!strictDirectedRole?slotItems:matched)
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
// Count only source-backed garment roles; distinct SKUs or shoes alone are
// not evidence that a different outfit construction reached the shortlist.
function architecturalOverlap(a,b){
  let comparable=0, same=0;
  for(const slot of ['top','outerwear','bottom']){
    const x=a.items.find(item=>item.macroCategory===slot);
    const y=b.items.find(item=>item.macroCategory===slot);
    if(!x && !y) continue;
    if(!x || !y){comparable++;continue;}
    if(!x.subtype || !y.subtype || x.subtype==='unknown' || y.subtype==='unknown') continue;
    comparable++;
    if(x.subtype===y.subtype) same++;
  }
  return comparable ? same/comparable : null;
}

// Preserve the best seed. Within an existing tight seed-quality window, show
// genuine garment architectures before merely varying SKU, brand or shoes.
// Missing subtype evidence falls back to the previous product-overlap policy.
export function diversifyBriefHead(candidates){
  candidates.sort((a,b)=>b.seedScore-a.seedScore);
  if(candidates.length<3)return;
  const chosen=[candidates.shift()];
  while(candidates.length&&chosen.length<MAX_VISUAL_CANDIDATES){
    const floor=candidates[0].seedScore-PALETTE_MATCH_WEIGHT-1e-9;
    let best=0,leastArchitecture=Infinity,leastProductOverlap=Infinity;
    for(const [index,candidate] of candidates.entries()){
      if(candidate.seedScore<floor)break;
      const measured=chosen.map(look=>architecturalOverlap(candidate,look)).filter(Number.isFinite);
      const architecturalSimilarity=measured.length?Math.max(...measured):1;
      const productOverlap=Math.max(...chosen.map(look=>sharedProducts(candidate,look)));
      if(architecturalSimilarity<leastArchitecture ||
        (architecturalSimilarity===leastArchitecture && productOverlap<leastProductOverlap)){
        best=index;
        leastArchitecture=architecturalSimilarity;
        leastProductOverlap=productOverlap;
      }
      if(leastArchitecture===0 && leastProductOverlap===0)break;
    }
    chosen.push(candidates.splice(best,1)[0]);
  }
  candidates.unshift(...chosen);
}
/** Evidence-backed visual architecture, not a change of brand or SKU.
 * An unknown "jacket" is intentionally not guessed to be a blazer or coat.
 * Pinned owned hero garments are excluded from the novelty requirement.
 */
export function outerwearArchetype(look, requiredIds = []) {
  const layer=(look?.items||[]).find(it=>it?.macroCategory==='outerwear');
  if (!layer) return 'unlayered';
  if ((requiredIds||[]).some(id=>String(id)===String(layer.id))) return null;
  const kind=text(layer.subtype);
  if (['blazer','suit_jacket','tailored_jacket'].includes(kind)) return 'tailored_jacket';
  if (['coat','overcoat','topcoat','peacoat','trench','trench_coat'].includes(kind)) return 'full_coat';
  if (['cardigan','knit_cardigan','sweater','fine_knit'].includes(kind)) return 'knit_layer';
  if (kind==='overshirt') return 'overshirt';
  if (['bomber','harrington','windbreaker'].includes(kind)) return 'casual_jacket';
  return null;
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
    products:c.items.map(i=>({id:i.id,slot:i.macroCategory,imageUrl:i.imageUrl||i.image_url,
      // Source-backed garment labels add explicit context the photo alone
      // cannot reveal (e.g. a terry resort shirt in office tailoring).
      name:String(i.name||'').slice(0,160),subtype:String(i.subtype||'').slice(0,48)}))
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
 * A visually reviewed *deck* needs genuine garment-architecture contrast.
 * Different SKUs, colored shoes or a renamed concept alone cannot turn the
 * same shirt + blazer + trousers into three independent styling directions.
 * Only explicit catalog subtype/texture evidence can certify a material
 * change; broad palette contrast requires three known slot-color changes.
 */
export function independentOutfitDirection(a, b, requiredIds = []) {
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const anchored = new Set((requiredIds || []).map(String));
  const nameColor = (item) => {
    const name = String(item?.name || '').toLowerCase();
    const match = name.match(/\b(?:navy|white|cream|grey|gray|beige|brown|black|tan|burgundy|charcoal|olive|blue|red)\b/);
    return match?.[0] === 'gray' ? 'grey' : match?.[0];
  };
  const get = (look, slot) => (look?.items || []).find(x => x.macroCategory === slot);
  const valid = value => typeof value === 'string' && value.trim() &&
    !['unknown','neutral','none','smooth','missing'].includes(value.trim().toLowerCase());
  let structure = 0, texture = 0, palette = 0, comparable = 0;
  for (const slot of slots) {
    const x = get(a, slot), y = get(b, slot);
    if (slot === 'outerwear' && Boolean(x) !== Boolean(y)) { structure++; comparable++; continue; }
    if (!x || !y) continue;
    const subA = text(x.subtype), subB = text(y.subtype);
    if (slot !== 'shoes' && valid(subA) && valid(subB)) {
      comparable++;
      if (subA !== subB) structure++;
    }
    const texA = text(x.texture), texB = text(y.texture);
    if (slot !== 'shoes' && valid(texA) && valid(texB) && texA !== texB) texture++;
    // User-fixed anchors may repeat by design. They are neither a novelty
    // gain nor evidence that two otherwise different looks are identical.
    if (anchored.has(String(x.id)) && anchored.has(String(y.id))) continue;
    const colA = text(valid(x.color) ? x.color : nameColor(x));
    const colB = text(valid(y.color) ? y.color : nameColor(y));
    if (valid(colA) && valid(colB) && colA !== colB) palette++;
  }
  // Sparse catalog metadata cannot prove *or* disprove an architecture.
  if (!comparable) return true;
  return structure > 0 || texture >= 2 || palette >= (anchored.size ? 2 : 3);
}

/** Replay verdicts against the exact judged candidate set without provider I/O. */
export function selectVisualDeck({generationVersion,shortlist,diagnostics},critiques,options={},criticFailed=false){
  assert(shortlist.length<=MAX_VISUAL_CANDIDATES,'invalid_visual_limit');
  const target=Math.min(3,Math.max(1,options.target??3));
  const minVisualQuality=options.minVisualQuality??0.68;
  const allowed=new Set(shortlist.map(c=>c.id));
  const seen=new Set(),duplicates=new Set(),valid=new Map(),accepted=new Map();
  // An explainable, bounded receipt from the exact candidates actually
  // examined by NVIDIA. Do not change the established quality threshold to
  // turn rejected outfits into synthetic successes. No images or secrets.
  const qualityDecisions=[];
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
    const decision=c.reject?'model_veto':c.confidence<0.6?'low_confidence'
      :quality<minVisualQuality?'below_quality_floor':'approved';
    qualityDecisions.push({
      candidateId:id,
      quality:Number(quality.toFixed(4)),
      qualityFloor:minVisualQuality,
      confidence:c.confidence,
      reject:c.reject,
      reason:decision,
      scores:{cohesion:c.visualCohesion,styleFit:c.styleFit,
        occasionFit:c.occasionFit,formalityFit:c.formalityFit,
        referenceFidelity:c.referenceless===true?null:c.referenceFidelity},
      // The model's short slot-pair observations are essential for
      // investigating real fashion failures, but never include image bytes.
      evidence:(Array.isArray(c.evidence)?c.evidence:[])
        .slice(0,3).map(line=>String(line).slice(0,160)),
    });
    if(decision==='approved')accepted.set(id,{...c,quality});
  }
  const eligible=shortlist.filter(c=>accepted.has(c.id));
  const strongest=Math.max(...eligible.map(c=>accepted.get(c.id).quality));
  const first=eligible.find(c=>accepted.get(c.id).quality>=strongest-VISUAL_NEAR_TIE);
  // The strongest visual card must not become a mandatory anchor: it can
  // prevent three genuinely distinct outfits even when three OTHER approved
  // looks can form a full deck. Enumerate all seeds (at most four verified
  // candidates), maximize the number of independent looks first, then sum
  // visual quality with bounded diversity. Preserve the earlier candidate
  // on a true <=0.015 single-look near tie.
  let selected=[],bestUtility=-Infinity;
  // NVIDIA certifies style, not financial suitability. Once every look
  // cleared the model quality floor, use the server's source-price evidence
  // as a bounded tie-break. Never impose a hidden user budget or alter a
  // deliberately requested luxury/anchored outfit.
  const valuePenalty = look => options.preferValue===true
    ? 0.72 * Math.max(0, Math.min(0.18,
      Number.isFinite(look?.fashion?.unbudgetedPricePenalty)
        ? look.fashion.unbudgetedPricePenalty : 0)) : 0;
  const visualUtility = look => accepted.get(look.id).quality - valuePenalty(look);
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
      // Repeating the hero shirt across separate looks is visually more
      // noticeable than recycling trousers. Prefer different verified tops
      // when they clear the same independent-look and model-quality gates.
      if([...cur].some(k=>prev.has(k)))adj-=slot==='top'?0.14:0.04;
    }
    // Owner visual QA: 3 approved Classic / Work cards recycled the exact
    // New York wool overcoat twice, and a Bennett Chelsea boot twice. All
    // cards were technically distinct, yet read like one navy suit repeated.
    // A source-confirmed repeat of an unpinned hero layer/footwear SKU is a
    // near-quality tie-break, not a hard rejection: sparse wardrobe decks
    // remain valid and all choices still require NVIDIA approval.
    const coatKeys=nonAnchorSlotKeys(look,'outerwear');
    if(coatKeys.size){
      const prevLayers=new Set(deck.flatMap(d=>[...nonAnchorSlotKeys(d,'outerwear')]));
      if([...coatKeys].some(k=>prevLayers.has(k)))adj-=0.085;
    }
    // Three different blazers are still one garment architecture. Prefer
    // a genuinely different layer shape (coat, knit layer, overshirt) among
    // independently NVIDIA-approved near-quality candidates. Do not force a
    // new architecture if the catalog/owned wardrobe cannot support one.
    const layerType=outerwearArchetype(look,options.requiredIds);
    const previousTypes=new Set(deck.map(d=>outerwearArchetype(d,options.requiredIds)).filter(Boolean));
    if(layerType && previousTypes.has(layerType)) adj-=0.12;
    else if(layerType && previousTypes.size) adj+=0.035;
    const shoeKeys=nonAnchorSlotKeys(look,'shoes');
    if(shoeKeys.size){
      const prevShoes=new Set(deck.flatMap(d=>[...nonAnchorSlotKeys(d,'shoes')]));
      adj+=[...shoeKeys].some(k=>prevShoes.has(k))?-0.065:0.012;
    }
    return adj;
  };
  // A greedy second pick can block the only viable third. The judged pool
  // is capped at four, so evaluate the small set of feasible continuations.
  function consider(deck,utility){
    // All entries in this enumeration are individually NVIDIA-approved and
    // already passed the unchanged model quality and physical-SKU gates.
    // A small, capped joint-direction reward can prefer a genuinely distinct
    // third outfit over a slightly higher-scored but repetitive alternative.
    // The callback is trusted in-process ranking evidence, never a client
    // parameter or a substitute for a real provider verdict.
    let directionBonus=0;
    if(deck.length===3 && typeof options.directionDeckBonus==='function'){
      try{const value=options.directionDeckBonus(deck);
        if(Number.isFinite(value))directionBonus=Math.max(0,Math.min(0.12,value));
      }catch{directionBonus=0}
    }
    const deckUtility=utility+directionBonus;
    if(deck.length>selected.length||(deck.length===selected.length&&deckUtility>bestUtility)){
      selected=deck;
      bestUtility=deckUtility;
    }
    if(deck.length===target)return;
    for(const c of eligible){
      if(deck.includes(c))continue;
      // Enforce independently verified architecture for EVERY pair, even
      // when choosing the second look. An earlier >=2 guard let a pair of
      // near-clones through and falsely called three looks "distinct".
      if(options.requireDistinctArchitectures===true &&
        deck.some(other=>!independentOutfitDirection(c,other,options.requiredIds)))continue;
      const difference=Math.min(...deck.map(other=>novelty(c,other)));
      if(difference<(options.minNovelty??0.30)||difference===0)continue;
      const repeats=deck.some(other=>other.briefId===c.briefId);
      consider([...deck,c],utility+visualUtility(c)+0.18*difference-(repeats?0.10:0)+diversityAdjustment(c,deck));
    }
  }
  for(const seed of eligible){
    const preferredNearTie=seed===first?VISUAL_NEAR_TIE:0;
    consider([seed],visualUtility(seed)+preferredNearTie);
  }
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
      visualQualityDecisions:qualityDecisions,
      criticFailed,selected:selected.length}
  };
}
