/**
 * Isolated Stylist 6 cloud staging handler (Node/Vercel).
 *
 * Runs the VERIFIED local Stylist 6 fashion engine server-side:
 * server-fetched catalog inventory, cached visual-profile hydration,
 * recommendFashionDeck ranking, and the hosted NVIDIA visual critic with
 * the same admission rules as the local dev bridge.
 *
 * The Stylist 5 compute route is untouched; this handler answers only the
 * new isolated `stylist6-generate-outfits` route and reports a truthful
 * `stylist-6-*` generation version. It never runs as Stylist 5 and never
 * relabels deterministic ranking as provider judgment.
 *
 * Admission order (fail-closed, at-most-once admission):
 *   transport size/HMAC -> user JWT -> rate bucket -> server inventory ->
 *   anchor validation -> durable claim -> quota debit -> ranking -> response.
 * Claims and debits are separate transactions; planning failures keep debit.
 *
 * Client-supplied inventory is NEVER authoritative: any `catalogRows` in
 * the request body are ignored and reported as ignored. Wardrobe rows
 * always come from `clothing_items` filtered by the verified JWT user;
 * shop rows always come from the server-side discovery RPC.
 */
import { createHash } from 'node:crypto';
import { adaptCatalog, devLookFeasibility, sourceHash } from './catalog_adapter.mjs';
import { hydrateCatalogObservations } from './catalog_observation.mjs';
import { recommendFashionDeck } from './fashion_deck.mjs';
import { designCreativeBriefsWithFallback, CREATIVE_DIRECTOR_VERSION } from './creative_director_ai.mjs';
import { selectFashionSlate } from './fashion_director_slate.mjs';
import { selectPhotoEditorialDeck } from './photo_editor_final.mjs';
import { composeCatalogSkuLooks } from './sku_composer_ai.mjs';
import {verifiedCatalogPhotoCache,resolveCatalogCachedPhoto} from './catalog_photo_cache.mjs';
import { feedbackTasteModel, FEEDBACK_TASTE_VERSION, FEEDBACK_TASTE_MAX_ROWS } from './feedback_taste.mjs';
import { referenceFirstBriefs, explicitLookbookPhotoRequested } from './lookbook_briefs.mjs';
import { buildReferenceBlueprints } from './reference_blueprint.mjs';
import { currentVisual } from '../../src/domain/visualFashion.ts';
import { resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { isWorkoutOccasion } from '../../src/domain/occasionRules.ts';
import { normalizeStylistLanguage } from '../../src/domain/stylingContext.ts';
import {
  createHostedVisualCritic,
  NVIDIA_VISION_MODEL,
  MAX_VISUAL_CANDIDATES,
  VISUAL_PROVIDER_TIMEOUT_MS,
  VISUAL_TIMEOUT_MS,
} from './visual_critic_hosted.mjs';
import { HOST_ALLOWLIST, isInferenceAuthorized } from '../../scripts/stylist6/image_access.mjs';
import { prepareCloudVisualMedia } from './cloud_visual_media.mjs';
import { verifyStylistTransport, STYLIST_TRANSPORT_MAX_BODY_BYTES } from '../stylist-node/transport/signature.mjs';
import { verifyMerchantOffer, summarizeDeckOffers } from './merchant_offers.mjs';
import { deliberatelyRelaxedIntent } from './editorial_intent.mjs';
import { BUDGET_CURRENCIES, parseCbuBudgetRates, validateBudgetFxSnapshot, referenceAmount } from '../../src/domain/budgetFx.ts';
import referenceIndex from '../../benchmarks/stylist6/references/image_teaste_46.json' with { type: 'json' };

/** Leave the actual photographic judge enough time while admitting the
 * smaller model-led fashion slate even on real cloud catalog retrievals.
 * A hard 22.5s reservation previously made the Director NOT_RUN on live iOS
 * despite 28 perfectly valid real outfits. No server deadline is extended.
 */
/** The quick whole-look fashion selector is permitted for workouts too.
 * It selects EXISTING admitted gear; unlike the AI architecture generator,
 * it cannot invent a training plan. The workout item-name and physical hard
 * gates remain in the retrieval scorer. Both switches are server-owned and
 * a genuine photo critic must be enabled before any selection occurs.
 */
export function serverFashionSlateEnabled(env, {visualAsked, visualCriticAvailable} = {}) {
  return /^(1|true)$/i.test(String(env?.STYLIST6_CREATIVE_DIRECTOR_ENABLED||'').trim())
    && /^(1|true)$/i.test(String(env?.STYLIST6_AI_SLATE_ENABLED||'').trim())
    && visualAsked===true && visualCriticAvailable===true;
}

/** An immutable source *sampling seed* for honest same-SKU visual QA.
 * The default remains the existing new-request discovery contract. A server-
 * owned >=72s isolated staging switch removes ephemeral request IDs from the
 * sampling seed, so a changed Creative Director can be judged against the
 * SAME real merchant products. This is NOT a client flag, a response cache,
 * a quota bypass or a guarantee merchant inventory never updates externally.
 */
export function shouldFreezeStylist6CatalogCohort(env,{budgetMs,sourceMode,hasAnchors}={}){
  return /^(1|true)$/i.test(String(env?.STYLIST6_STABLE_COHORT_ENABLED||'').trim())&&
    Number.isFinite(budgetMs)&&budgetMs>=70000&&
    sourceMode==='catalog'&&hasAnchors===false;
}
export function stylist6CatalogDiscoverySeed(userId,body,{
  stableCohort=false,style='',occasion='',sourceMode='catalog',
  weather=null,budget=null,
}={}){
  // The wearer remains a component of the sample key. Never permit a
  // pooled fixed cohort to mix one account's catalog eligibility with another.
  const hash=createHash('sha256').update(String(userId)).update('|');
  if(stableCohort){
    hash.update('stylist6_stable_source_cohort_v1|').update(JSON.stringify({
      style:String(style||''),occasion:String(occasion||''),
      sourceMode:String(sourceMode||''),
      weatherCelsius:Number.isFinite(Number(weather?.temp))
        ?Number(weather.temp):null,
      budget:budget&&Number.isFinite(Number(budget.amount))
        ?{amount:Number(budget.amount),currency:String(budget.currency||'')}:null,
    }));
  }else hash.update(String(body||''));
  return hash.digest('hex').slice(0,16);
}

export function fashionSlateDecisionBudgetMs(remainingMs) {
  if (!Number.isFinite(remainingMs) || remainingMs < 11500) return 0;
  return Math.min(2600, Math.max(600, Math.floor(remainingMs - 9000)));
}

export const STYLIST6_CLOUD_VERSION = 'stylist-6-cloud-staging-1';
const QUOTA_FEATURE = 'aiw-stylist6-staging';
const RATE_LIMIT_PER_MINUTE = 10;
const CATALOG_SLOTS = ['top', 'bottom', 'shoes', 'outerwear'];
const CATALOG_PER_SLOT = 100;
// Shop discovery returns 100 per RPC. Only the shoe lane takes two extra
// pages; hydration and inference remain capped at 100 verified shoes.
const SHOE_DISCOVERY_PAGES = 3;
// Discover more REAL garments than the 100/slot visual/price hydration limit,
// then choose a style-rich source window. One RPC page per non-shoe lane
// previously made the first 100 shirts and pullovers crowd coats, overshirts
// and different trouser constructions out of the AI's 12-token per-role menu.
// All extra RPCs are read-only and run concurrently; ONLY the same 400
// trusted source rows are hydrated or sent to the stylist.
const GARMENT_DISCOVERY_PAGES = 2;
const WARDROBE_PAGE = 500;
const VISUAL_PROFILE_CHUNK = 100;
const PRODUCT_FETCH_TIMEOUT_MS = 8000;
const MAX_PRODUCT_IMAGE_BYTES = 4 * 1024 * 1024;
const MAX_VISUAL_IMAGE_BYTES = 512 * 1024;
/** Safe staging-only quality trace: never log private user data, SKU IDs,
 * prompts, merchant links, reference images or auth material. Exact review
 * reasons show whether a 2-look deck failed from image quality or from the
 * portfolio's inability to combine otherwise approved outfits.
 */
export function stylist6SafeQualityTrace(diagnostics, displayed, elapsedMs) {
  const d=diagnostics&&typeof diagnostics==='object'?diagnostics:{};
  const r=d.visualReviewCoverage&&typeof d.visualReviewCoverage==='object'
    ?d.visualReviewCoverage:{};
  const f=d.modelFashionSelection&&typeof d.modelFashionSelection==='object'
    ?d.modelFashionSelection:{};
  const allow=new Set([
    'approved','below_quality_floor','style_mismatch','occasion_mismatch',
    'formality_mismatch','model_veto','low_confidence','excluded_visual_color',
    'visual_color_unverified',
  ]);
  const reasons=Array.isArray(d.visualQualityDecisions)
    ?d.visualQualityDecisions.slice(0,8).map(x=>
      allow.has(x?.reason)?x.reason:'other')
    :[];
  const safe=(n,max=100000)=>Number.isSafeInteger(n)&&n>=0&&n<=max?n:null;
  return {
    elapsedMs:safe(elapsedMs,180000),
    displayed:safe(displayed,3),
    requestedPhoto:r.mode==='displayed_deck',
    attempted:safe(r.attempted,12),
    approved:safe(r.approved,12),
    displayedEvidence:safe(r.displayedEvidence,3),
    directorSelected:f.status==='selected'||f.status==='composed',
    directSkuComposition:f.status==='composed',
    skuCompositionStatus:[
      'not_run','composed','timeout','deadline','invalid_composition',
      'incomplete','provider_unavailable','provider_schema_rejected','insufficient_inventory',
      'not_configured','load_shed','composition_unavailable',
    ].includes(f.skuCompositionReceipt?.status)
      ?f.skuCompositionReceipt.status:'not_run',
    skuCompositionProposed:safe(f.skuCompositionReceipt?.proposed,8),
    skuCompositionAdmitted:safe(f.skuCompositionReceipt?.admitted,6),
    skuCompositionExploration:safe(f.skuCompositionReceipt?.exploratoryProducts,48),
    skuCompositionParallel:f.skuCompositionReceipt?.parallel===true,
    skuCompositionAiFirst:f.skuCompositionReceipt?.aiFirst===true,
    skuCompositionRetrievalSkipped:f.skuCompositionReceipt?.retrievalSkipped===true,
    skuCompositionStudio:['nemotron','cross_lab_kimi',
      'nemotron_fallback','nemotron_plus_kimi_initial',
      'cross_lab_kimi_initial_fallback']
      .includes(f.skuCompositionReceipt?.studio)
      ?f.skuCompositionReceipt.studio:'not_run',
    skuCompositionPhotoFirst:[
      'applied','not_run','ready','invalid_input','insufficient_inventory',
      'insufficient_photo_inventory','deadline','oversized','media_unavailable',
      'invalid_visual_composition','incomplete_visual_composition',
      'visual_provider_unavailable','visual_provider_timeout',
    ].includes(f.skuCompositionReceipt?.visualMenuStatus)
      ?f.skuCompositionReceipt.visualMenuStatus:'not_run',
    // This source-of-truth guard lives on the FINAL cloud deck diagnostics,
    // not nested inside the earlier modelFashionSelection receipt.
    catalogFourItemContract:d.catalogFourItemContract===true,
    catalogFourItemDropped:safe(d.catalogFourItemDropped,12),
    // Photo-critic TEACHER: only safe discrete status and bounded sample
    // count; never emit a SKU, personal taste, merchant URL or model weights.
    photoTeacherStatus:[
      'unavailable','insufficient_independent_photos',
      'no_comparable_photo_pair','uncalibrated_on_photos','photo_validated',
      'graded_photo_validated',
    ].includes(f.photoTeacherStatus)?f.photoTeacherStatus:'not_run',
    photoTeacherExamples:safe(f.photoTeacherExamples,8),
    photoTeacherPositive:safe(f.photoTeacherPositive,8),
    photoTeacherNegative:safe(f.photoTeacherNegative,8),
    photoTeacherSelectionApplied:f.photoTeacherSelectionApplied===true,
    photoRevisionInventory:['unseen_source_skus','source_fallback',
      'remaining_source_skus'].includes(f.photoRevisionInventory)
      ?f.photoRevisionInventory:'not_run',
    photoRevisionStatus:['composed','unavailable','deadline','timeout',
      'invalid_composition','incomplete','provider_unavailable',
      'provider_schema_rejected','insufficient_inventory','load_shed']
      .includes(f.photoRevisionStatus)?f.photoRevisionStatus:'not_run',
    photoRevisionAttempted:safe(f.photoRevisionAttempted,3),
    photoRevisionApproved:safe(f.photoRevisionApproved,3),
    photoRevisionArtist:['nemotron','nemotron_fallback','cross_lab_kimi','nemotron_fast_photo_revision']
      .includes(f.photoRevisionArtist)?f.photoRevisionArtist:'not_run',
    photoEditorialStatus:['applied','deadline','timeout','provider_unavailable',
      'invalid_combination','invalid_selection','not_enough_approved']
      .includes(f.photoEditorial?.status)?f.photoEditorial.status:'not_run',
    photoEditorialChosen:safe(f.photoEditorial?.chosen,3),
    photoEditorialVisualJury:['applied','not_run','image_unavailable',
      'invalid_selection','timeout','provider_unavailable','unavailable']
      .includes(f.photoEditorial?.visualJury)
      ?f.photoEditorial.visualJury:'not_run',
    photoEditorialIndependentJury:['not_run','agreed','narrowed',
      'disagreed','unavailable','standalone',
      'arbitrated_second','arbitrated_primary']
      .includes(f.photoEditorial?.independentJury)
      ?f.photoEditorial.independentJury:'not_run',
    photoEditorialIndependentVoteSize:safe(f.photoEditorial?.independentVoteSize,3),
    photoEditorialVisualMedia:['ready','not_run','invalid_input',
      'unauthorized_or_missing','deadline','source_missing',
      'photo_processing','oversized']
      .includes(f.photoEditorial?.visualMediaStatus)
      ?f.photoEditorial.visualMediaStatus:'not_run',
    photoEditorialEligibleTriples:safe(f.photoEditorial?.eligibleTriples,24),
    photoEditorialEligiblePairs:safe(f.photoEditorial?.eligiblePairs,12),
    photoEditorialCreativeRevision:[
      'not_run','attempted','applied','insufficient_safe_capacity',
    ].includes(f.photoEditorial?.feedbackRevision)
      ?f.photoEditorial.feedbackRevision:'not_run',
    photoEditorialCreativeSource:[
      'visual_editor_gap','approved_photo_evidence',
    ].includes(f.photoEditorial?.feedbackSource)
      ?f.photoEditorial.feedbackSource:'not_run',
    photoEditorialCreativeInventory:[
      'unseen_source_skus','original_source_skus',
    ].includes(f.photoEditorial?.feedbackInventory)
      ?f.photoEditorial.feedbackInventory:'not_run',
    photoEditorialCreativeRevisionAttempted:
      safe(f.photoEditorial?.feedbackRevisionAttempted,3),
    photoEditorialCreativeRevisionApproved:
      safe(f.photoEditorial?.feedbackRevisionApproved,3),
    photoEditorialCreativeArtist:['nemotron','nemotron_fallback','cross_lab_kimi','nemotron_fast_photo_revision']
      .includes(f.photoEditorial?.feedbackRevisionArtist)
      ?f.photoEditorial.feedbackRevisionArtist:'not_run',
    formalPhotoProbe:f.formalPhotoProbe===true,
    prePhotoUniqueTriple:f.uniqueTripleAvailable===true,
    reasons,
  };
}

/**
 * Catalog-shoppable stylist contract: one COMPLETE photographed outfit has
 * exactly four different merchandise roles (top, meaningful outer layer,
 * bottom, shoes), not three items padded in the app.
 *
 * All other hard gates and NVIDIA verdicts already ran in recommendFashionDeck.
 * Never invent a fourth SKU or relabel an unjudged garment as approved.
 * Owned wardrobe / mixed-mode requests retain their original semantics.
 */
export function enforceCatalogFourItemDeck(deck,sourceMode){
  if(sourceMode!=='catalog'||!deck||!Array.isArray(deck.outfits))return deck;
  const slots=['top','outerwear','bottom','shoes'];
  const isComplete=look=>Array.isArray(look?.items)&&look.items.length===4&&
    slots.every(slot=>look.items.filter(item=>item?.macroCategory===slot).length===1)&&
    new Set(look.items.map(item=>String(item.id||''))).size===4&&
    look.items.every(item=>item?.isShopItem===true&&typeof item.id==='string');
  const kept=deck.outfits.filter(isComplete);
  return {...deck,
    outfits:kept,
    status:kept.length===3?'complete':kept.length?'partial':'no_feasible_looks',
    diagnostics:{...deck.diagnostics,
      catalogFourItemContract:true,
      catalogFourItemDropped:deck.outfits.length-kept.length},
  };
}

const REQUEST_TIMEOUT_MS = 50_000;
/** Trusted styling-first discovery vocabulary. The existing Supabase RPC
 * gives preferredSubtypes two relevance points and matching dressCode one,
 * but cloud compute previously sent ONLY budget: the first 100-per-slot
 * windows were full of irrelevant tees, pullovers and sportwear. All hints
 * are fixed backend taxonomy terms, never untrusted user text or product IDs.
 * The RPC remains read-only, bounded at 100/slot, with its unchanged source,
 * price, photo and merchant gates.
 */
export function catalogStylingDiscoveryHints({style,occasion,brief='',weather=null}={}) {
  const s=String(style||'').trim().toLowerCase().replace(/[\s-]+/g,'_');
  const event=String(occasion||'').toLowerCase();
  const intent=String(brief||'').toLowerCase().slice(0,300);
  const work=/\b(work|office|business|meeting|conference)\b/.test(event);
  const date=/\b(date|dinner|romantic|cocktail|evening)\b/.test(event);
  const casual=deliberatelyRelaxedIntent(intent);
  const quiet=['classic','semi_classic','old_money','business_casual','minimalist'].includes(s);
  const athletic=/\b(workout|gym|training|fitness|sport)\b/.test(event);
  if(athletic)return {};// dedicated activity-specific source discovery is already active
  if(!quiet)return {};
  const formal=/\b(formal|black[ -]?tie|wedding)\b/.test(event) && !casual;
  const dressCode=formal?'formal':casual?'casual':'smart';
  const workRole=['dress-shirt','shirt','blazer','coat','tailored-trouser',
    'trouser','oxford','derby','loafer','chelsea','monk'];
  // Formal is not ordinary office-smart: allocating the same 100-shoe
  // discovery slots to casual loafers/Chelseas crowded OUT real dress
  // Oxfords/Derbies/Monks. On the real 2026-11-17 catalog (247 formal
  // shoes), exact RPC replay improved explicit formal shoes from 45/100
  // to 68/100 with this narrower relevance hint. It is a SOURCE-SORT
  // preference, not a hard ban: the independent photo critic still owns
  // full-look style, footwear and occasion suitability.
  const formalRole=['dress-shirt','shirt','blazer','tailored-trouser',
    'trouser','oxford','derby','monk'];
  const dateRole=['dress-shirt','shirt','fine-knit','polo','blazer','cardigan',
    'coat','tailored-trouser','trouser','chino','loafer','derby','chelsea','oxford'];
  const weekendRole=['shirt','fine-knit','polo','cardigan','overshirt',
    'blazer','coat','chino','tailored-trouser','trouser','loafer','derby','chelsea','boot'];
  const casualRole=['shirt','polo','fine-knit','cardigan','overshirt',
    'chino','trouser','jeans','loafer','driver','minimal-sneaker','chelsea'];
  const roles=formal?formalRole:casual?casualRole:work?workRole:date?dateRole:weekendRole;
  // At <=5°C prefer coats and actual boots within the existing bounded
  // inventory rather than extra uninsulated polos or summer slippers.
  const cold=Number.isFinite(weather?.temp) && weather.temp<=5;
  return {dressCode,preferredSubtypes:[...new Set([
    ...(cold?['coat','boot','chelsea']:[]),...roles,
  ])]};
}


/** Diverse merchant offers, not 40 shoes from one maker. Input is ONLY
 * server-returned catalog RPC rows. Stable round-robin by manufacturer
 * preserves each vendor's internal relevance/price order and keeps the
 * same 100-SKU hydration budget. Unknown types follow classified shoes
 * within their manufacturer, without being relabelled as known fashion.
 */
export function balancedShoeCatalogWindow(rows, limit=CATALOG_PER_SLOT) {
  if (!Array.isArray(rows) || !Number.isInteger(limit) || limit<1 || limit>CATALOG_PER_SLOT) return [];
  const byBrand=new Map(),seen=new Set();
  for(const row of rows){
    if(!row?.id || seen.has(String(row.id)))continue;
    seen.add(String(row.id));
    const brand=String(row.brand||'unbranded').trim().toLowerCase();
    if(!byBrand.has(brand))byBrand.set(brand,[]);
    byBrand.get(brand).push(row);
  }
  const groups=[...byBrand.values()].map(group=>group.sort((a,b)=>{
    const aUnknown=(a?.fashion_intelligence?.descriptor?.subtype||a?.fashion_subtype||'unknown')==='unknown';
    const bUnknown=(b?.fashion_intelligence?.descriptor?.subtype||b?.fashion_subtype||'unknown')==='unknown';
    return Number(aUnknown)-Number(bUnknown);
  }));
  const balanced=[];
  while(balanced.length<limit){
    let progress=false;
    for(const group of groups){
      if(!group.length)continue;
      const row=group.shift();
      balanced.push(row);progress=true;
      if(balanced.length>=limit)break;
    }
    if(!progress)break;
  }
  return balanced;
}

/** Whole-outfit AI needs a catalog of real construction alternatives,
 * not 100 variants of the first manufacturer/subtype returned by an RPC.
 * Keep the SERVER-ranked source order for relevance, but reserve most of
 * this fixed-size window for distinct known garment constructions.
 * Manufacturer round-robin inside each subtype prevents one merchant from
 * monopolizing a silhouette; NO product is fabricated, promoted for beauty
 * or exempted from hard gates / independent PHOTO criticism.
 */
export function balancedGarmentCatalogWindow(rows, limit=CATALOG_PER_SLOT) {
  if(!Array.isArray(rows)||!Number.isInteger(limit)||
      limit<1||limit>CATALOG_PER_SLOT)return [];
  const unique=[],usedIds=new Set();
  const subtype=row=>{
    const value=row?.fashion_subtype||
      row?.fashion_intelligence?.descriptor?.subtype||'unknown';
    const key=String(value).trim().toLowerCase();
    return /^[a-z0-9_-]{2,35}$/.test(key)?key:'unknown';
  };
  for(const row of rows){
    const id=String(row?.id||'');
    if(!id||usedIds.has(id))continue;
    usedIds.add(id);
    unique.push(row);
  }
  const bySubtype=new Map();
  for(const row of unique){
    const key=subtype(row);
    if(!bySubtype.has(key))bySubtype.set(key,[]);
    bySubtype.get(key).push(row);
  }
  // The discovery RPC has already applied dress-code, occasion, price,
  // merchant validity and preferred subtype relevance. Preserve the
  // index's first-seen subtype priority; UNKNOWN metadata gets no invented
  // construction and is used as a last-resort source fallback.
  const groupOrder=[...bySubtype.keys()].filter(k=>k!=='unknown');
  // Unknown is not a *design direction*. Use it as honest source fallback
  // only if no classified construction exists or the remaining capacity
  // needs filling; never invent a class from merchant product titles.
  if(!groupOrder.length&&bySubtype.has('unknown'))groupOrder.push('unknown');
  const groups=groupOrder.map(k=>balancedShoeCatalogWindow(
    bySubtype.get(k),Math.min(limit,bySubtype.get(k).length)));
  const chosen=[],chosenIds=new Set();
  const add=row=>{
    if(!row||chosenIds.has(String(row.id)))return false;
    chosenIds.add(String(row.id));chosen.push(row);return true;
  };
  // A majority of the source menu exposes different REAL subtypes rather
  // than colorways. Broad catalog quality is still the AI/photo editors'
  // responsibility, not a hard-coded apparel-matching decision.
  const diversityBudget=Math.min(limit,Math.ceil(limit*.64));
  while(chosen.length<diversityBudget){
    let grew=false;
    for(const group of groups){
      if(chosen.length>=diversityBudget)break;
      const next=group.shift();
      if(next&&add(next))grew=true;
    }
    if(!grew)break;
  }
  // Remaining capacity follows the original server-ranked merchandise
  // relevance order. No required minimum for any particular garment type.
  for(const row of unique){
    if(chosen.length>=limit)break;
    add(row);
  }
  return chosen;
}

const fail = (code, status, extra = {}) => ({
  status,
  body: { success: false, outfits: [], code, error: code, generationVersion: STYLIST6_CLOUD_VERSION, ...extra },
});

const hostOf = (value) => {
  try {
    const url = new URL(String(value));
    if (url.protocol !== 'https:' || url.username || url.password || url.port) return '';
    return url.hostname;
  } catch {
    return '';
  }
};

const seasonOf = (temp) => (temp >= 25 ? 'summer' : temp < 12 ? 'winter' : 'autumn');

const normalizeSourceMode = (mode) => {
  const normalized = String(mode || 'catalog').toLowerCase();
  if (normalized === 'shop' || normalized === 'catalog') return 'catalog';
  if (normalized === 'wardrobe' || normalized === 'mixed') return normalized;
  if (normalized === 'hybrid') return 'mixed';
  return null;
};

/** Slot projection for owned wardrobe rows (S5-proven EN+RU keyword mapping). */
const WARDROBE_SLOT_RULES = [
  ['shoes', /кроссовк|кеды|туфл|ботинок|ботинки|лофер|сапог|мокасин|сандал|shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derb|footwear|brogue|oxford(?!\s*shirt)/],
  ['outerwear', /жилет|куртк|пальто|пиджак|жакет|кардиган|свитер|худи|толстовк|blazer|overcoat|topcoat|trench|parka|puffer|windbreaker|bomber|coats?|jackets?|cardigans?|sweaters?|hoodies?|vests?|pullovers?|fleece|outerwear/],
  ['bottom', /джинс|брюк|шорт|юбк|леггинс|\b(pants?|trousers?|jeans|chinos?|slacks?|joggers?|shorts?|skirts?|sweatpants?|bottoms?)\b/],
  ['top', /футболк|рубашк|блуз|майк|платье|\b(t-shirts?|tshirts?|tees?|polos?|blouses?|shirts?|tops?|dresses?)\b|upper[_ ]?body/],
];

const WARDROBE_SUBTYPE_RULES = [
  ['shirt', /\b(shirts?|рубашк|блуз)\b/],
  ['blazer', /\b(blazers?|пиджак|жакет)\b/],
  ['fine_knit', /\b(knit|sweaters?|pullovers?|cardigans?|свитер|кардиган|водолазк|гольф)\b/],
  ['polo', /\b(polos?|поло)\b/],
  ['jeans', /\b(jeans?|джинс)\b/],
  ['tailored_trouser', /\b(trousers?|slacks?|chinos?|брюк)\b/],
  ['chino', /\b(chinos?|чинос)\b/],
  ['loafer', /\b(loafers?|moccasins?|лофер|мокасин)\b/],
  ['derby', /\b(derb|brogues?|oxfords?|дерби|броги|оксфорд)\b/],
  ['minimal_sneaker', /\b(sneakers?|trainers?|кроссовк|кеды)\b/],
  ['overshirt', /\b(overshirts?|s?tuleikas?)\b/],
];

export function projectWardrobeItem(row) {
  if (!row || typeof row !== 'object') return null;
  const id = String(row.id || '').trim();
  if (!id) return null;
  const blob = `${row.category || ''} ${row.type || ''} ${row.sub_category || ''} ${row.name || ''}`.toLowerCase();
  const slot = (WARDROBE_SLOT_RULES.find(([, re]) => re.test(blob)) || [])[0] || null;
  if (!slot) return null;
  const imageUrl = String(row.image_url || '');
  if (!imageUrl.startsWith('https://')) return null;
  const subtype = (WARDROBE_SUBTYPE_RULES.find(([, re]) => re.test(blob)) || [])[0] || 'unknown';
  const color = String(
    (Array.isArray(row.color) ? row.color.join(', ') : row.primary_color || row.color) || 'neutral',
  ).slice(0, 64);
  return Object.freeze({
    id,
    name: String(row.name || row.type || row.category || 'Wardrobe item').slice(0, 160),
    brand: String(row.brand || '').slice(0, 80),
    macroCategory: slot,
    subtype,
    color,
    texture: 'unknown',
    imageUrl,
    shopUrl: null,
    price: null,
    currency: null,
    isShopItem: false,
    visualEvidenceVerified: false,
    projection: { kind: 'owned_wardrobe', evidence: [`wardrobe:${slot}/${subtype}`] },
  });
}

/**
 * Fit raw image bytes to the hosted critic contract without a native
 * toolchain: pass through bytes that are already small JPEGs, otherwise
 * abstain honestly. Never invents pixels.
 */
export function fitVisualBytesCloud(bytes) {
  if (!Buffer.isBuffer(bytes) || bytes.length < 128 || bytes.length > MAX_VISUAL_IMAGE_BYTES) return null;
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  return bytes;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = PRODUCT_FETCH_TIMEOUT_MS) {
  // Keep the signal alive while the response body streams; cancelling it
  // immediately after headers allowed slow/hung CDN bodies to consume the
  // entire 18s visual budget despite the advertised 8s merchant timeout.
  return fetch(url, { ...options, signal: options.signal || AbortSignal.timeout(timeoutMs) });
}

/**
 * Server-side IO for the cloud handler. The default implementation talks
 * to Supabase over HTTPS; tests inject fakes. No client-supplied rows are
 * ever read here.
 */
export function createCloudIo({ url, anonKey, serviceKey, supabaseJs }) {
  const authHeaders = (key) => ({ apikey: key, 'Content-Type': 'application/json' });
  const check = async (response, what) => {
    if (!response.ok) throw new Error(what);
    return response.json();
  };
  return {
    async verifyUser(jwt) {
      const { createClient } = supabaseJs;
      const client = createClient(url, anonKey, { auth: { persistSession: false, autoRefreshToken: false } });
      const { data, error } = await client.auth.getUser(jwt);
      if (error || !data?.user?.id) throw new Error('stylist6_unauthorized');
      return data.user.id;
    },
    adminClient() {
      const { createClient } = supabaseJs;
      return createClient(url, serviceKey, { auth: { persistSession: false, autoRefreshToken: false } });
    },
    // Cheap early admission: rate bucket only, never debits quota.
    async checkRate(userId) {
      const admin = this.adminClient();
      const nowMs = Date.now();
      const bucketMs = Math.floor(nowMs / 60_000) * 60_000;
      const bucket = await admin.rpc('increment_rate_bucket', {
        p_user: userId, p_feature: `rl:${QUOTA_FEATURE}:${bucketMs}`,
      });
      if (bucket.error || !Number.isSafeInteger(bucket.data) || bucket.data < 1) {
        throw new Error('stylist6_rate_unavailable');
      }
      if (bucket.data > RATE_LIMIT_PER_MINUTE) {
        const denied = new Error('stylist6_rate_limited');
        denied.code = 'rate_limited';
        throw denied;
      }
    },
    // Transactional charge: the durable logical claim is locked and marked
    // debited in the SAME Postgres transaction as consume_edge_quota. A crash
    // cannot leave an ambiguous debit without a durable quota marker.
    async debitQuota(userId, requestId) {
      const admin = this.adminClient();
      const quota = await admin.rpc('charge_stylist6_request', { p_user: userId, p_request_id: requestId });
      if (quota.error || typeof quota.data !== 'boolean') throw new Error('stylist6_rate_unavailable');
      if (!quota.data) {
        const denied = new Error('stylist6_quota_exhausted');
        denied.code = 'feature_limit_reached';
        throw denied;
      }
    },
    /**
     * Durable idempotency claim for one logical generation request via the
     * `claim_stylist6_request` RPC (atomic insert wins; concurrent
     * duplicates serialize on the primary key). There is NO process-local
     * fallback: if durable claim storage is unavailable the request fails
     * closed BEFORE any quota debit. Returns {duplicate, prior?}.
     * Duplicates never re-debit and never re-plan: the caller answers 409
     * from the recorded outcome.
     */
    async claimRequest(userId, requestId, fingerprint) {
      const admin = this.adminClient();
      let claimed;
      try {
        claimed = await admin.rpc('claim_stylist6_request', { p_user: userId, p_request_id: requestId, p_fingerprint: fingerprint });
      } catch {
        throw new Error('stylist6_claim_unavailable');
      }
      if (claimed?.error) throw new Error('stylist6_claim_unavailable');
      // Table-returning RPCs arrive as row arrays; tolerate both shapes.
      const row = Array.isArray(claimed?.data) ? claimed.data[0] : claimed?.data;
      if (!row || typeof row !== 'object' || typeof row.duplicate !== 'boolean') {
        throw new Error('stylist6_claim_unavailable');
      }
      return {
        duplicate: row.duplicate === true,
        conflict: row.outcome_code === 'request_content_conflict',
        prior: row.duplicate === true ? { status: row.status_code, code: row.outcome_code } : null,
        durable: true,
      };
    },
    /**
     * Records the terminal outcome for a claimed request so a redelivery
     * replays it via 409. Returns true when recorded; a false return is
     * surfaced on the response (`idempotencyRecorded`) but never fails an
     * otherwise successful generation — the claim itself already prevents
     * double billing.
     */
    // The long-lived native Supabase HTTP response can be lost AFTER a
    // successful paid generation (confirmed in real iOS QA). Persist
    // the EXACT already-approved deck for owner-only read-only recovery.
    // Never issue new NVIDIA calls or debit quota during a recovery read.
    async saveCompletedResult(userId,requestId,payload) {
      if(typeof userId!=='string'||typeof requestId!=='string'||
         requestId.length>128||payload?.success!==true||
         payload?.generationRequestId!==requestId||
         !Array.isArray(payload?.outfits)||
         payload.outfits.length<1||payload.outfits.length>3)
        return false;
      const admin=this.adminClient();
      try{
        const {error}=await admin.from('stylist6_completed_results')
          .insert({user_id:userId,request_id:requestId,
            response_payload:payload,
            // No wardrobe/person details beyond original response.
            // The owner cannot SELECT this after the short TTL.
            expires_at:new Date(Date.now()+45*60*1000).toISOString()})
          .abortSignal(AbortSignal.timeout(2500));
        return !error;
      }catch{return false;}
    },
    async finishRequest(userId, requestId, status, code) {
      const admin = this.adminClient();
      try {
        // Terminal failure + refund commit atomically and idempotently;
        // successful generations retain their one legitimate quota debit.
        const finished = await admin.rpc('settle_stylist6_request', {
          p_user: userId, p_request_id: requestId, p_status: status, p_code: String(code || ''),
        });
        return !finished?.error;
      } catch {
        return false;
      }
    },
    async fetchShopRows(seed = null, context = null, discoveryIntent = null) {
      const rows = new Map();
      // An unanchored fresh request gets a stable but rotated bounded catalog
      // window. Keep legacy sampling for explicit anchors (no missing locked
      // product due to a random discovery page). No client-supplied raw seed.
      const safeSeed = typeof seed === 'string' && /^[a-f0-9]{16}$/.test(seed) ? seed : null;
      // A generic (100/slot) rotation regularly omitted the *only* real
      // training shirts and returned linen polos + jeans for Workout. Work
      // from fixed, server-owned activity search terms first, then refill the
      // same per-slot windows from diversified general results. The 400-row
      // media/price hydration budget and merchant truth gates DO NOT change.
      const workout = discoveryIntent === 'workout';
      const cold = discoveryIntent === 'cold';
      const localContext = workout ? { ...(context || {}), dressCode: 'athletic' } : context;
      const hasContext = Boolean(localContext && typeof localContext === 'object' && !Array.isArray(localContext) && Object.keys(localContext).length > 0);
      const workouts = workout ? [
        { slot: 'top', search: 'training', limit: 25 },
        { slot: 'bottom', search: 'jogger', limit: 35 },
        { slot: 'shoes', search: 'running', limit: 20 },
        { slot: 'shoes', search: 'training', limit: 15 },
        { slot: 'outerwear', search: 'training', limit: 15 },
      ] : [];
      // Freezing weather needs boots and genuine coats. A generic casual
      // window previously offered blazers plus suede loafers at -2°C.
      const coldWeather = cold ? [
        { slot: 'outerwear', search: 'puffer', limit: 22 },
        { slot: 'outerwear', search: 'coat', limit: 32 },
        { slot: 'shoes', search: 'boot', limit: 25 },
      ] : [];
      const queries = [
        ...workouts, ...coldWeather,
        ...CATALOG_SLOTS.flatMap(slot=>Array.from({
          length:slot==='shoes'?SHOE_DISCOVERY_PAGES:GARMENT_DISCOVERY_PAGES,
        },(_,page)=>({
          slot,search:'',limit:CATALOG_PER_SLOT,offset:page*CATALOG_PER_SLOT,
        }))),
      ];
      const slotBatches = await Promise.all(queries.map(async (query) => {
        const rpcBody = {
          p_slot: query.slot,
          p_search: query.search,
          p_seed: query.search
            ? (safeSeed ? `s6cloud:${query.slot}:${query.search}:${safeSeed}` : `s6cloud:${query.slot}:${query.search}`)
            : (safeSeed ? `s6cloud:${query.slot}:${safeSeed}` : `s6cloud:${query.slot}`),
          p_limit: query.limit,
          p_offset: query.offset || 0,
          ...(hasContext ? { p_context: localContext } : {}),
        };
        const response = await fetch(`${url}/rest/v1/rpc/catalog_discovery_candidates`, {
          method: 'POST',
          headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
          body: JSON.stringify(rpcBody),
        });
        return { slot: query.slot, search:query.search, data: await check(response, 'stylist6_catalog_unavailable') };
      }));
      // Keep context-specific verified workout / freezing-source results
      // FIRST, then choose a broader CONSTRUCTION-BALANCED catalog window
      // from page 0 and page 1 (shoes retain their proven brand-balanced
      // three-page discovery). Identical 100/slot => 400 hydrated/returned
      // products; no extra photo-media fanout, catalog bypass or schema.
      const admittedBatches=CATALOG_SLOTS.map(slot=>{
        const focused=slotBatches.filter(b=>b.slot===slot&&b.search)
          .flatMap(b=>Array.isArray(b.data)?b.data:[]);
        const generic=slotBatches.filter(b=>b.slot===slot&&!b.search)
          .flatMap(b=>Array.isArray(b.data)?b.data:[]);
        return {slot,data:[...focused,...(slot==='shoes'
          ?balancedShoeCatalogWindow(generic,CATALOG_PER_SLOT)
          :balancedGarmentCatalogWindow(generic,CATALOG_PER_SLOT))]};
      });
      const bySlot = Object.fromEntries(CATALOG_SLOTS.map(slot => [slot, 0]));
      for (const { slot, data } of admittedBatches) {
        for (const row of Array.isArray(data) ? data : []) {
          if (bySlot[slot] >= CATALOG_PER_SLOT) break;
          if (row && row.id && !rows.has(String(row.id))) {
            // The RPC guarantees its normalized fashion slot and only active
            // items; no user-supplied inventory may be smuggled into the pool.
            rows.set(String(row.id), {
              ...row,
              is_active: row.is_active ?? true,
              fashion_subtype: row.fashion_subtype
                ?? (row.fashion_intelligence && typeof row.fashion_intelligence === 'object'
                  ? row.fashion_intelligence?.descriptor?.subtype
                  : undefined),
            });
            bySlot[slot] += 1;
          }
        }
      }
      // catalog_discovery_candidates projects the catalog SKU but omits the
      // last feed-import observation. Reread only selected IDs, bounded and
      // read-only; changed/deactivated rows are dropped. A failed REST read
      // retains UNKNOWN recency (never a fabricated current price).
      return hydrateCatalogObservations({
        rows: [...rows.values()], url, serviceKey,
      });
    },
    async fetchWardrobeRows(userId) {
      const rows = [];
      const pageSize = WARDROBE_PAGE;
      for (let offset = 0; ; offset += pageSize) {
        const params = new URLSearchParams({
          select: 'id,type,category,sub_category,color,primary_color,brand,name,description,image_url',
          user_id: `eq.${userId}`,
          order: 'id.asc',
          offset: String(offset),
          limit: String(pageSize),
        });
        const response = await fetch(`${url}/rest/v1/clothing_items?${params}`, {
          headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
        });
        const data = await check(response, 'stylist6_wardrobe_unavailable');
        if (!Array.isArray(data) || !data.length) break;
        rows.push(...data);
        if (data.length < pageSize) break;
      }
      return rows;
    },
    async fetchVisualProfiles(ids) {
      // The trusted server already caps catalog discovery at 400 SKUs.
      // Four 100-row REST reads were SERIAL, consuming precious NVIDIA time.
      // Concurrent source-hash versioned reads preserve one atomic result:
      // any failed page aborts the entire hydration, never partial evidence.
      if (!Array.isArray(ids) || ids.length > CATALOG_PER_SLOT * CATALOG_SLOTS.length) {
        throw new Error('stylist6_visual_hydration_limit_exceeded');
      }
      const chunks = [];
      for (let offset=0;offset<ids.length;offset+=VISUAL_PROFILE_CHUNK) {
        chunks.push(ids.slice(offset,offset+VISUAL_PROFILE_CHUNK));
      }
      const pages=await Promise.all(chunks.map(async chunk=>{
        const params=new URLSearchParams({
          select:'product_id,version,source_hash,image_url,profile',
          version:'eq.1',
          product_id:'in.(' + chunk.map(id => '"' + String(id).replace(/"/g,'') + '"').join(',') + ')',
        });
        const response=await fetch(url+'/rest/v1/catalog_visual_fashion?'+params,{
          headers:{...authHeaders(serviceKey),Authorization:'Bearer '+serviceKey},
        });
        const data=await check(response,'stylist6_catalog_unavailable');
        return Array.isArray(data)?data:[];
      }));
      return pages.flat();
    },
    // Privileged, consent-verified source-photo provenance; never exposed
    // to user JWT or public REST callers. Under owner opt-in only, read
    // bounded matching catalog SKU IDs, and compare again to ORIGINAL URLs.
    async fetchPhotoCacheRows(ids) {
      if(!Array.isArray(ids)||ids.length>CATALOG_PER_SLOT*CATALOG_SLOTS.length)
        throw Error('stylist6_photo_cache_query_limit');
      const chunks=[];
      for(let i=0;i<ids.length;i+=VISUAL_PROFILE_CHUNK)
        chunks.push(ids.slice(i,i+VISUAL_PROFILE_CHUNK));
      const pages=await Promise.all(chunks.map(async chunk=>{
        const params=new URLSearchParams({
          select:'product_id,source_image_url,source_hash,storage_bucket,storage_path,image_sha256,media_type,status,rights_basis,rights_reference,cache_hosting_granted,external_ai_inference_granted,rights_granted_at,rights_expires_at,last_verified_at',
          status:'eq.ready',
          product_id:'in.('+chunk.map(id=>'"'+String(id).replace(/"/g,'')+'"').join(',')+')',
        });
        const response=await fetch(url+'/rest/v1/catalog_photo_cache_v1?'+params,{
          headers:{...authHeaders(serviceKey),Authorization:'Bearer '+serviceKey},
        });
        const records=await check(response,'stylist6_photo_cache_unavailable');
        return Array.isArray(records)?records:[];
      }));
      return pages.flat();
    },
    // Gate D taste source: this user's own explicit thumbs rows, newest
    // first, bounded to the pure module's row cap. One indexed query on
    // feedback_owner_recent (user_id, created_at desc) — plus deterministic
    // request/look tiebreakers so the same rows always rank the same way.
    // Service-role key: the caller is the verified JWT user (RLS unaffected).
    async fetchFeedbackTaste(userId) {
      const params = new URLSearchParams({
        select: 'rating,concept_id,item_ids,look_signature,created_at',
        user_id: `eq.${userId}`,
        order: 'created_at.desc,request_id.desc,look_id.desc',
        limit: String(FEEDBACK_TASTE_MAX_ROWS),
      });
      const response = await fetch(`${url}/rest/v1/stylist_look_feedback?${params}`, {
        headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
      });
      const data = await check(response, 'stylist6_feedback_unavailable');
      return Array.isArray(data) ? data : [];
    },
    async fetchBudgetFxSnapshot() {
      return fetchCbuBudgetFxSnapshot();
    },
  };
}

const CBU_FX_URL = 'https://cbu.uz/ru/arkhiv-kursov-valyut/json/';
const CBU_FX_CACHE_TTL_MS = 60 * 60 * 1000;
let cbuFxCache = { expiresAt: 0, snapshot: null };

/**
 * Fetches and validates the official Central Bank of Uzbekistan daily FX
 * reference rates. Never invents rates; returns null when unavailable so
 * budgeted requests fail closed or enforce same-currency pricing honestly.
 */
export async function fetchCbuBudgetFxSnapshot(fetchFn = fetch, now = Date.now()) {
  if (cbuFxCache.snapshot && now < cbuFxCache.expiresAt) {
    try {
      return validateBudgetFxSnapshot(cbuFxCache.snapshot, now);
    } catch {
      cbuFxCache = { expiresAt: 0, snapshot: null };
    }
  }
  const response = await fetchFn(CBU_FX_URL, {
    headers: { Accept: 'application/json', 'User-Agent': 'AIWardrobe-Stylist6-Cloud/1.0' },
    signal: AbortSignal.timeout(4000),
  });
  if (!response?.ok) throw new Error(`cbu_fx_http_${response?.status || 'failed'}`);
  const payload = await response.json();
  const snapshot = validateBudgetFxSnapshot(parseCbuBudgetRates(payload, now), now);
  cbuFxCache = { expiresAt: now + CBU_FX_CACHE_TTL_MS, snapshot };
  return snapshot;
}

export function hydrateProfiles(rows, profiles) {
  const byId = new Map(profiles.map((entry) => [String(entry.product_id), entry]));
  let attached = 0, drifted = 0;
  for (const row of rows) {
    const entry = byId.get(String(row?.id));
    if (!entry || entry.version !== 1 || !entry.profile) continue;
    if (entry.image_url !== row.image_url || entry.source_hash !== sourceHash(row)) {
      drifted += 1;
      continue;
    }
    if (!currentVisual({ ...row, imageUrl: row.image_url, visualFashion: entry.profile, visualSourceHash: entry.source_hash })) {
      drifted += 1;
      continue;
    }
    row.visual_fashion = entry.profile;
    row.visual_source_hash = entry.source_hash;
    attached += 1;
  }
  return { fetched: profiles.length, attached, drifted };
}

function mapDeckItem(item, rankScore, offerVerification = null) {
  return {
    id: String(item.id),
    type: item.type || item.macroCategory,
    macroCategory: item.macroCategory,
    color: item.color || 'unknown',
    name: item.name,
    brand: item.brand || '',
    description: item.description || item.name,
    imageUrl: item.imageUrl,
    image: item.imageUrl,
    recommendation: `Stylist 6 rank · score ${Number(rankScore).toFixed(2)}`,
    isShopItem: item.isShopItem === true,
    // Truthful owned-piece provenance: the app's quote/buy pipeline skips
    // isWardrobe pieces (no price, no Buy button, zero spend).
    isWardrobe: item.isShopItem !== true,
    price: item.price ?? null,
    currency: item.currency || null,
    shopUrl: item.shopUrl || null,
    // Merchant-offer price freshness (gate F): when the server last saw this
    // listing price, as days and a bounded tier. Null/unknown when the row
    // carries no last_seen_at — never an invented recency claim. Owned
    // wardrobe pieces carry no merchant offer and stay unknown.
    priceAgeDays: Number.isFinite(item.priceAgeDays) ? item.priceAgeDays : null,
    priceFreshness: typeof item.priceFreshness === 'string' && item.isShopItem === true
      ? item.priceFreshness
      : null,
    ...(offerVerification ? { offerVerification } : {}),
  };
}

/** Same truth table as the local bridge: never relabels ranking as NVIDIA. */
export function mapCloudVisualStatus(deck) {
  const verdict = deck?.diagnostics?.visualVerdict;
  if (verdict === 'JUDGED') {
    const applied = (deck?.outfits || []).some((look) => look.visualEvidence);
    return applied ? 'NVIDIA_APPLIED' : 'NVIDIA_EVALUATED_NO_CHANGE';
  }
  if (verdict === 'ABSTAINED') return 'NVIDIA_ABSTAINED';
  return 'NVIDIA_NOT_RUN';
}

const IMAGE_FAILURE_CLASSES = new Set([
  'http_401','http_403','http_404','http_429','http_5xx',
  'invalid_media','oversize','decode_failed','fetch_failed',
  'host_not_allowlisted',
]);

function mediaFailure(code) {
  const error = new Error('product_image_unresolved');
  error.mediaFailure = IMAGE_FAILURE_CLASSES.has(code) ? code : 'fetch_failed';
  return error;
}

/** Memory-bounded streaming reader: hostile Content-Length is not trusted. */
export async function readBoundedProductImage(response, maxBytes = MAX_PRODUCT_IMAGE_BYTES) {
  if (!response || !Number.isSafeInteger(maxBytes) || maxBytes < 128 ||
    maxBytes > MAX_PRODUCT_IMAGE_BYTES) throw mediaFailure('invalid_media');
  const header = response.headers?.get?.('content-length');
  if (header && /^\d+$/.test(header) && Number(header) > maxBytes) {
    void response.body?.cancel?.().catch(() => {});
    throw mediaFailure('oversize');
  }
  if (!response.body?.getReader) throw mediaFailure('invalid_media');
  const reader = response.body.getReader();
  const parts = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      length += value.byteLength;
      if (length > maxBytes) {
        await reader.cancel().catch(() => {});
        throw mediaFailure('oversize');
      }
      parts.push(Buffer.from(value));
    }
    if (length < 128) throw mediaFailure('invalid_media');
    return Buffer.concat(parts, length);
  } finally {
    reader.releaseLock();
  }
}

/**
 * One request's in-flight and successful photo fetches are shared by URL.
 * Two concurrent looks using the same shoes no longer double-fetch the
 * merchant CDN; a failed response is shared too (no retry storm/429).
 * The cache is created INSIDE the generation handler, never persisted or
 * shared across users, requests or merchant price windows.
 */
export async function resolveProductAsset(imageUrl, cache, fetchFn) {
  const url = String(imageUrl || '');
  if (!HOST_ALLOWLIST.has(hostOf(url))) throw mediaFailure('host_not_allowlisted');
  if (!(cache instanceof Map)) throw new Error('request_image_cache_required');
  if (cache.has(url)) return cache.get(url);
  const task = (async () => {
    const get = fetchFn || fetchWithTimeout;
    let response;
    try {
      response = await get(url, {
        redirect: 'error',
        headers: { 'User-Agent': 'AIWardrobe-Stylist6-Cloud/1.0', Accept: 'image/*,*/*;q=0.8' },
        // Covers both HTTP headers and the image body. Injected test stubs
        // receive the same request shape with an abortable deadline.
        signal: AbortSignal.timeout(PRODUCT_FETCH_TIMEOUT_MS),
      });
    } catch {
      throw mediaFailure('fetch_failed');
    }
    if (!response?.ok) {
      const code = Number(response?.status);
      const failure = [401,403,404,429].includes(code) ? `http_${code}`
        : code >= 500 ? 'http_5xx' : 'fetch_failed';
      void response?.body?.cancel?.().catch(() => {});
      throw mediaFailure(failure);
    }
    let raw;
    try {
      raw = await readBoundedProductImage(response);
    } catch (error) {
      if (IMAGE_FAILURE_CLASSES.has(error?.mediaFailure)) throw error;
      throw mediaFailure('fetch_failed');
    }
    const fitted = await prepareCloudVisualMedia(raw);
    if (!fitted) throw mediaFailure('decode_failed');
    return { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
  })();
  cache.set(url, task);
  return task;
}

/** Start every independent permitted source-photo fetch in one bounded
 * wave. Eight real outfits can contain 32 distinct merchant images; awaiting
 * six-image waves serially took 4-6 CDN timeout windows (up to 48s), leaving
 * the actual 120B PHOTO JUDGE no time under the 50s request deadline.
 *
 * Media remains URL-allowlisted, bounded by resolveProductAsset's 8s timeout
 * and max image bytes, and shared in a request-scoped in-flight Map. Timebox
 * the optional *warmup*; unfinished tasks populate that exact same cache
 * while the critic starts. We never interpret a warmup failure as approval.
 */
export async function prewarmProductPhotos(urls, resolvePhoto,
  { remainingMs = () => 50_000, signal } = {}) {
  if(!Array.isArray(urls)||typeof resolvePhoto!=='function')return {attempted:0,completed:0};
  const unique=[...new Set(urls.filter(url=>typeof url==='string'&&
    isInferenceAuthorized(url)))].slice(0,MAX_VISUAL_CANDIDATES*4);
  if(!unique.length||signal?.aborted||remainingMs()<=5000)return {attempted:0,completed:0};
  let completed=0;
  // Critic fetches share these in-flight cache promises, so issuing all
  // photorequests together (max 32), instead of six serial batches, removes
  // CDN head-of-line blocking without a second fetch or provider call.
  const started=unique.map(imageUrl=>Promise.resolve()
    .then(()=>resolvePhoto({imageUrl}))
    .then(()=>{completed++;},()=>{}));
  let timer;
  try{
    const budgetMs=Math.min(6000,Math.max(1,remainingMs()-5000));
    await Promise.race([
      Promise.allSettled(started),
      new Promise(resolve=>{timer=setTimeout(resolve,budgetMs);}),
    ]);
  } finally {if(timer)clearTimeout(timer);}
  return {attempted:unique.length,completed};
}

/**
 * Reference bytes are licensed media and never ship with the bundle. When
 * the owner stages them in a private Supabase Storage bucket
 * (STYLIST6_REFERENCE_BUCKET), reference-guided judging becomes available;
 * otherwise reference admission fails honestly to NOT_RUN.
 */
async function resolveReferenceAsset(env, url, referenceId, fetchFn) {
  const bucket = String(env.STYLIST6_REFERENCE_BUCKET || '').trim();
  if (!bucket || !/^[a-zA-Z0-9_-]{1,64}$/.test(referenceId)) throw new Error('reference_unresolved');
  const entry = (referenceIndex?.images || []).find((item) => item.id === referenceId);
  const filename = entry?.filename ? String(entry.filename).split('/').pop() : '';
  if (!filename || filename.includes('..')) throw new Error('reference_unresolved');
  // First-party staged bytes only: the owner-controlled bucket is the one
  // inference-authorized reference source in cloud.
  const get = fetchFn || fetch;
  const response = await get(
    `${url}/storage/v1/object/${bucket}/${encodeURIComponent(filename)}`,
    { headers: { apikey: env.SUPABASE_SERVICE_ROLE_KEY, Authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}` } },
  );
  if (!response.ok) throw new Error('reference_unresolved');
  const raw = Buffer.from(await response.arrayBuffer());
  const fitted = await prepareCloudVisualMedia(raw);
  if (!fitted) throw new Error('reference_unresolved');
  return { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
}

// JSON object ordering must not turn an identical logical retry into a conflict.
export function canonicalRequestFingerprint(request) {
  const canonical = (value) => Array.isArray(value) ? value.map(canonical)
    : value && typeof value === 'object'
      ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]))
      : value;
  return createHash('sha256').update(JSON.stringify(canonical(request))).digest('hex');
}

/**
 * Normalizes both `{ request: { ... } }` (Stylist 6 staging contract) and
 * top-level `generate-outfits` payloads (mobile release contract) into a
 * single canonical Stylist 6 request shape. Client-supplied inventory is
 * never copied into the normalized request.
 */
export function normalizeStylist6CloudPayload(payload, env = {}) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return {};
  if (payload.request && typeof payload.request === 'object' && !Array.isArray(payload.request)) {
    return payload.request;
  }
  const styleFromPrefs = Array.isArray(payload.stylePreferences)
    ? payload.stylePreferences.find((entry) => typeof entry === 'string' && entry.trim())
    : typeof payload.stylePreferences === 'string' && payload.stylePreferences.trim()
      ? payload.stylePreferences.split(',')[0].trim()
      : '';
  const style = typeof payload.style === 'string' && payload.style.trim()
    ? payload.style.trim()
    : (styleFromPrefs ? styleFromPrefs.trim() : '');
  const occasion = typeof payload.occasion === 'string' ? payload.occasion.trim() : '';
  const prompt = typeof payload.prompt === 'string' ? payload.prompt.trim() : '';
  const brief = typeof payload.brief === 'string' && payload.brief.trim()
    ? payload.brief.trim()
    : [occasion, style, prompt].filter(Boolean).join(', ');
  const requiredIds = Array.isArray(payload.requiredIds)
    ? payload.requiredIds
    : [
        payload.anchorItemId,
        ...(Array.isArray(payload.selectedItemIds) ? payload.selectedItemIds : []),
      ].filter((id) => typeof id === 'string' && id.trim());
  const shopOnlyOccasion = /(business[_ ]?casual|team[_ ]?collaboration)/i.test(`${style} ${occasion}`);
  const sourceMode = payload.sourceMode !== undefined
    ? payload.sourceMode
    : payload.wardrobeOnly === true
      ? 'wardrobe'
      : (payload.catalogOnly === true || payload.useProvidedWardrobeOnly === true || shopOnlyOccasion)
        ? 'catalog'
        : 'mixed';
  const rawPrefs = payload.preferences && typeof payload.preferences === 'object' && !Array.isArray(payload.preferences)
    ? payload.preferences
    : {};
  const { maxShopBudget, ...preferences } = rawPrefs;
  const budget = payload.budget !== undefined ? payload.budget : maxShopBudget;
  const extractExcludedIds = (list) => (Array.isArray(list) ? list : [])
    .map((entry) => String(entry || '').trim())
    .filter(Boolean)
    .map((entry) => (entry.startsWith('id:') ? entry.slice(3).trim() : (!entry.includes(':') ? entry : '')))
    .filter(Boolean);
  const excludeItemIds = [...new Set([
    ...extractExcludedIds(payload.excludeItemIds),
    ...extractExcludedIds(payload.excludeProductKeys),
    ...extractExcludedIds(payload.previousDeckProductKeys),
    ...extractExcludedIds(payload.olderRecentProductKeys),
    ...extractExcludedIds(payload.recentShoeProductKeys),
    ...extractExcludedIds(payload.previousShoeProductKeys),
  ])];
  const recentSignatures = Array.isArray(payload.recentSignatures)
    ? payload.recentSignatures
    : (Array.isArray(payload.recentLookSignatures) ? payload.recentLookSignatures : []);
  const weather = payload.weather && typeof payload.weather === 'object' && !Array.isArray(payload.weather)
    ? {
        ...(typeof payload.weather.temp === 'number' && Number.isFinite(payload.weather.temp) ? { temp: payload.weather.temp } : {}),
        ...(typeof payload.weather.condition === 'string' && payload.weather.condition.trim() ? { condition: payload.weather.condition.trim() } : {}),
      }
    : undefined;
  const enableVisual = payload.enableVisual !== undefined
    ? Boolean(payload.enableVisual)
    : payload.visualEvaluation !== undefined
      ? Boolean(payload.visualEvaluation)
      : /^(1|true)$/i.test(String(env?.STYLIST6_DEFAULT_VISUAL_EVAL || '').trim());
  return {
    style,
    occasion,
    brief,
    sourceMode,
    ...(weather ? { weather } : {}),
    ...(budget !== undefined ? { budget } : {}),
    preferences,
    requiredIds,
    excludeItemIds,
    recentSignatures,
    ...(payload.generationRequestId !== undefined ? { generationRequestId: payload.generationRequestId } : {}),
    ...(payload.language !== undefined ? { language: payload.language } : {}),
    ...(enableVisual ? { enableVisual: true } : {}),
  };
}

/**
 * Authenticated Stylist 6 cloud generation. `io` performs all Supabase IO
 * (injectable fakes in tests); `env` carries server-only configuration.
 * Returns {status, body} with an edge-shaped payload.
 */
export async function handleStylist6CloudGenerate({ method, headers, bodyText, env, io, fetchImpl, timeoutMs = REQUEST_TIMEOUT_MS }) {
  const secret = env.STYLIST6_NODE_SHARED_SECRET || '';
  const previousSecret = env.STYLIST6_NODE_SHARED_SECRET_PREVIOUS || '';
  if (!secret || secret.length < 32) return fail('stylist_compute_unconfigured', 503);
  if (method !== 'POST') return fail('method_not_allowed', 405);
  const body = String(bodyText || '');
  if (new TextEncoder().encode(body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return fail('invalid_generation_request', 413);
  }
  const authorized = await verifyStylistTransport({
    secret, previousSecret, body, authorization: headers.authorization || '',
    timestamp: headers['x-aiw-timestamp'], nonce: headers['x-aiw-nonce'], signature: headers['x-aiw-signature'],
  });
  if (!authorized) return fail('unauthorized', 401);

  let payload;
  try {
    payload = JSON.parse(body);
  } catch {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const request = normalizeStylist6CloudPayload(payload, env);
  const { style, occasion } = request;
  if (typeof style !== 'string' || !style || typeof occasion !== 'string' || !occasion) {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  let sourceMode = normalizeSourceMode(request.sourceMode);
  if (!sourceMode) return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  // Display language for the deck's editorial copy. Never an admission
  // error: an unsupported value degrades to English per output string.
  const language = normalizeStylistLanguage(request.language);
  if (request.budget != null && (typeof request.budget !== 'object' ||
      typeof request.budget.amount !== 'number' || !Number.isFinite(request.budget.amount) ||
      request.budget.amount <= 0 || !/^[A-Z]{3}$/.test(String(request.budget.currency)))) {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const budget = request.budget ?? undefined;
  if (budget && !/^[A-Z]{3}$/.test(String(budget.currency))) {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const requiredIds = [...new Set((Array.isArray(request.requiredIds) ? request.requiredIds : [])
    .map((id) => String(id || '').trim()).filter(Boolean))].slice(0, 8);
  const excludedIds = [...new Set((Array.isArray(request.excludeItemIds) ? request.excludeItemIds : [])
    .map((id) => String(id || '').trim()).filter(Boolean))].slice(0, 64);
  const clientInventoryIgnored =
    (Array.isArray(payload?.catalogRows) && payload.catalogRows.length > 0) ||
    (Array.isArray(payload?.wardrobeItems) && payload.wardrobeItems.length > 0);

  // JWT first: no user, no inventory, no quota touch.
  const authorization = headers.authorization || '';
  const jwt = /^Bearer\s+(\S+)$/i.exec(authorization)?.[1] || '';
  let userId;
  try {
    userId = await io.verifyUser(jwt);
  } catch {
    return fail('unauthorized', 401, { stage: 'stylist6_admission' });
  }

  // Cheap early rate limit: bucket only, never debits quota.
  try {
    await io.checkRate(userId);
  } catch (error) {
    if (error?.code === 'rate_limited') return fail('rate_limited', 429, { stage: 'stylist6_admission' });
    return fail('rate_limit_unavailable', 503, { stage: 'stylist6_admission' });
  }

  const budgetMs = Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : REQUEST_TIMEOUT_MS;
  const started = Date.now();
  const phaseAt = {};
  let activePhase = 'fx_and_admission';
  let phaseStart = started;
  const nextPhase = (name) => {
    phaseAt[activePhase] = Date.now() - phaseStart;
    activePhase = name; phaseStart = Date.now();
  };
  const remainingMs = () => budgetMs - (Date.now() - started);
  const deadline = () => {
    if (remainingMs() <= 0) throw new Error('stylist6_timeout');
  };
  const deadlineController = new AbortController();
  const runWithinRemainingDeadline = async (work) => {
    const left = remainingMs();
    if (left <= 0) throw new Error('stylist6_timeout');
    let timer;
    try {
      return await Promise.race([
        Promise.resolve().then(work),
        new Promise((_, reject) => {
          timer = setTimeout(() => {
            deadlineController.abort();
            reject(new Error('stylist6_timeout'));
          }, left);
        }),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  };
  // Logical request identity: an explicit client id when supplied,
  // otherwise the body hash (identical bytes = identical request).
  const requestId = (() => {
    const explicit = String(request.generationRequestId || '').trim();
    if (/^[A-Za-z0-9_-]{1,128}$/.test(explicit)) return explicit;
    return `body-${createHash('sha256').update(body).digest('hex').slice(0, 32)}`;
  })();
  const finish = (status, code) => io.finishRequest(userId, requestId, status, code).catch(() => {});
  let claimed = false;
  let debited = false;
  try {
    // Owned wardrobe pieces are free. When a budget is requested, cross-currency
    // pricing uses only a validated CBU FX snapshot; without valid FX we never
    // invent exchange rates and strictly admit same-currency shop items only.
    let fxSnapshot = null;
    let fxUnavailableReason = null;
    // A reference-rate snapshot also lets unbudgeted New Clothing compare
    // a €595 luxury jacket with genuinely cheaper UZS/EUR/GBP jackets.
    // A failed/slow CBU request degrades to the existing native-currency
    // ranking; it never invents rates or rejects the user's outfit.
    if (budget || sourceMode === 'catalog') {
      if (typeof io.fetchBudgetFxSnapshot === 'function') {
        try {
          fxSnapshot = validateBudgetFxSnapshot(await io.fetchBudgetFxSnapshot());
        } catch {
          fxSnapshot = null;
          fxUnavailableReason = 'fx_unavailable';
        }
      } else {
        fxUnavailableReason = 'fx_not_configured';
      }
    }
    const budgetDiscoveryContext=budget && !requiredIds.length && BUDGET_CURRENCIES.includes(String(budget.currency))
      ? {
          budget: budget.amount,
          currency: budget.currency,
          ...(fxSnapshot?.uzsPerUnit ? { rates: fxSnapshot.uzsPerUnit } : {}),
        }
      : {};
    // Explicitly pinned wardrobe/catalog anchors need the original stable
    // full-source window: a style-biased 100-SKU cap might hide the exact
    // requested product before the anchor-lookup and wrongly report 422.
    const stylingDiscoveryContext=requiredIds.length?{}:catalogStylingDiscoveryHints({
      style,occasion,brief:request.brief,weather:request.weather,
    });
    const discoveryContext=Object.keys({...budgetDiscoveryContext,...stylingDiscoveryContext}).length
      ? {...budgetDiscoveryContext,...stylingDiscoveryContext} : null;
    // Server-side inventory only. Shop rows come from the discovery RPC;
    // wardrobe rows come from clothing_items filtered by the verified user.
    // Stable over retries of identical request bytes, different on fresh
    // requests (including a new client request id), private user scope; no
    // extra RPCs or cross-user data. Keep anchored inventory stable.
    const stableSourceCohort=shouldFreezeStylist6CatalogCohort(env,{
      budgetMs,sourceMode,hasAnchors:requiredIds.length>0,
    });
    const discoverySeed=requiredIds.length?null:stylist6CatalogDiscoverySeed(
      userId,body,{stableCohort:stableSourceCohort,style,occasion,sourceMode,
        weather:request.weather,budget});
    nextPhase('catalog_discovery');
    const [shopRows, wardrobeRows] = await Promise.all([
      sourceMode === 'wardrobe' ? [] : io.fetchShopRows(discoverySeed, discoveryContext,
        isWorkoutOccasion(request.occasion) ? 'workout' : Number(request.weather?.temp) < 5 ? 'cold' : null),
      sourceMode === 'catalog' ? [] : io.fetchWardrobeRows(userId),
    ]);
    deadline();
    nextPhase('visual_profile_hydration');
    // The photo-first experiment is OFF without both server-owned opt-ins.
    // Old iOS / wardrobe / prod Stylist5 make NO new DB reads or CDN calls.
    const photoCacheOptIn=sourceMode==='catalog'&&
      env.STYLIST6_PHOTO_FIRST_COMPOSER_ENABLED==='true'&&
      env.STYLIST6_PHOTO_ASSETS_READY==='source-hash-verified';
    const [visualHydration,photoCacheRecords]=await Promise.all([
      io.fetchVisualProfiles(shopRows.map((row)=>row.id))
        .then(profiles=>hydrateProfiles(shopRows,profiles)),
      photoCacheOptIn&&typeof io.fetchPhotoCacheRows==='function'
        ?io.fetchPhotoCacheRows(shopRows.map(row=>row.id)).catch(()=>[])
        :Promise.resolve([]),
    ]);
    deadline();
    const { items: shopItems } = adaptCatalog(shopRows);
    // Bound every cached image to the current immutable merchant SOURCE
    // record. Missing rights, expired evidence, source drift or a failed
    // private read is NOT a usable photo candidate.
    const currentShopById=new Map(shopRows.map(row=>[String(row.id),row]));
    const verifiedPhotoRows=new Map();
    for(const row of photoCacheRecords){
      const source=currentShopById.get(String(row?.product_id));
      const item=source?{id:source.id,imageUrl:source.image_url}:null;
      if(item&&verifiedCatalogPhotoCache(item,row,{
        supabaseOrigin:env.SUPABASE_URL,
      }))verifiedPhotoRows.set(String(item.id),row);
    }
    let wardrobeItems = wardrobeRows.map(projectWardrobeItem).filter(Boolean);
    if (sourceMode === 'catalog' && !payload?.request && requiredIds.length > 0) {
      const shopIdSet = new Set(shopItems.map((item) => String(item.id)));
      const hasMissingCatalogAnchor = requiredIds.some((id) => !shopIdSet.has(id) && !(id.startsWith('shop_') && shopIdSet.has(id.slice(5))));
      if (hasMissingCatalogAnchor) {
        const anchorWardrobeRows = await io.fetchWardrobeRows(userId);
        deadline();
        const requiredSet = new Set(requiredIds);
        const lockedWardrobeAnchors = anchorWardrobeRows
          .map(projectWardrobeItem)
          .filter((item) => item && requiredSet.has(String(item.id)));
        if (lockedWardrobeAnchors.length > 0) {
          wardrobeItems = lockedWardrobeAnchors;
          sourceMode = 'mixed';
        }
      }
    }
    const pool = sourceMode === 'catalog' ? shopItems
      : sourceMode === 'wardrobe' ? wardrobeItems
        : [...wardrobeItems, ...shopItems];
    if (!pool.length) {
      return fail('stylist6_empty_catalog', 502, { stage: 'catalog_adaptation', serverInventory: true, clientInventoryIgnored });
    }
    nextPhase('claim_and_feedback');
    // Locked anchors are validated against the server pool BEFORE any
    // quota debit: invalid anchors fail without a generation charge and
    // are never silently dropped.
    const poolIds = new Set(pool.map((item) => String(item.id)));
    const resolvePoolId = (id) => (poolIds.has(id) ? id : id.startsWith('shop_') && poolIds.has(id.slice(5)) ? id.slice(5) : id);
    const resolvedRequiredIds = requiredIds.map(resolvePoolId);
    const resolvedExcludedIds = excludedIds.map(resolvePoolId);
    const missingAnchors = resolvedRequiredIds.filter((id) => !poolIds.has(id));
    if (missingAnchors.length) {
      return fail('anchor_not_found', 422, { stage: 'stylist6_admission', missingAnchors, serverInventory: true });
    }
    // Idempotency: durable claim first. A redelivered logical request
    // replays its recorded outcome without another debit and without
    // re-planning. Unavailable claim storage fails closed before debit.
    let claim;
    try {
      claim = await io.claimRequest(userId, requestId, canonicalRequestFingerprint(request));
    } catch {
      return fail('stylist6_claim_unavailable', 503, { stage: 'stylist6_admission' });
    }
    if (claim.conflict) return fail('request_content_conflict', 409, { stage: 'stylist6_admission' });
    if (claim.duplicate) {
      const prior = claim.prior || {};
      const hasTerminalStatus = typeof prior.status === 'number' && prior.status > 0;
      return fail('duplicate_request', 409, {
        stage: 'stylist6_admission',
        priorStatus: prior.status ?? null,
        priorCode: prior.code || null,
        outcomeState: hasTerminalStatus ? 'terminal_recorded' : 'in_progress_or_interrupted',
        retryWithNewRequestId: !hasTerminalStatus,
        replayProtection: 'durable',
      });
    }
    claimed = true;
    // Paid debit exactly once, after full admission. Planning failures
    // keep the debit, mirroring production semantics.
    try {
      await io.debitQuota(userId, requestId);
      debited = true;
    } catch (error) {
      if (error?.code === 'feature_limit_reached') {
        await finish(403, 'feature_limit_reached');
        return fail('feature_limit_reached', 403, { stage: 'stylist6_admission' });
      }
      await finish(503, 'rate_limit_unavailable');
      return fail('rate_limit_unavailable', 503, { stage: 'stylist6_admission' });
    }
    const currencyConversion = !budget
      ? { active: false, reason: 'unbudgeted' }
      : fxSnapshot
        ? {
            active: true,
            source: fxSnapshot.source,
            asOf: fxSnapshot.asOf,
            snapshotId: fxSnapshot.id,
            budgetCurrency: budget.currency,
            snapshot: fxSnapshot,
          }
        : { active: false, reason: fxUnavailableReason || 'same_currency_only', budgetCurrency: budget.currency };
    const priceToBudgetCurrency = fxSnapshot
      ? (item, currency) => {
          const converted = referenceAmount(item.price, item.currency, currency, fxSnapshot);
          return Number.isFinite(converted) ? converted : NaN;
        }
      : (item, currency) => (item.currency === currency ? item.price : NaN);
    const feasible = budget
      ? pool.filter((item) => item.isShopItem !== true || Number.isFinite(priceToBudgetCurrency(item, budget.currency)))
      : pool;
    if (!feasible.length) {
      // Terminal outcome of an already-claimed (and debited) request: record
      // it like the empty-deck path below so a redelivery replays 409 with
      // the real prior code instead of an unrecorded "processing" claim.
      await finish(502, 'stylist6_no_feasible_looks');
      return fail('stylist6_no_feasible_looks', 502, {
        stage: 'stylist6_ranking',
        serverInventory: true,
        currencyConversion,
      });
    }
    // Gate D taste: one bounded, indexed read of this user's explicit thumbs
    // feedback, then a pure, decayed, capped model. This sits on the ranking
    // latency path (a single feedback_owner_recent query); ANY failure here
    // — transport, schema, io shape, missing io method — degrades to an
    // empty set plus an honest diagnostic field, never a failed generation.
    let feedbackRows = [];
    let feedbackTasteStatus = 'ok';
    try {
      if (typeof io.fetchFeedbackTaste === 'function') {
        feedbackRows = (await io.fetchFeedbackTaste(userId)).slice(0, FEEDBACK_TASTE_MAX_ROWS);
      } else {
        feedbackTasteStatus = 'not_wired';
      }
    } catch {
      feedbackRows = [];
      feedbackTasteStatus = 'read_failed';
    }
    const feedbackTaste = feedbackTasteModel(feedbackRows, Date.now());
    const weather = resolveStylistWeather(request.brief || '', request.weather || {});
    const temp = Number(weather?.temp);
    const refs = referenceFirstBriefs(referenceIndex, {
      style, occasion, season: Number.isFinite(temp) ? seasonOf(temp) : undefined,
    });
    // Same metadata-reference match the local bridge uses: the reference is
    // an input common to both engines, only when the shortlist contains the
    // requested style and occasion.
    const norm = (v) => String(v || '').toLowerCase().replaceAll(' ', '_');
    const reference = refs.find(({ referenceId }) => {
      const ref = (referenceIndex?.images || []).find((entry) => entry.id === referenceId);
      return Array.isArray(ref?.styles) && Array.isArray(ref?.occasions)
        && ref.styles.includes(style) && ref.occasions.includes(norm(occasion));
    }) || null;
    const input = {
      brief: request.brief || '',
      weather,
      referenceId: reference?.referenceId,
      recentSignatures: Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : [],
      excludedIds: resolvedExcludedIds,
      requiredIds: resolvedRequiredIds,
      sourceMode,
      budget,
      style,
      occasion,
      // Owner requirement: every shoppable CATALOG look has 4 real items.
      // The fourth is a meaningful photo-verifiable layer (not a filler).
      // Other existing owned/mixed compositions retain optional layering.
      optionalLayer: !(sourceMode==='catalog' && budgetMs>=70000),
      preferences: request.preferences && typeof request.preferences === 'object' ? request.preferences : {},
      // Language pass-through: fashionStylingContext spreads the request into
      // the styling context, so deck_editorial_copy reads it from there. The
      // fingerprint includes it — copy language is request content.
      language,
    };
    // INVENTORY-GROUNDED CREATIVE DIRECTOR (staging canary).
    // One short, text-only model call plans three actual fashion architectures
    // before deterministic SKU retrieval. Only counts/subtypes and explicit
    // style preferences are sent; no user ID, merchant link or garment photo.
    // A failed AI plan never bypasses a constraint or blocks the existing
    // house-formula fallback. Enable ONLY on the isolated S6 compute preview.
    const creativeDirectorEnabled =
      /^(1|true)$/i.test(String(env.STYLIST6_CREATIVE_DIRECTOR_ENABLED || '').trim()) &&
      !isWorkoutOccasion(occasion) && remainingMs() >= 12000;
    nextPhase('creative_architecture');
    const creativePlan = creativeDirectorEnabled
      ? await designCreativeBriefsWithFallback({
          catalog: feasible, request: input,feedbackTaste,
          allowIndependentFallback:budgetMs>=70000&&remainingMs()>=33000,
          parallelIndependentFallback:budgetMs>=70000,
          // Both models now start together, NOT after a slow failed 120B.
          // This preserves the 72s deadline and gives real 30B creativity
          // ~6.5s instead of a near-guaranteed 3s sequential timeout.
          backupTimeoutMs:budgetMs>=70000?8500:3000,
          apiKey: env.NVIDIA_API_KEY || '',
          ...(fetchImpl ? { fetchImpl } : {}),
          signal: deadlineController.signal,
          timeoutMs: Math.min(7500, Math.max(1000, remainingMs() - 7000)),
          // Only 72s isolated staging: professional 120B creative director.
          // 50s preview and untouched production S5 keep original behavior.
          highTaste: budgetMs>=70000,
        })
      : { status: 'disabled_or_deadline', briefs: [], providerCalls: 0 };
    // Hosted NVIDIA review: strictly opt-in, server-side key only, bounded
    // to the critic's own batch limits. Reference-guided judging needs
    // staged reference bytes; without them admission fails honestly and the
    // deck completes deterministically as NOT_RUN. Looks without a matching
    // reference are judged purely on product images (reference-optional).
    // STYLIST6_VISUAL_DISABLED=1 (or true) is the owner kill-switch: it
    // forces an honest NOT_RUN even when the client asks for review.
    let visualCritic, canJudgeOutfit;
    let canDisplayOutfit;
    const visualEvents = [];
    const visualAsked = request.enableVisual === true;
    const visualDisabled = ['1', 'true'].includes(String(env.STYLIST6_VISUAL_DISABLED || '').trim().toLowerCase());
    const explicitVisualFlag = payload?.enableVisual !== undefined
      || payload?.visualEvaluation !== undefined
      || payload?.request?.enableVisual !== undefined
      || payload?.request?.visualEvaluation !== undefined;
    const visualPolicy = visualDisabled
      ? 'server_killswitch'
      : explicitVisualFlag
        ? (visualAsked ? 'explicit_client' : 'default_off')
        : (/^(1|true)$/i.test(String(env.STYLIST6_DEFAULT_VISUAL_EVAL || '').trim()) ? 'server_default' : 'default_off');
    const nvidiaKey = env.NVIDIA_API_KEY || '';
    const photoCache = new Map();
    const verifiedPhotoTasks=new Map();
    // Exact image bytes are shared with the independent photo-critic cache.
    let photoPortfolioResolver=null;
    let photoComposerCachedOnlyResolver=null;
    if (visualAsked && !visualDisabled && nvidiaKey.startsWith('nvapi-')) {
      const resolveProduct=(product)=>{
        const url=product.imageUrl||product.image_url;
        const originalResolver=()=>resolveProductAsset(url,photoCache,
          fetchImpl?(prodUrl,options)=>fetchImpl(prodUrl,options):undefined);
        const record=verifiedPhotoRows.get(String(product?.id||''));
        if(!record||record.source_image_url!==url)return originalResolver();
        const taskKey=String(product.id)+'|'+url;
        if(!verifiedPhotoTasks.has(taskKey)){
          const task=resolveCatalogCachedPhoto({
            item:product,record,supabaseOrigin:env.SUPABASE_URL,
            fetchImpl:fetchImpl||fetch,decodeImage:prepareCloudVisualMedia,
            originalResolver,
          });
          verifiedPhotoTasks.set(taskKey,task);
        }
        return verifiedPhotoTasks.get(taskKey);
      };
      photoPortfolioResolver=resolveProduct;
      // Creative composer NEVER triggers a live merchant-CDN wave:
      // it uses ONLY source-verified licensed FIRST-PARTY assets. Missing,
      // expired or corrupted cached assets abstain; the original text AI
      // composer remains the bounded fallback. The INDEPENDENT quality
      // critic above is still allowed to fetch original merchant media.
      photoComposerCachedOnlyResolver=product=>{
        const record=verifiedPhotoRows.get(String(product?.id||''));
        if(!record||record.source_image_url!==(product?.imageUrl||product?.image_url))
          return null;
        return resolveCatalogCachedPhoto({
          item:product,record,supabaseOrigin:env.SUPABASE_URL,
          fetchImpl:fetchImpl||fetch,decodeImage:prepareCloudVisualMedia,
          originalResolver:async()=>null,deadlineMs:1600,
        });
      };
      // Even an explicitly unreviewed third catalog card must have real,
      // decodable product photos. Reuse the request-scoped image cache and
      // avoid new paid NVIDIA calls. Never let expired request time preflight.
      canDisplayOutfit=async(look)=>{
        if(remainingMs()<5000 || !Array.isArray(look?.items) || look.items.length<3)return false;
        const urls=[...new Set(look.items.map(item=>item.imageUrl||item.image_url))];
        if(urls.length<3 || urls.some(url=>!isInferenceAuthorized(url)))return false;
        const checked=await Promise.allSettled(urls.map(imageUrl=>resolveProduct({imageUrl})));
        return checked.every(item=>item.status==='fulfilled' && item.value?.bytes?.length>0);
      };
      const rawVisualCritic = createHostedVisualCritic({
        provider: 'nvidia',
        apiKey: nvidiaKey,
        model: NVIDIA_VISION_MODEL,
        // Referenceless visual critique is allowed only when the entire
        // reference bucket was never configured. If an explicitly configured
        // bucket cannot serve the selected reference, the critic abstains.
        referenceMediaConfigured: Boolean(String(env.STYLIST6_REFERENCE_BUCKET || '').trim() &&
          explicitLookbookPhotoRequested(request,reference)),
        referenceBlueprints: buildReferenceBlueprints(referenceIndex),
        timeoutMs: Math.min(VISUAL_TIMEOUT_MS, Math.max(1000, remainingMs() - 1000)),
        providerTimeoutMs: VISUAL_PROVIDER_TIMEOUT_MS,
        // All eight admissible alternatives must be judged in a SINGLE
        // bounded wave. At concurrency six, the seventh and eighth were
        // queued behind slow 120B calls, causing late 504/partial decks.
        // The existing 38s provider batch and 50s request deadlines still
        // apply; no unreviewed look is ever labeled approved.
        concurrency: 8,
        ...(fetchImpl ? { fetchImpl } : {}),
        resolveReference: (id, signal) => resolveReferenceAsset(env, env.SUPABASE_URL, id,
          fetchImpl ? (refUrl, refOptions) => fetchImpl(refUrl, { ...refOptions, signal }) : undefined),
        resolveProduct,
        onEvent: (event) => {
          visualEvents.push({
            outcome: event.outcome,
            dispatched: event.dispatched === true,
            reason: event.reason || null,
            ...(event.mediaSlot ? { mediaSlot: event.mediaSlot } : {}),
            ...(event.mediaFailure ? { mediaFailure: event.mediaFailure } : {}),
            httpStatus: event.httpStatus ?? null,
            latencyMs: Math.round(event.latencyMs || event.providerMs || 0),
            ...(Number.isSafeInteger(event.images) ? { images: event.images } : {}),
            ...(event.cacheHit === true ? { cacheHit: true } : {}),
            ...(event.retried === true ? { retried: true } : {}),
            ...(event.usage && typeof event.usage === 'object' ? { usage: event.usage } : {}),
          });
        },
      });
      // Bounded concurrent media pre-warm before Wave 1 hits the strict NIM
      // deadline so merchant CDN fetches never wait behind each other inside
      // provider evaluation. Only touches URLs already permitted for inference.
      visualCritic = async (candidates, critiqueRequest, criticOptions = {}) => {
        const urls = [...new Set(
          (Array.isArray(candidates) ? candidates : []).flatMap((look) =>
            (look?.items || [])
              .map((item) => item?.imageUrl || item?.image_url)
              .filter((imageUrl) => isInferenceAuthorized(imageUrl)),
          ),
        )];
        // A second/third photo wave must share its own shorter deadline
        // through BOTH merchant prewarm and the NVIDIA critic. Otherwise a
        // CDN stall can consume the remaining 72s staging request budget.
        const combinedSignal = criticOptions?.signal
          ? AbortSignal.any([deadlineController.signal, criticOptions.signal])
          : deadlineController.signal;
        // CRITICAL: use the tested parallel prewarm on the REAL request
        // path. Previously the helper existed but this closure still ran
        // serial batches of six; formal 4-slot looks repeatedly timed out.
        // The critic itself retains all photo authorization/quality gates.
        nextPhase('merchant_photos');
        await prewarmProductPhotos(urls, resolveProduct, {
          remainingMs, signal: combinedSignal,
        });
        nextPhase('nvidia_visual_critic');
        return rawVisualCritic(candidates, critiqueRequest, { ...criticOptions, signal: combinedSignal });
      };
      // Transmission gate, separate from download eligibility: every product
      // image must carry explicit external-inference authorization, or the
      // look completes as NOT_RUN with full deterministic ranking.
      canJudgeOutfit = async (look) => (look?.items || []).every((item) =>
        isInferenceAuthorized(item.imageUrl || item.image_url));
    }
    // Server-only canary: one short model decision over admitted WHOLE looks.
    // It cannot bypass SKU/merchant/weather/budget gates or visual approval.
    // Do not couple the SKU-only slate to the separate freeform architecture
    // planner, which is deliberately disabled for workouts. That coupling
    // stranded Sport on eight overlapping deterministic finalists: four
    // independently photo-approved sets but only two SKU-disjoint cards.
    const modelSlateEnabled=serverFashionSlateEnabled(env,{
      visualAsked,visualCriticAvailable:!!visualCritic,
    });
    // REAL native iOS fashion QA on dbefcdd/6ae1c2c/45428a8 showed
    // merchant photo-board TIMEOUT and only 1/6 approved looks. Do not
    // impose unreliable multi-CDN downloads on end users by default.
    // Reenable PHOTO-FIRST only after a source-hash-verified, explicitly
    // authorized original-photo cache can provide all four garment roles.
    // TEXT AI-first and the proven independent final PHOTO editor remain on.
    // Server-owned license/cache gate AND actual >=3 bound source images
    // PER garment role. Never start 48 live merchant CDN downloads to
    // discover during a paying user's request that media is unavailable.
    const verifiedPhotoCounts=new Map([
      ['top',0],['outerwear',0],['bottom',0],['shoes',0],
    ]);
    // Count only products still eligible for this exact catalog request.
    // A license for an out-of-currency, unavailable or filtered SKU cannot
    // satisfy the required FOUR-role source PHOTO quorum.
    for(const item of feasible)if(item.isShopItem===true&&
      verifiedPhotoRows.has(String(item.id))&&
      verifiedPhotoCounts.has(item.macroCategory))
      verifiedPhotoCounts.set(item.macroCategory,
        verifiedPhotoCounts.get(item.macroCategory)+1);
    const photoFirstCatalogReady=photoCacheOptIn &&
      [...verifiedPhotoCounts.values()].every(n=>n>=3);
    const selectDirectorFinalists=modelSlateEnabled
      ? async({options,context})=>{
          const timeLeftMs=Math.max(0,Math.floor(remainingMs()));
          const timeoutMs=fashionSlateDecisionBudgetMs(timeLeftMs);
          if(!timeoutMs)return {status:'deadline',tokens:[],providerCalls:0,timeLeftMs};
          const result=await selectFashionSlate({
            options,context,apiKey:nvidiaKey,
            ...(fetchImpl?{fetchImpl}:{}),
            signal:deadlineController.signal,
            timeoutMs,
          });
          return {...result,timeLeftMs,timeoutMs};
        } : null;
    deadline();
    nextPhase('catalog_fashion_and_selection');
    // An actual personal stylist must COMPOSE from real clothing pieces.
    // Keep this new paid text call on the isolated 72s staging only; the
    // production S5 route and 50s Vercel preview do not pay for it.
    const selectCatalogCompositions=modelSlateEnabled&&budgetMs>=70000 &&
      sourceMode==='catalog' && !requiredIds.length
      ? async({pool,context,directions,layerRequired,allowLayer,outfitCount=3,
          feedbackTaste,parallel=false})=>{
          const left=remainingMs();
          if(left<27000)return {status:'deadline',looks:[],providerCalls:0};
          // Parallel creative call starts BEFORE the old 18-21s retrieval.
          // Do not switch the sequential phase timer while that retrieval
          // continues: this call overlaps it by design, without changing
          // the native Supabase gateway, safety or paid photo-review budget.
          if(!parallel)nextPhase('ai_direct_sku_composition');
          return composeCatalogSkuLooks({
            pool,context,directions,layerRequired,allowLayer,outfitCount,
            feedbackTaste,parallel,
            // Isolated 72s cloud staging: let a SECOND independent designer
            // propose genuinely different photo-verifiable outfits BEFORE
            // the expensive final visual critic, rather than recolor the
            // first model's exhausted cardigans after it is too late.
            initialStudioDuel:parallel&&budgetMs>=70000,
            resolvePhoto:parallel&&photoFirstCatalogReady
              ?photoComposerCachedOnlyResolver:null,
            authorizePhoto:parallel&&photoFirstCatalogReady?isInferenceAuthorized:null,
            apiKey:nvidiaKey,...(fetchImpl?{fetchImpl}:{}),
            signal:deadlineController.signal,
            timeoutMs:parallel
              ?Math.min(16000,Math.max(2000,left-24500))
              :Math.min(3000,Math.max(500,left-24500)),
          });
        }:null;
    const selectCatalogRescueCompositions=selectCatalogCompositions
      ? async({pool,context,directions,critiqueFeedback,layerRequired,allowLayer,feedbackTaste})=>{
          const left=remainingMs();
          if(left<16500)return {status:'deadline',looks:[],providerCalls:0};
          nextPhase('ai_vision_feedback_revision');
          return composeCatalogSkuLooks({
            pool,context,directions,critiqueFeedback,layerRequired,allowLayer,feedbackTaste,
            // Isolated 72s staging: use a SECOND model family for creative
            // response to genuine photo rejections; no S5 or preview change.
            crossLabRevision:true,
            fastPhotoRevision:budgetMs>=70000&&
              env.STYLIST6_FAST_PHOTO_REVISION_ENABLED==='true',
            apiKey:nvidiaKey,...(fetchImpl?{fetchImpl}:{}),
            signal:deadlineController.signal,
            timeoutMs:budgetMs>=70000&&
              env.STYLIST6_FAST_PHOTO_REVISION_ENABLED==='true'
              ?Math.min(8000,Math.max(550,left-15000))
              :Math.min(4900,Math.max(550,left-13900)),
          });
        }:null;
    const selectPhotoApprovedPortfolio=modelSlateEnabled&&budgetMs>=70000
      ? async({looks,critiques,portfolios,context,creativePlans})=>{
          const left=remainingMs();
          if(left<6500)return {status:'deadline',tokens:[],providerCalls:0};
          nextPhase('final_photo_portfolio_editor');
          return selectPhotoEditorialDeck({
            looks,critiques,portfolios,context,creativePlans,
            apiKey:nvidiaKey,
            // The optional visual jury sees ONLY source photos independently
            // authorized and photo-judged earlier in this same request.
            // A slow/unavailable visual model retains the original text editor.
            resolvePhoto:photoPortfolioResolver,
            authorizePhoto:isInferenceAuthorized,
            // Different-lab independent VISUAL fashion curator. Bounded
            // parallel trial on isolated >=72s staging only; never S5.
            independentVisionJury:budgetMs>=70000,
            // Only TWO conflicting, already source- and PHOTO-verified
            // collection votes trigger a bounded third creative arbiter.
            // No prod S5, no new garment/photo inference, no third-card fill.
            crossLabTieBreak:budgetMs>=70000,
            ...(fetchImpl?{fetchImpl}:{}),
            signal:deadlineController.signal,
            timeoutMs:Math.min(9600,Math.max(500,left-3300)),
          });
        } : null;
    const computedDeck = await runWithinRemainingDeadline(() => recommendFashionDeck({
      request: input,
      catalog: feasible,
      referenceBriefs: refs,
      aiCreativeBriefs: creativePlan.briefs,
      photoFirstPriorityIds:photoFirstCatalogReady
        ?new Set(feasible.filter(item=>item.isShopItem===true&&
          verifiedPhotoRows.has(String(item.id))).map(item=>String(item.id))):null,
      ...(selectDirectorFinalists?{selectDirectorFinalists}:{}),
      ...(selectCatalogCompositions?{selectCatalogCompositions}:{}),
      ...(selectCatalogRescueCompositions?{selectCatalogRescueCompositions}:{}),
      hardGate: async (items) => devLookFeasibility(items),
      priceToBudgetCurrency,
      feedbackTaste,
      valueFxRates: sourceMode === 'catalog' && !budget ? fxSnapshot?.uzsPerUnit : null,
      // Staging-only visual-first quality pass: review every admissible card
      // in one bounded batch, not just the first selection.
      visualDeckReview: true,
      // User-facing quality requires a three-look deck even when the stricter
      // style/occasion/formality floors reject a candidate. Six candidates
      // yielded only two verified looks in real cloud QA; make all eight
      // independently model-reviewed rescue options available in one wave.
      // Still bounded by 38s/50s, 8 paid critiques, with NO unreviewed fill.
      visualCandidateLimit: 8,
      // QUALITY FIRST: never pad two genuinely NVIDIA-approved looks with an
      // unreviewed third. An honest partial recommendation beats a pretty
      // three-card count with unverified fashion quality.
      allowUnreviewedVisualFill: false,
      // Only the isolated 72s authenticated Render staging has enough time
      // for a small SECOND real-photo review wave after an initial fashion
      // rejection. Never widen paid inference or deadlines in Vercel/S5.
      photoRescueRemainingMs: budgetMs >= 70000 ? remainingMs : null,
      ...(selectPhotoApprovedPortfolio?{selectPhotoApprovedPortfolio}:{}),
      ...(visualCritic ? { visualCritic, canJudgeOutfit, canDisplayOutfit } : {}),
    }));
    // Enforce the complete four-SKU contract at the FINAL cloud boundary.
    // No client-side card can silently lose the outerwear and claim an outfit.
    const deck=budgetMs>=70000
      ?enforceCatalogFourItemDeck(computedDeck,sourceMode)
      :computedDeck;
    nextPhase('result_mapping');
    const visualStatus = mapCloudVisualStatus(deck);
    if ((deck.status !== 'complete' && deck.status !== 'partial') || !deck.outfits.length) {
      // Distinguish a real feasibility shortage from an actual visual
      // rejection. A valid AI verdict that approves zero outfits must not
      // silently reopen the originally feasible (but rejected) catalog deck.
      const qualityFailed = (deck.diagnostics?.visualReviewCoverage?.valid ?? 0) > 0 &&
        (deck.diagnostics?.visualReviewCoverage?.approved ?? 0) === 0;
      // A present but incompatible locked item cannot produce an outfit.
      // This is actionable (422), never a fabricated replacement for the
      // anchor. Settlement atomically refunds the previously charged quota.
      const anchorUnavailable = requiredIds.length > 0 && !qualityFailed &&
        (deck.diagnostics?.candidateCount ?? 0) === 0;
      const failureCode = anchorUnavailable ? 'anchor_outfit_unavailable'
        : qualityFailed ? 'stylist6_visual_quality_unmet' : 'stylist6_no_feasible_looks';
      const failureStatus = anchorUnavailable ? 422 : 502;
      // In an ACTUAL authenticated iPhone Work generation, all 120B
      // photographic quality decisions failed; previous logging occurred
      // only AFTER the success branch, making the rejection mix invisible.
      // Keep only the same bounded, non-identity enum/counter trace as
      // successful S6 requests. Never log garment IDs, prices, user/account
      // IDs, model prompts, JPEG bytes or merchant URLs.
      if(modelSlateEnabled&&visualAsked) {
        console.info('stylist6_safe_visual_quality', {
          ...stylist6SafeQualityTrace(deck.diagnostics,0,Date.now()-started),
          failedReason:['stylist6_visual_quality_unmet','stylist6_no_feasible_looks',
            'anchor_outfit_unavailable'].includes(failureCode)
            ?failureCode:'other',
          creativePlanCount:Math.min(3,creativePlan.briefs.length),
        creativePlanDetailedArtDirections:creativePlan.briefs.filter(plan=>
          typeof plan.artDirection==='string'&&plan.artDirection.length>=85)
          .length,
        creativePlanBackup:['used_independent_source_grounded',
          'supplemented_independent_source_grounded','not_improved','invalid_plan',
          'provider_unavailable','timeout','load_shed','not_configured']
          .includes(creativePlan.invalidPlanBackupStatus)
          ?creativePlan.invalidPlanBackupStatus:'not_needed',
          stableSourceCohort,
          creativePlanVisualExamples:Math.max(0,Math.min(64,
            Number(creativePlan.visualAtlasExamples)||0)),
          creativePlanStatus:['planned','planned_partial','cached','invalid_plan',
            'disabled_or_deadline','timeout','provider_unavailable','not_configured']
            .includes(creativePlan.status)?creativePlan.status:'other',
          phases:{...phaseAt,[activePhase]:Date.now()-phaseStart},
        });
      }
      await finish(failureStatus, failureCode);
      return fail(failureCode, failureStatus, {
        stage: 'stylist6_ranking',
        engineVersion: STYLIST6_CLOUD_VERSION,
        visualStatus,
        serverInventory: true,
        clientInventoryIgnored,
        replayProtection: 'durable',
        diagnostics: deck.diagnostics || null,
      });
    }
    const excluded = new Set(resolvedExcludedIds);
    const previousDeckOverlapBySlot = { top: 0, outerwear: 0, bottom: 0, shoes: 0 };
    for (const look of deck.outfits) {
      for (const item of look.items) {
        if (excluded.has(String(item.id)) && previousDeckOverlapBySlot[item.macroCategory] !== undefined) {
          previousDeckOverlapBySlot[item.macroCategory] += 1;
        }
      }
    }
    let liveOfferMap = null;
    if (typeof io.fetchMerchantOffers === 'function') {
      try {
        const ids = [...new Set(deck.outfits.flatMap((look) => look.items.map((item) => String(item.id))))];
        const fetched = await io.fetchMerchantOffers(ids);
        liveOfferMap = fetched instanceof Map ? fetched : new Map(Object.entries(fetched || {}));
      } catch {
        liveOfferMap = null;
      }
    }
    const mappedOutfits = deck.outfits.map((look, index) => {
      const mappedItems = look.items.map((item) => mapDeckItem(
        item,
        look.fashion?.score,
        verifyMerchantOffer(item, { offerRegistry: liveOfferMap, targetMarket: 'UZ' }),
      ));
      const lookVerified = Array.isArray(look.visualEvidence);
      return {
        // Same positional id convention as the S5 edge (`outfit_${index+1}`):
        // the stylist_look_feedback constraint and the app's feedback guard
        // require it, so a thumbs write can attribute this exact deck.
        id: `outfit_${index + 1}`,
        selectionSource: lookVerified ? 'ai' : 'deterministic_fallback',
        items: mappedItems,
        offerVerification: summarizeDeckOffers([{ items: mappedItems }]),
        description: look.description,
        concept: { id: look.concept?.id, title: look.concept?.title, story: '', score: look.fashion?.score },
        // Actual evidence from the selected SKU deck, never a cosmetic
        // variant label or a promise that the catalog has every direction.
        direction: look.direction ?? null,
        stylingTips: [],
        confidence: look.fashion?.score,
        matchScore: look.fashion?.score,
        engineVersion: STYLIST6_CLOUD_VERSION,
        fashionDiagnostics: {
          score: look.fashion?.score,
          referenceMetadata: look.fashion?.referenceMetadata || null,
          // Verdict of THIS displayed look, not a global batch verdict
          // that may refer to a rejected or non-displayed challenger.
          referenceEvaluationMode: lookVerified ? look.visualReferenceMode || 'unknown' : 'not_judged',
          visualVerdict: lookVerified
            ? 'JUDGED' : deck.diagnostics?.visualVerdict === 'JUDGED'
              ? 'NOT_JUDGED' : deck.diagnostics?.visualVerdict || 'NOT_RUN',
          imageSupportedItems: look.fashion?.imageSupportedItems ?? null,
          // Only catalog-grounded positive matches; never expose the full
          // preference profile or assert body/fit facts without evidence.
          personalization: look.fashion?.personalization || null,
        },
      };
    });
    const offerVerification = summarizeDeckOffers(mappedOutfits);
    const idempotencyRecorded = await finish(200, deck.status);
    // The request-scoped data is sensitive; expose only bounded counters and
    // enumerated reasons in Render server logs. This is not a fashion verdict
    // and never changes the number of shown outfits.
    if(modelSlateEnabled && visualAsked) {
      const safeTrace=stylist6SafeQualityTrace(
        deck.diagnostics,deck.outfits.length,Date.now()-started);
      console.info('stylist6_safe_visual_quality', {
        ...safeTrace,
        creativePlanCount:Math.min(3,creativePlan.briefs.length),
        creativePlanDetailedArtDirections:creativePlan.briefs.filter(plan=>
          typeof plan.artDirection==='string'&&plan.artDirection.length>=85)
          .length,
        creativePlanBackup:['used_independent_source_grounded',
          'supplemented_independent_source_grounded','not_improved','invalid_plan',
          'provider_unavailable','timeout','load_shed','not_configured']
          .includes(creativePlan.invalidPlanBackupStatus)
          ?creativePlan.invalidPlanBackupStatus:'not_needed',
        stableSourceCohort,
        creativePlanVisualExamples:Math.max(0,Math.min(64,
          Number(creativePlan.visualAtlasExamples)||0)),
        creativePlanStatus:['planned','planned_partial','cached','invalid_plan',
          'disabled_or_deadline','timeout','provider_unavailable','not_configured']
          .includes(creativePlan.status)?creativePlan.status:'other',
        phases:{...phaseAt,
          [activePhase]:Date.now()-phaseStart},
      });
    }
    const completedBody={
        success: true,
        generationVersion: STYLIST6_CLOUD_VERSION,
        source: 'stylist-6-cloud-staging',
        // Logical request identity echoed back so the app can attribute
        // feedback (like/dislike) to this exact deck, as the S5 edge does.
        generationRequestId: requestId,
        compositionProtocol: 'optional-layer-v1',
        deckTarget: 3,
        status: deck.status,
        sourceMode,
        serverInventory: true,
        clientInventoryIgnored,
        language,
        currencyConversion,
        offerVerification,
        replayProtection: 'durable',
        idempotencyRecorded,
        previousDeckOverlapBySlot,
        visualHydration,
        visualStatus,
        visualReceipt: {
          // Cost metering is tokens-based only: no per-call pricing table
          // exists in this codebase, so no currency estimate is invented.
          model: visualCritic ? NVIDIA_VISION_MODEL : null,
          policy: visualPolicy,
          requested: visualAsked,
          serverDisabled: visualAsked && visualDisabled,
          // Owner reference storage is optional; without it the critic may
          // still judge actual product images, but never score "reference
          // fidelity". This metadata is non-sensitive and explicitly names
          // the evidence mode of the attempt.
          referenceBucketConfigured: Boolean(String(env.STYLIST6_REFERENCE_BUCKET || '').trim()),
          providerCalls: visualEvents.filter((event) => event.dispatched).length,
          imagesSent: visualEvents.reduce((sum, event) =>
            sum + (event.dispatched && Number.isSafeInteger(event.images) ? event.images : 0), 0),
          cacheHits: visualEvents.filter((event) => event.cacheHit === true).length,
          retriedCalls: visualEvents.filter((event) => event.retried === true).length,
          validVerdicts: visualEvents.filter((event) => event.outcome === 'judged').length,
          abstentions: visualEvents.filter((event) => event.outcome === 'abstain').length,
          displayedLooks: deck.outfits.length,
          displayedJudged: deck.outfits.filter((look) => !!look.visualEvidence).length,
          // Explicitly separate a reviewed deck from a successful request,
          // and full 3-card proof from merely having at least one verdict.
          visualApproval: !visualAsked ? 'not_requested'
            : visualDisabled ? 'disabled'
              : !visualCritic ? 'unavailable'
            : deck.outfits.length === 3 &&
                deck.outfits.every((look) => Array.isArray(look.visualEvidence))
              ? 'full_three_verified'
              : deck.outfits.length > 0 &&
                  deck.outfits.every((look) => Array.isArray(look.visualEvidence))
                ? 'verified_partial'
                : 'unverified_fallback',
          totalTokens: visualEvents.reduce((sum, event) =>
            sum + (Number.isFinite(event.usage?.total_tokens) ? event.usage.total_tokens : 0), 0) || null,
          events: visualEvents,
        },
        diagnostics: deck.diagnostics || null,
        // Gate D read receipt: the bounded feedback read this ranking used
        // (or honestly did not use). Never a taste profile, only a status.
        creativeDirector: {
          version: CREATIVE_DIRECTOR_VERSION,
          status: creativePlan.status,
          providerCalls: creativePlan.providerCalls,
          briefsAccepted: creativePlan.briefs.length,
        },
        feedbackTaste: {
          status: feedbackTasteStatus,
          rows: feedbackRows.length,
          active: feedbackTaste.active === true,
          // Distinguishes an explicit single thumbs example from a
          // statistically learned, held-out-validated whole-look preference.
          // Status ONLY; no owner's garment details or training weights.
          learnedStatus: feedbackTaste.learnedPreference?.status || 'untrained',
          version: FEEDBACK_TASTE_VERSION,
        },
        outfits: mappedOutfits,
      };
    // A failed recovery-store write NEVER invalidates an already paid,
    // independently PHOTO approved deck. Save once, before the HTTP response,
    // for the same request owner; no use of client-supplied inventory.
    let recoveryStored=false;
    try{
      if(typeof io.saveCompletedResult==='function')
        recoveryStored=await io.saveCompletedResult(
          userId,requestId,completedBody)===true;
    }catch{/* response remains authoritative; no second charge/retry */}
    if(modelSlateEnabled && visualAsked)console.info(
      'stylist6_result_delivery',{
        outcome:'completed_paid_deck',
        recoverable:recoveryStored,
        displayed:Math.min(3,mappedOutfits.length),
        wallMs:Math.min(180000,Math.max(0,Date.now()-started)),
      });
    return {status:200,body:completedBody};
  } catch (error) {
    const timedOut = error?.message === 'stylist6_timeout';
    const status = timedOut ? 504 : 500;
    const failureCode = timedOut ? 'stylist6_timeout' : 'stylist6_ranking_failed';
    // Internal failure evidence, never sent to the app. Only a server source
    // location, standard JS error class and safe system-code enum are logged:
    // no private JWT, garment/personal payload, URL, key or raw error message.
    const firstServerFrame = String(error?.stack || '').match(
      /backend\/(?:stylist6|stylist-node)\/[a-z0-9_./-]+\.(?:mjs|ts):\d+:\d+/i,
    )?.[0] || null;
    const errorClass = ['Error', 'TypeError', 'RangeError', 'SyntaxError', 'ReferenceError']
      .includes(error?.name) ? error.name : 'UnknownError';
    const systemCode = /^(?:E(?:CONNRESET|TIMEDOUT|HOSTUNREACH|PIPE|AI_AGAIN)|ERR_[A-Z_]{1,40})$/
      .test(String(error?.code || '')) ? error.code : null;
    console.error('stylist6_cloud_failure', {
      failureCode, errorClass, systemCode, firstServerFrame, claimed, debited,
      activePhase,
      safeWallMs:Math.min(180000,Math.max(0,Date.now()-started)),
      phaseTimings:{...phaseAt,[activePhase]:Date.now()-phaseStart},
    });
    if (claimed) {
      await finish(debited ? status : 503, debited ? failureCode : 'aborted_pre_debit').catch(() => {});
    }
    return fail(failureCode, status, { stage: 'stylist6_ranking', retryWithNewRequestId: debited && timedOut });
  }
}
