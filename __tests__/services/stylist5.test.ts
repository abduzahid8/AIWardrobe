/** @jest-environment node */
import {createOutfitPlan} from '../../src/domain/validatedQuartets';
import {stylingContext} from '../../src/domain/stylingContext';
import {describeGarment,prepareGarment} from '../../src/domain/garmentDescriptor';
import {valueIntelligence} from '../../src/domain/editorialIntelligence';
import {fourSlotOutfitReason} from '../../src/domain/outfitStructure';
import {rejectForStyleOccasion} from '../../src/domain/occasionRules';
import {boundedGenerationHistory,validateOutfitGenerationRequest,OutfitRequestError} from '../../src/domain/outfitGenerationRequest';
import {savedQuartetReason} from '../../src/services/savedOutfitContract';
import {stylistCalendarItems} from '../../features/calendar/outfitSlots';

const names=[['White cotton shirt','Blue Oxford dress shirt','Navy knit polo shirt'],['Navy wool blazer','Cream button cardigan','Grey wool coat'],['Navy tailored trousers','Beige cotton chinos','Black straight trousers'],['Brown leather loafers','Black Oxford dress shoes','Brown Derby shoes','White minimal leather sneakers']];
const inventory=names.flatMap((slot,s)=>slot.flatMap((name,n)=>Array.from({length:3},(_,i)=>({id:`${s}-${n}-${i}`,name:`${name} option ${i}`,macroCategory:['top','outerwear','bottom','shoes'][s],isShopItem:true,price:100+n,currency:'USD'}))));
const options={requiredIds:[] as string[],target:3,priority:()=>[] as number[],cost:(i:{price:number})=>i.price,maxEvaluations:96,maxSyncMs:2000};
const ctx=()=>stylingContext({style:'old_money',occasion:'work',sourceMode:'catalog'});

it('builds briefs first, honors physical anchors and budget, and bounds full evaluations',()=>{
 let calls=0;const anchor='0-0-0';
 const p=createOutfitPlan(inventory,ctx(),parts=>{calls++;return {valid:!fourSlotOutfitReason(parts),score:0.9}},{...options,requiredIds:[anchor],budget:410});
 expect(p.seed.length).toBeGreaterThan(0);expect(calls).toBeLessThanOrEqual(96);
 for(const look of p.seed){
  expect(look.items.some(i=>i.id===anchor)).toBe(true);
  expect(look.items.reduce((n,i)=>n+i.price,0)).toBeLessThanOrEqual(410);
  for(const item of look.items){const slot=['top','outerwear','bottom','shoes'].indexOf(item.macroCategory);expect(look.concept.slots[slot]).toContain(describeGarment(item).subtype);}
 }
});
it('never lets arbitrarily large history or provider rank displace a substantially stronger look',()=>{
 const p=createOutfitPlan(inventory,ctx(),parts=>({valid:true,score:parts.some(i=>i.id==='0-0-0')?0.99:0.65}),{
  ...options,target:1,priority:parts=>[parts.some(i=>i.id==='0-0-0')?-1e12:1e12]});
 const ranks=new Map(p.candidates.slice().reverse().map((l,i)=>[l.id,i]));
 expect(p.select(ranks)[0].items.some(i=>i.id==='0-0-0')).toBe(true);
});
it('returns fewer honest identities instead of three renamed copies of one silhouette',()=>{
 const single=inventory.filter(i=>i.id.split('-')[1]==='0');
 const p=createOutfitPlan(single,ctx(),()=>({valid:true,score:0.9}),options);
 expect(p.seed).toHaveLength(1);
});
it('keeps missing mandatory shoes, source mode and aliases authoritative',()=>{
 expect(createOutfitPlan(inventory.filter(i=>i.macroCategory!=='shoes'),ctx(),()=>({valid:true,score:1}),options).seed).toEqual([]);
 expect(createOutfitPlan(inventory,stylingContext({sourceMode:'wardrobe'}),()=>({valid:true,score:1}),options).seed).toEqual([]);
 const aliases=inventory.map(i=>({...i,imageUrl:'https://example.test/same.jpg'}));
 expect(createOutfitPlan(aliases,ctx(),()=>({valid:true,score:1}),options).seed).toEqual([]);
});
it('uses actual comparable native prices with an explicit luxury override and no budget inference',()=>{
 const peers=Array.from({length:8},(_,i)=>({id:String(i),name:'Wool blazer',macroCategory:'outerwear',isShopItem:true,currency:'UZS',price:1_000_000+i*100_000}));
 const luxury={...peers[0],id:'luxury',price:53_319_510};
 const normal=valueIntelligence([...peers,luxury],ctx());
 expect(normal(luxury).score).toBeLessThan(normal(peers[1]).score);expect(ctx().budget).toBeUndefined();
 const explicit=valueIntelligence([...peers,luxury],{...ctx(),request:'I want luxury designer pieces'});
 expect(explicit(luxury).score).toBe(0.5);
 expect(valueIntelligence([{...luxury,currency:'USD'}],ctx())({...luxury,currency:'USD'}).ratio).toBeNull();
});
it('reconciles smart work cardigan admission without waiving formal evening dress',()=>{
 const knit={id:'knit',name:'Fine gauge button cardigan',macroCategory:'outerwear'};
 expect(rejectForStyleOccasion(knit,'old_money','Work')).toBeNull();
 expect(rejectForStyleOccasion(knit,'old_money','Formal')).toBeTruthy();
});
it('negotiates truthful three-piece looks through structure, save and calendar boundaries',()=>{
 const parts=inventory.filter(i=>['0-0-0','2-0-0','3-0-0'].includes(i.id));
 expect(fourSlotOutfitReason(parts)).toBe('missing_layer');
 expect(fourSlotOutfitReason(parts,false)).toBeNull();
 expect(savedQuartetReason(parts)).toBeTruthy();expect(savedQuartetReason(parts,'optional-layer-v1')).toBeNull();
 expect(stylistCalendarItems(parts,'optional-layer-v1')).toHaveLength(3);
 const p=createOutfitPlan(inventory,stylingContext({style:'casual',occasion:'Travel',optionalLayer:true}),()=>({valid:true,score:0.9}),options);
 expect(p.alternatives.some(l=>l.items.length===3)).toBe(true);
});
it('bounds only optional persisted history and identifies rejected fields without payload values',()=>{
 const history=boundedGenerationHistory({previousDeckProductKeys:Array.from({length:90},(_,i)=>'id:p'+i),olderRecentProductKeys:Array.from({length:600},(_,i)=>'id:o'+i),rotationDecks:[{requestId:'one',productKeys:Array(70).fill('id:valid')}]});
 expect(history.previousDeckProductKeys).toHaveLength(60);expect(history.excludeProductKeys.length).toBeLessThanOrEqual(450);
 const request={stylePreferences:'Old Money',occasion:'Work',prompt:'actual intent',selectedItemIds:['locked'],...history};
 expect(validateOutfitGenerationRequest(request)).toBe(request);
 try{validateOutfitGenerationRequest({...request,recentLookSignatures:Array(37).fill('secret-value')});throw Error('accepted');}
 catch(e){expect(e).toBeInstanceOf(OutfitRequestError);expect(e).toMatchObject({field:'recentLookSignatures',reason:'too_many'});expect(String(e)).not.toContain('secret-value');}
 expect(()=>validateOutfitGenerationRequest({...request,prompt:'x'.repeat(1001)})).toThrow();
});

it('refreshes positive construction facts over cached text without treating a button-down shirt as insulation',()=>{
 const cached=describeGarment({name:'Everyday jacket',macroCategory:'outerwear'});
 const updated=prepareGarment({name:'Flannel down mid jacket',macroCategory:'outerwear'},cached);
 expect(describeGarment(updated).insulated).toBe(true);
 expect(describeGarment(prepareGarment({name:'Cotton button-down shirt',macroCategory:'top'})).insulated).toBe(false);
});
