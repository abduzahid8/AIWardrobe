/** Focused critical checks using a bounded real-catalog read. No providers.
 * One planner pass per required acceptance context; not an exhaustive replay. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import esbuild from 'esbuild';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url),root=new URL('../../',import.meta.url).pathname;
const source=fs.readFileSync(root+'backend/supabase/functions/generate-outfits/index.ts','utf8')
 .replace(/^import \{ serve \} from .*$/m,'const serve=()=>{};')
 .replace(/^import \{ createClient \} from .*$/m,'const createClient=()=>{throw Error("network forbidden")};')
 + '\nexport {createOutfitPlan,stylingContext,prepareGarment,assessOutfitQuality,styleJudgeReason,fourSlotOutfitReason,validateOutfitAgainstStyle,validateOutfitCompatibility,garmentContextReason,visualAdmissionReason,detectSlotMismatch,isFourSlotCandidate,isNonOutfitItem,rejectForStyleOccasion,rejectForClimate,rejectForRequestIntent,rejectForActivity};';
const b=await esbuild.build({stdin:{loader:'ts',contents:source,resolveDir:root+'backend/supabase/functions/generate-outfits',sourcefile:'critical.ts'},bundle:true,format:'cjs',platform:'node',target:'node22',write:false});
const module={exports:{}};new Function('module','exports','require','Deno',b.outputFiles[0].text)(module,module.exports,name=>name.startsWith('https:')?{createClient:()=>{throw Error('network forbidden')}}:require(name),{env:{get:()=>undefined}});
const api=module.exports;
const rows=JSON.parse(fs.readFileSync(process.argv[2]||'/tmp/stylist4-catalog-before.json'));
const items=rows.map(r=>api.prepareGarment({...r,id:'shop_'+r.id,isShopItem:true,isWardrobe:false,macroCategory:r.fashion_slot,type:r.garment_type,
 imageUrl:r.image_url,color:r.fashion_intelligence?.descriptor?.color,visualFashion:r.visual_fashion||undefined,visualSourceHash:r.visual_source_hash||undefined}));
const results=[];
for(const [style,occasion] of [['classic','Work'],['old_money','Work'],['semi_classic','Date night'],['casual','Travel']]){
 const context=api.stylingContext({style,occasion,sourceMode:'catalog'});
 const eligible=items.filter(i=>!api.visualAdmissionReason(i)&&!api.garmentContextReason(i,context)&&api.isFourSlotCandidate(i)&&!api.isNonOutfitItem(i)&&!api.rejectForStyleOccasion(i,style,occasion)&&!api.rejectForClimate(i,null)&&!api.rejectForRequestIntent(i,occasion,null)&&!api.rejectForActivity(i,occasion));
 const map=new Map(eligible.map(i=>[i.id,i]));let assessed=0;
 const evaluate=parts=>{
  assessed++;if(api.fourSlotOutfitReason(parts,true)||parts.some(i=>i.macroCategory==='shoes'&&i.type!=='shoes'))return {valid:false,score:0};
  if(!api.validateOutfitAgainstStyle({items:parts},style,map,true,{}).ok||!api.validateOutfitCompatibility({items:parts},map,style).ok)return {valid:false,score:0};
  const a=api.assessOutfitQuality(parts,style,occasion,null,null,{},context);
  return {valid:a.acceptance.passed&&!api.styleJudgeReason(parts,style,occasion,null,null,{},a,context),score:a.score};
 };
 const cpu=process.cpuUsage(),start=performance.now();const plan=api.createOutfitPlan(eligible,context,evaluate,{requiredIds:[],target:3,priority:()=>[],cost:i=>i.price,maxEvaluations:256,maxSyncMs:600});
 const elapsed=performance.now()-start,usage=process.cpuUsage(cpu),cpuMs=(usage.user+usage.system)/1000;
 assert(assessed<=256);assert(cpuMs<1200,'planner CPU risk');
 assert(plan.seed.every(look=>look.items.length===4));
 results.push({style,occasion,wallMs:Math.round(elapsed),cpuMs:Math.round(cpuMs),...plan.diagnostics,concepts:plan.seed.map(look=>({id:look.concept.id,fit:look.conceptScore})),completeLooks:plan.seed.length});
}
const checks=[['top','The Slim-Fit Japanese Oxford | Slate Grey | Regular',null],['top','Sandalwood Crewneck Sweater Cash Linen Gg 7',null],['shoes','Oxford Leather Dress Shoes',null],['top','Oxford Leather Dress Shoes','shoes'],['bottom','Barnard Short-Sleeved Polo Shirt','top']];
for(const [slot,name,want] of checks)assert.equal(api.detectSlotMismatch(slot,name),want);
const report={measuredAt:new Date().toISOString(),scope:'local Node kernel, bounded real catalog, weather unknown, no production CPU or visual QA claim',classificationChecks:5,providerCalls:0,results};
fs.writeFileSync(root+'docs/reports/stylist-4-20261107/critical-catalog.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
