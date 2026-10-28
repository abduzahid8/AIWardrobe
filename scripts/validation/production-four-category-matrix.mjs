/** Real authenticated production requests. No model, database or network mocks. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
import {createClient} from '@supabase/supabase-js';
import ts from 'typescript';
const url=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if(!url||!anon||!process.env.SUPABASE_SERVICE_KEY)throw Error('Audit credentials unavailable');
const options={auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(65000)})}};
const admin=createClient(url,process.env.SUPABASE_SERVICE_KEY,options),client=createClient(url,anon,options);
const modules={};for(const name of ['outfitStructure','occasionRules']){const target={};Function('exports',ts.transpileModule(await readFile(`backend/supabase/functions/_shared/${name}.ts`,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(target);modules[name]=target;}
const cases=[
 {name:'classic_work_no_weather',style:'classic',occasion:'Work'},
 {name:'old_money_date_no_budget',style:'old_money',occasion:'date'},
 {name:'minimalist_travel_no_budget',style:'minimalist',occasion:'travel'},
 {name:'casual_everyday_no_budget',style:'casual',occasion:'Everyday'},
 {name:'sport_workout_no_budget',style:'sport',occasion:'Gym workout'},
 {name:'business_work_no_budget',style:'business_casual',occasion:'Team Collaboration'},
 {name:'classic_work_10000000_uzs',style:'classic',occasion:'Work',budget:{amount:10000000,currency:'UZS'}},
 {name:'old_money_date_10000000_uzs',style:'old_money',occasion:'date',budget:{amount:10000000,currency:'UZS'}},
 {name:'minimalist_travel_10000000_uzs',style:'minimalist',occasion:'travel',budget:{amount:10000000,currency:'UZS'}},
 {name:'casual_everyday_6000000_uzs',style:'casual',occasion:'Everyday',budget:{amount:6000000,currency:'UZS'}},
 {name:'sport_workout_6000000_uzs',style:'sport',occasion:'Gym workout',budget:{amount:6000000,currency:'UZS'}},
 {name:'honest_rejection_1_uzs',style:'classic',occasion:'Work',budget:{amount:1,currency:'UZS'},expectedReject:true},
 {name:'classic_work_250_usd',style:'classic',occasion:'Work',budget:{amount:250,currency:'USD'}},
];
const selectedCases=process.env.MATRIX_UNBUDGETED_ONLY==='1'?cases.filter(c=>!c.budget):cases;
const reportName=process.env.MATRIX_REPORT || 'oct26-real-four-category-matrix.json';
if(!/^[a-z0-9-]+\.json$/.test(reportName))throw Error('Invalid report filename');
const report={measuredAt:new Date().toISOString(),scope:'Sequential real production requests with a disposable trial owner; server discovers live catalog products. No native UI, retailer stock/price refresh or load certification.',cases:[]};
let owner,session;
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(created.error||!created.data.user)throw Error('Fixture provisioning failed');owner=created.data.user.id;report.fixtureUserId=owner;
 const signed=await client.auth.signInWithPassword({email,password});if(signed.error||!signed.data.session)throw Error('Fixture sign-in failed');session=signed.data.session;
 const trial=await admin.from('profiles').update({trial_started_at:new Date().toISOString()}).eq('id',owner);if(trial.error)throw Error('Fixture trial initialization failed');
 for(const scenario of selectedCases){
  const started=performance.now();let result={name:scenario.name,requestedBudget:scenario.budget??null,expectedRejection:!!scenario.expectedReject};
  try{
   const response=await fetch(`${url}/functions/v1/generate-outfits`,{method:'POST',headers:{apikey:anon,Authorization:`Bearer ${session.access_token}`,'Content-Type':'application/json'},body:JSON.stringify({stylePreferences:scenario.style,occasion:scenario.occasion,prompt:scenario.occasion,preferences:scenario.budget?{maxShopBudget:scenario.budget}:{},limit:3,deckTarget:3,useProvidedWardrobeOnly:true,wardrobeOnly:false,wardrobeItems:[],deckSalt:randomUUID()}),signal:AbortSignal.timeout(65000)});
   result.status=response.status;result.latencyMs=Math.round(performance.now()-started);const data=await response.json();
   result.code=[data.code,data.error].find(x=>typeof x==='string'&&/^[a-z_]{1,80}$/.test(x))??null;
   result.source=data.source??null;result.provider=data.provider??null;result.providerAttempts=data.providerAttempts??[];result.deterministicFallback=data.deterministicFallback??null;
   const outfits=Array.isArray(data.outfits)?data.outfits:[];result.outfitCount=outfits.length;
   result.failures=[];
   if(scenario.expectedReject){if(response.status!==422||outfits.length)result.failures.push('infeasible_budget_not_rejected');}
   else{
    if(!response.ok)result.failures.push(`http_${response.status}`);
    if(outfits.length!==3)result.failures.push('three_complete_looks_not_returned');
    if(data.layered!==true)result.failures.push('required_layer_flag_missing');
   }
   const ids=[...new Set(outfits.flatMap(o=>(o.items||[]).map(i=>String(i.id).replace(/^shop_/,''))))];
   const catalog=ids.length?await admin.from('shop_catalog').select('id,name,brand,price,currency,image_url,source_url,is_active').in('id',ids):{data:[],error:null};
   if(catalog.error)throw Error('Authoritative catalog verification failed');const rows=new Map(catalog.data.map(row=>[row.id,row]));
   result.outfits=outfits.map(outfit=>{
    const structure=modules.outfitStructure.fourSlotOutfitReason(outfit.items),style=modules.occasionRules.styleJudgeReason(outfit.items,scenario.style,scenario.occasion,{});
    if(structure)result.failures.push(structure);if(style)result.failures.push(style);
    let spend=0;const listedTotals={};const items=(outfit.items||[]).map(item=>{
     const row=rows.get(String(item.id).replace(/^shop_/,''));
     if(!row||!row.is_active)result.failures.push('product_not_in_active_catalog');
     else{
      for(const [field,value]of [['name',row.name],['currency',row.currency],['brand',row.brand||''],['shopUrl',row.source_url],['imageUrl',row.image_url]])if((item[field]??'')!==(value??''))result.failures.push(`catalog_${field}_mismatch`);
      if(!(Number(row.price)>0)||Number(item.price)!==Number(row.price))result.failures.push('catalog_price_mismatch');
      listedTotals[row.currency]=(listedTotals[row.currency]||0)+Number(row.price);
      if(row.currency==='UZS')spend+=Number(row.price);else if(scenario.budget?.currency==='UZS')result.failures.push('mixed_currency_spend_requires_reference');
     }
     return {id:item.id,name:item.name,category:item.macroCategory,price:item.price,currency:item.currency};
    });
    if(scenario.budget?.currency==='UZS'&&spend>scenario.budget.amount)result.failures.push('budget_exceeded');
    return {items,listedTotals,uzsSubtotal:spend,structureReason:structure,styleReason:style};
   });
   result.failures=[...new Set(result.failures)];result.passed=result.failures.length===0;
  }catch(error){result.latencyMs=Math.round(performance.now()-started);result.passed=false;result.transportError=error.name;}
  report.cases.push(result);console.log(JSON.stringify({case:result.name,status:result.status??null,latencyMs:result.latencyMs,passed:result.passed,failures:result.failures??[result.transportError]}));
 }
}catch(error){report.failure=error.name;process.exitCode=1;}
finally{
 report.cleanup={};if(session)report.cleanup.sessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;
 if(owner&&report.cleanup.sessionsRevoked)report.cleanup.ownerDeleted=!(await admin.auth.admin.deleteUser(owner)).error;
 report.passed=report.cases.length===selectedCases.length&&report.cases.every(c=>c.passed)&&report.cleanup.sessionsRevoked&&report.cleanup.ownerDeleted&&!report.failure;
 await writeFile('docs/reports/readiness-evidence/'+reportName,JSON.stringify(report,null,2)+'\n');if(!report.passed)process.exitCode=1;
}
