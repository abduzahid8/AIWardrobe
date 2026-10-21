/** One real disposable wardrobe; mixed owned/catalog generation through the deployed handler. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
import {loadBudgetHelpers} from './load-outfit-budget-helpers.mjs';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;
const reportName=process.env.SUPPLEMENT_AUDIT_REPORT||'production-generation-supplement.json';
if(!/^[a-z0-9-]+\.json$/.test(reportName))throw Error('Invalid report name');
const avoidColor=process.env.SUPPLEMENT_AUDIT_AVOID_COLOR;
if(avoidColor&&!['black','blue','navy','red','green'].includes(avoidColor))throw Error('Unsupported audit color');
const auditBudgetUsd=Number(process.env.SUPPLEMENT_AUDIT_BUDGET_USD||300);
if(!Number.isInteger(auditBudgetUsd)||auditBudgetUsd<1||auditBudgetUsd>1000)throw Error('Invalid audit budget');
if(!base||!anon||!key)throw Error('Fixture configuration required');
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(120000)})}};
const admin=createClient(base,key,options),client=createClient(base,anon,options);
const report={measuredAt:new Date().toISOString(),scope:`One disposable production owner with18real fixture garments: six valid tops, six valid shoes, six formal-ineligible bottoms. Real Auth/catalog/NVIDIA; one USD${auditBudgetUsd} outfit request. No native image/UI or capacity claim.`,checks:[]};
report.preferences=avoidColor?{avoidColors:[avoidColor]}:{};
let owner;
const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('Fixture assertion failed');};
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(created.error||!created.data.user)throw Error('Fixture creation failed');owner=created.data.user.id;report.fixtureUserId=owner;
 const login=await client.auth.signInWithPassword({email,password});if(login.error)throw Error('Fixture login failed');
 if((await client.from('profiles').update({trial_started_at:new Date().toISOString()}).eq('id',owner)).error)throw Error('Fixture trial start failed');
 const rows=['top','bottom','shoes'].flatMap(category=>Array.from({length:6},()=>({id:randomUUID(),user_id:owner,category,name:category==='top'?'White long sleeve cotton shirt':category==='bottom'?'100% LINEN RELAXED FIT TROUSERS':'Brown leather penny loafers',image_url:'https://example.invalid/readiness-supplement.jpg'})));
 if((await client.from('clothing_items').insert(rows)).error)throw Error('Fixture wardrobe setup failed');
 const {core}=await loadBudgetHelpers();const rates=await client.functions.invoke('budget-rates',{body:{}});
 if(rates.error)throw Error('Fixture rates failed');
 const snapshot=core.validateBudgetFxSnapshot(rates.data.snapshot),fx=core.budgetFxContext(snapshot);
 const start=performance.now();
 const response=await fetch(new URL('/functions/v1/generate-outfits',base),{method:'POST',headers:{apikey:anon,Authorization:`Bearer ${login.data.session.access_token}`,'Content-Type':'application/json'},body:JSON.stringify({stylePreferences:'semi_classic',occasion:'formal',prompt:`Create one complete formal outfit. Budget under $${auditBudgetUsd}.`,weather:{temp:20,condition:'Clear'},limit:1,preferences:{maxShopBudget:{amount:auditBudgetUsd,currency:'USD'},budgetRatesId:snapshot.id,...report.preferences}}),signal:AbortSignal.timeout(120000)});
 const data=await response.json();report.response={status:response.status,latencyMs:Math.round(performance.now()-start),error:data.error??null,code:data.code??null,provider:data.provider??null,providerAttempts:data.providerAttempts??[]};
 check('one_real_generated_outfit_returned',response.ok&&data.outfits?.length===1);
 const outfit=data.outfits[0],judge={},structure={};
 for(const [name,target] of [['occasionRules',judge],['outfitStructure',structure]])new Function('exports',ts.transpileModule(await readFile(`backend/supabase/functions/_shared/${name}.ts`,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(target);
 check('complete_formal_outfit_passes_shared_rules',!structure.outfitStructureReason(outfit.items,!!data.layered)&&!judge.styleJudgeReason(outfit.items,'semi_classic','formal',{temp:20}));
 const owned=new Map(rows.map(row=>[row.id,row])),invalid=new Set(rows.filter(row=>row.category==='bottom').map(row=>row.id));
 check('ineligible_owned_bottoms_are_not_returned',outfit.items.every(item=>!invalid.has(item.id)));
 const shop=outfit.items.filter(item=>String(item.id).startsWith('shop_'));
 check('shop_supplement_supplies_the_missing_bottom',shop.some(item=>item.macroCategory==='bottom')&&outfit.items.some(item=>owned.has(item.id)));
 const catalog=await client.from('shop_catalog').select('id,name,brand,price,currency,source_url,is_active').in('id',shop.map(item=>item.id.slice(5)));
 if(catalog.error)throw Error('Fixture catalog verification failed');const byId=new Map(catalog.data.map(row=>[row.id,row]));let spend=0,exact=true;
 for(const item of outfit.items){
  if(owned.has(item.id)){exact=exact&&item.name===owned.get(item.id).name;continue;}
  const row=byId.get(String(item.id).slice(5));
  if(!row||!row.is_active||row.name!==item.name||Number(row.price)!==Number(item.price)||row.currency!==item.currency||row.source_url!==item.shopUrl||(row.brand||'')!==(item.brand||'')){exact=false;continue;}
  spend+=core.referenceAmount(Number(row.price),row.currency,'UZS',snapshot,fx.checkedAt);
 }
 check('returned_product_fields_match_owned_and_active_catalog_rows',exact);
 const cap=core.referenceAmount(auditBudgetUsd,'USD','UZS',snapshot,fx.checkedAt);
 check('catalog_spend_respects_exact_request_budget',core.fitsBudgetAmount(spend,cap)&&data.budgetReference?.id===snapshot.id);
 if(avoidColor)check('requested_shop_color_is_absent',shop.every(item=>!String(item.color||'').toLowerCase().includes(avoidColor)&&!new RegExp('\\b'+avoidColor+'\\b','i').test(item.name||'')));
 report.budget={spendUzs:spend,capUzs:cap,reference:snapshot.id};report.items=outfit.items.map(item=>({id:item.id,name:item.name,slot:item.macroCategory,source:owned.has(item.id)?'owned':'catalog',price:item.price??null,currency:item.currency??null}));
}catch(error){report.failure=/^Fixture .* failed$/.test(error.message)?error.message:error.name;process.exitCode=1;}
finally{
 report.cleanup={};try{report.cleanup.refreshSessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;}catch{report.cleanup.refreshSessionsRevoked=false;}
 if(owner)try{report.cleanup.userDeleted=!(await admin.auth.admin.deleteUser(owner)).error;}catch{report.cleanup.userDeleted=false;}
 report.passed=report.checks.length===(avoidColor?7:6)&&report.checks.every(c=>c.passed)&&report.cleanup.refreshSessionsRevoked&&report.cleanup.userDeleted&&!report.failure;
 if(!report.passed)process.exitCode=1;
 await writeFile('docs/reports/readiness-evidence/'+reportName,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
