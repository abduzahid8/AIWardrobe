/** Real hosted reproduction of the reported shopping request; disposable owner only. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
import {createClient} from '@supabase/supabase-js';
import ts from 'typescript';
const url=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const options={auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(120000)})}};
const admin=createClient(url,process.env.SUPABASE_SERVICE_KEY,options),client=createClient(url,anon,options);
const report={measuredAt:new Date().toISOString(),scope:'One real hosted Old Money/Work shopping request, no weather or entered budget, limit five. Controlled catalog pool; not exact screenshot account or capacity proof.',checks:[]};
const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('Audit assertion failed');};
const reportName=process.env.OLD_MONEY_REPORT||'v115-old-money-work-hosted.json';
if(!/^[a-z0-9-]+\.json$/.test(reportName))throw Error('Invalid report filename');
let owner;
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(created.error||!created.data.user)throw Error('Audit provisioning failed');owner=created.data.user.id;report.fixtureUserId=owner;
 const login=await client.auth.signInWithPassword({email,password});if(login.error)throw Error('Audit login failed');
 if((await client.from('profiles').update({trial_started_at:new Date().toISOString()}).eq('id',owner)).error)throw Error('Audit trial failed');
 const judge={},structure={};
 for(const [name,target] of [['occasionRules',judge],['outfitStructure',structure]])Function('exports',ts.transpileModule(await readFile(`backend/supabase/functions/_shared/${name}.ts`,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(target);
 const catalogs=await Promise.all(['tops','bottoms','shoes'].map(category=>admin.from('shop_catalog').select('*').eq('is_active',true).eq('category',category).order('id').limit(500)));
 if(catalogs.some(c=>c.error))throw Error('Audit catalog failed');
 const rows=catalogs.flatMap(c=>c.data).filter(row=>!structure.isNonOutfitItem(row)&&!judge.styleJudgeReason([row],'old_money','Work',{}));
 const outer=row=>/\b(blazers?|jackets?|coats?|cardigans?)\b/i.test(row.name||'');
 const pools=[rows.filter(row=>row.category==='tops'&&!outer(row)).slice(0,10),rows.filter(row=>row.category==='tops'&&outer(row)).slice(0,8),rows.filter(row=>row.category==='bottoms').slice(0,10),rows.filter(row=>row.category==='shoes').slice(0,12)];
 const selected=pools.flat().slice(0,40);report.candidateCount=selected.length;
 check('all_three_catalog_slots_available',pools.every(pool=>pool.length>0));
 const provided=selected.map(row=>({...row,id:`shop_${row.id}`,isShopItem:true,isWardrobe:false,imageUrl:row.image_url,shopUrl:row.source_url}));
 const started=performance.now();
 const response=await fetch(new URL('/functions/v1/generate-outfits',url),{method:'POST',headers:{apikey:anon,Authorization:`Bearer ${login.data.session.access_token}`,'Content-Type':'application/json'},body:JSON.stringify({stylePreferences:'old_money',occasion:'Work',prompt:'Work',preferences:{},limit:5,wardrobeItems:provided,useProvidedWardrobeOnly:true,wardrobeOnly:false,deckSalt:randomUUID()}),signal:AbortSignal.timeout(120000)});
 const data=await response.json();report.response={status:response.status,latencyMs:Math.round(performance.now()-started),code:data.code??null,error:data.error??null,provider:data.provider??null,providerAttempts:data.providerAttempts??[],outfitCount:data.outfits?.length??0};
 check('usable_outfits_returned',response.ok&&data.outfits?.length>0&&data.outfits.length<=5);
 const returnedIds=[...new Set(data.outfits.flatMap(o=>o.items.map(item=>String(item.id).replace(/^shop_/,''))))];
 const authoritative=await admin.from('shop_catalog').select('id,name,price,currency,source_url,is_active,brand').in('id',returnedIds);
 if(authoritative.error)throw Error('Audit authoritative catalog read failed');
 const byId=new Map(authoritative.data.map(row=>[`shop_${row.id}`,row]));
 report.supplementedIds=returnedIds.filter(id=>!selected.some(row=>row.id===id));
 report.outfits=data.outfits.map(outfit=>({items:outfit.items.map(item=>({id:item.id,name:item.name,price:item.price,currency:item.currency})),structureReason:structure.outfitStructureReason(outfit.items,!!data.layered),styleReason:judge.styleJudgeReason(outfit.items,'old_money','Work',{})}));
 check('every_outfit_passes_structure_and_style',report.outfits.every(o=>!o.structureReason&&!o.styleReason));
 check('every_item_matches_authoritative_active_catalog',data.outfits.every(o=>o.items.every(item=>{const row=byId.get(item.id);return row&&row.is_active&&row.name===item.name&&Number(row.price)===Number(item.price)&&row.currency===item.currency&&row.source_url===item.shopUrl;})));
 report.exactFiveReturned=data.outfits.length===5;
}catch(error){report.failure=error.message.startsWith('Audit ')?error.message:error.name;process.exitCode=1;}
finally{
 report.cleanup={};try{report.cleanup.sessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;}catch{report.cleanup.sessionsRevoked=false;}
 if(owner)try{report.cleanup.ownerDeleted=!(await admin.auth.admin.deleteUser(owner)).error;}catch{report.cleanup.ownerDeleted=false;}
 report.passed=report.checks.length===4&&report.checks.every(c=>c.passed)&&report.cleanup.sessionsRevoked&&report.cleanup.ownerDeleted&&!report.failure;
 if(!report.passed)process.exitCode=1;
 await writeFile('docs/reports/readiness-evidence/'+reportName,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}
