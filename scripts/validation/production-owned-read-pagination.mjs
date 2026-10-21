/** Real owner SDK + actual client readers. Fixture-only writes; no provider/media calls. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
import {createClient} from '@supabase/supabase-js';
import {loadWardrobeSync} from './load-wardrobe-sync.mjs';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!service)throw Error('Fixture configuration unavailable');
const report={measuredAt:new Date().toISOString(),scope:'Actual client readers and owner SDK against501garments/1001wear logs; fixture metadata only, not native rendering or capacity',checks:[]};
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
let owner,client;
const call=async(path,body,method='POST')=>{const response=await fetch(new URL(path,base),{method,headers:admin,...body===undefined?{}:{body:JSON.stringify(body)},signal:AbortSignal.timeout(30000)});let data;try{data=await response.json();}catch{}return{ok:response.ok,status:response.status,data};};
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await call('/auth/v1/admin/users',{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(!created.ok||!created.data?.id)throw Error('Fixture setup failed');owner=created.data.id;
 const requests=[];
 client=createClient(base,anon,{auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:async(input,init)=>{const response=await fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(30000)});const route=new URL(typeof input==='string'?input:input.url).pathname.split('/').pop();requests.push({route:route==='clothing_items'||route==='wear_logs'?route:'other',status:response.status,method:init?.method||'GET'});return response;}}});
 const login=await client.auth.signInWithPassword({email,password});if(login.error)throw Error('Fixture login failed');
 const items=Array.from({length:501},()=>({id:randomUUID(),user_id:owner,category:'top',name:'Readiness owned reader fixture',image_url:'https://example.invalid/readiness-reader.jpg'}));
 const stored=await client.from('clothing_items').insert(items);if(stored.error)throw Error('Fixture garment setup failed');
 const logs=Array.from({length:1001},()=>({id:randomUUID(),user_id:owner,item_ids:[items[0].id],date:'2026-10-21'}));
 const worn=await client.from('wear_logs').insert(logs);if(worn.error)throw Error('Fixture log setup failed');
 const runtime=await loadWardrobeSync(client,base,owner);report.sourceHashes=runtime.hashes;requests.length=0;
 const start=performance.now();const garments=await runtime.fetchItemsFromServer();const wardrobeMs=Math.round(performance.now()-start);
 const historyStart=performance.now();const history=await runtime.fetchWearLogsFromServer();const historyMs=Math.round(performance.now()-historyStart);
 const itemIds=new Set(items.map(item=>item.id)),logIds=new Set(logs.map(log=>log.id));
 report.checks.push({name:'complete_owned_wardrobe',count:garments?.length??null,latencyMs:wardrobeMs,passed:garments?.length===501&&new Set(garments.map(item=>item.id)).size===501&&garments.every(item=>item.userId===owner&&itemIds.has(item.id))});
 report.checks.push({name:'complete_owned_wear_history',count:history?.length??null,latencyMs:historyMs,passed:history?.length===1001&&new Set(history.map(log=>log.id)).size===1001&&history.every(log=>log.userId===owner&&logIds.has(log.id)&&log.itemIds.length===1&&log.itemIds[0]===items[0].id)});
 report.requests=[...requests];
 report.checks.push({name:'actual_cursor_page_counts',passed:requests.length===8&&requests.filter(request=>request.route==='clothing_items').length===3&&requests.filter(request=>request.route==='wear_logs').length===5&&requests.every(request=>request.status===200&&request.method==='GET')});
}catch(error){report.failure=/^Fixture .* failed$/.test(error.message)?error.message:error.name;process.exitCode=1;}
finally{
 report.cleanup={};
 if(client){try{const result=await client.auth.signOut({scope:'global'});report.cleanup.refreshSessionsRevoked=!result.error;}catch{report.cleanup.refreshSessionsRevoked=false;}}
 if(owner){try{const removed=await call(`/auth/v1/admin/users/${owner}`,undefined,'DELETE');report.cleanup.userDeleted=removed.ok;}catch{report.cleanup.userDeleted=false;}}
 report.passed=report.checks.length===3&&report.checks.every(check=>check.passed)&&report.cleanup.userDeleted===true&&report.cleanup.refreshSessionsRevoked===true&&!report.failure;
 if(!report.passed)process.exitCode=1;
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-owned-read-pagination-deadline.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
