/** Bounded direct-API capacity check on one disposable free fixture. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import {loadWardrobeSync} from './load-wardrobe-sync.mjs';
import { createClient } from '@supabase/supabase-js';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!key)throw Error('Fixture configuration required');
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(30000)})}};
const admin=createClient(base,key,options),client=createClient(base,anon,options);
const report={measuredAt:new Date().toISOString(),scope:'One disposable free user with expired trial; direct owner SDK inserts, actual sync service retained-item/retry verification; no load/capacity certification',checks:[]};
let owner;
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(created.error||!created.data.user)throw Error('Fixture creation failed');owner=created.data.user.id;
 if((await admin.from('profiles').update({trial_started_at:new Date(Date.now()-8*86400000).toISOString()}).eq('id',owner)).error)throw Error('Fixture expiry setup failed');
 if((await client.auth.signInWithPassword({email,password})).error)throw Error('Fixture login failed');
 const status=await client.rpc('get_subscription_status');
 if(status.error||status.data?.[0]?.tier!=='free')throw Error('Fixture free verification failed');
 const garment=()=>({id:randomUUID(),user_id:owner,category:'top',name:'Readiness capacity fixture',image_url:'https://example.invalid/capacity-fixture.jpg'});
 const rows=Array.from({length:20},garment);
 const setup=await client.from('clothing_items').insert(rows);
 if(setup.error)throw Error('Fixture twenty-item insert failed');
 const extra=await client.from('clothing_items').insert(garment());
 const count=await client.from('clothing_items').select('id',{count:'exact',head:true}).eq('user_id',owner);
 if(count.error)throw Error('Fixture count failed');
 report.observed={extraInsertStatus:extra.status,extraInsertErrorCode:extra.error?.code??null,ownedItemCount:count.count};
 report.checks.push({name:'server_rejects_twenty_first_free_garment',passed:!!extra.error&&count.count===20});
 const existing=await client.from('clothing_items').upsert({...rows[0],name:'Readiness capacity existing row edited'});
 report.checks.push({name:'existing_row_upsert_remains_usable_at_capacity',passed:!existing.error});
 const runtime=await loadWardrobeSync(client,base,owner);report.sourceHashes=runtime.hashes;
 const queued={id:randomUUID(),userId:owner,category:'top',name:'Readiness capacity retained work',imageUrl:'https://example.invalid/capacity-fixture.jpg',updatedAt:new Date().toISOString(),createdAt:new Date().toISOString(),wearCount:0,isFavorite:false};
 const action={id:'capacity-retained-action',type:'add_item',payload:queued,createdAt:queued.createdAt};
 const rejected=await runtime.processPendingActions([action],[queued],owner);
 report.checks.push({name:'actual_service_reports_cap_and_retains_unconfirmed_item',passed:rejected.processedIds.length===0&&rejected.capacityIssue?.limit===20&&rejected.capacityIssue?.actionIds[0]===action.id&&rejected.updatedItems[0]?.imageUrl===queued.imageUrl});
 const removed=await client.from('clothing_items').delete().eq('id',rows[19].id).eq('user_id',owner);
 if(removed.error)throw Error('Fixture capacity release failed');
 const retry=await runtime.processPendingActions([action],[queued],owner);
 report.checks.push({name:'same_queued_action_confirms_after_capacity_is_freed',passed:retry.processedIds.includes(action.id)&&!retry.capacityIssue});
 const final=await client.from('clothing_items').select('id',{count:'exact',head:true}).eq('user_id',owner);
 report.checks.push({name:'retry_creates_one_row_at_original_id_and_respects_cap',passed:!final.error&&final.count===20&&!(await client.from('clothing_items').select('id').eq('id',queued.id).eq('user_id',owner).single()).error});

}catch(error){report.failure=/^Fixture .* failed$/.test(error.message)?error.message:error.name;}
finally{
 report.cleanup={};try{report.cleanup.refreshSessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;}catch{report.cleanup.refreshSessionsRevoked=false;}
 if(owner)try{report.cleanup.userDeleted=!(await admin.auth.admin.deleteUser(owner)).error;}catch{report.cleanup.userDeleted=false;}
 report.passed=report.checks.length===5&&report.checks.every(x=>x.passed)&&report.cleanup.refreshSessionsRevoked&&report.cleanup.userDeleted&&!report.failure;
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-wardrobe-cap-feedback.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}
