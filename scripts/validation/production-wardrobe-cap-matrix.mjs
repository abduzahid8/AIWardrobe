/** Bounded direct-API capacity check on one disposable free fixture. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { createClient } from '@supabase/supabase-js';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!key)throw Error('Fixture configuration required');
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(30000)})}};
const admin=createClient(base,key,options),client=createClient(base,anon,options);
const report={measuredAt:new Date().toISOString(),scope:'One disposable free user with expired trial; direct owner SDK inserts, 111 simultaneous fixture garments maximum; six-request concurrency; no load/capacity certification',checks:[]};
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
 const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('Fixture assertion failed');};
 const countOwned=async()=>{const r=await client.from('clothing_items').select('id',{count:'exact',head:true}).eq('user_id',owner);if(r.error)throw Error('Fixture count failed');return r.count;};
 if((await client.from('clothing_items').delete().eq('id',rows[19].id).eq('user_id',owner)).error)throw Error('Fixture slot setup failed');
 const concurrent=await Promise.all(Array.from({length:6},()=>client.from('clothing_items').insert(garment())));
 report.concurrent={requests:6,accepted:concurrent.filter(r=>!r.error).length,rejected:concurrent.filter(r=>r.error?.code==='P0001'&&r.error?.message==='WARDROBE_LIMIT_REACHED').length};
 check('six_concurrent_free_inserts_admit_exactly_one_remaining_slot',report.concurrent.accepted===1&&report.concurrent.rejected===5&&await countOwned()===20);
 if((await client.from('clothing_items').delete().eq('user_id',owner)).error)throw Error('Fixture reset failed');
 const bulk=await client.from('clothing_items').insert(Array.from({length:21},garment));
 check('oversized_free_batch_rolls_back_all_rows',bulk.error?.message==='WARDROBE_LIMIT_REACHED'&&await countOwned()===0);
 const subId=randomUUID();
 if((await admin.from('subscriptions').insert({id:subId,user_id:owner,tier:'lite',status:'active',end_date:new Date(Date.now()+86400000).toISOString(),platform:'ios',revenuecat_managed:false})).error)throw Error('Fixture Lite setup failed');
 const liteRows=Array.from({length:100},garment);
 const lite=await client.from('clothing_items').insert(liteRows);
 check('verified_lite_accepts_one_hundred_garments',!lite.error&&await countOwned()===100);
 const liteExtra=await client.from('clothing_items').insert(garment());
 check('verified_lite_rejects_one_hundred_first_garment',liteExtra.error?.message==='WARDROBE_LIMIT_REACHED'&&await countOwned()===100);
 if((await admin.from('subscriptions').update({end_date:new Date(Date.now()-60000).toISOString()}).eq('id',subId).eq('user_id',owner)).error)throw Error('Fixture downgrade failed');
 const edit=await client.from('clothing_items').upsert({...liteRows[0],name:'Readiness downgraded editable fixture'});
 const denied=await client.from('clothing_items').insert(garment());
 check('downgrade_retains_editable_existing_rows_but_blocks_new_capacity',!edit.error&&denied.error?.message==='WARDROBE_LIMIT_REACHED'&&await countOwned()===100);
 if((await admin.from('profiles').update({trial_started_at:new Date().toISOString()}).eq('id',owner)).error)throw Error('Fixture trial setup failed');
 const trial=await client.from('clothing_items').insert(Array.from({length:10},garment));
 check('active_original_trial_grants_premium_capacity_above_lite_cap',!trial.error&&await countOwned()===110);
 if((await admin.from('profiles').update({trial_started_at:new Date(Date.now()-8*86400000).toISOString()}).eq('id',owner)).error)throw Error('Fixture trial expiry failed');
 check('expired_trial_rejects_new_garments', (await client.from('clothing_items').insert(garment())).error?.message==='WARDROBE_LIMIT_REACHED'&&await countOwned()===110);
 if((await admin.from('subscriptions').update({tier:'vip',end_date:new Date(Date.now()+86400000).toISOString()}).eq('id',subId).eq('user_id',owner)).error)throw Error('Fixture VIP setup failed');
 check('verified_vip_grants_capacity_above_lite_cap',!(await client.from('clothing_items').insert(garment())).error&&await countOwned()===111);
 check('owner_cannot_insert_for_another_identity',(await client.from('clothing_items').insert({...garment(),user_id:randomUUID()})).error?.code==='42501');
 const anonymous=createClient(base,anon,options);
 check('anonymous_cannot_insert_for_owner',(await anonymous.from('clothing_items').insert(garment())).error?.code==='42501');

}catch(error){report.failure=/^Fixture .* failed$/.test(error.message)?error.message:error.name;}
finally{
 report.cleanup={};try{report.cleanup.refreshSessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;}catch{report.cleanup.refreshSessionsRevoked=false;}
 if(owner)try{report.cleanup.userDeleted=!(await admin.auth.admin.deleteUser(owner)).error;}catch{report.cleanup.userDeleted=false;}
 report.passed=report.checks.length===12&&report.checks.every(x=>x.passed)&&report.cleanup.refreshSessionsRevoked&&report.cleanup.userDeleted&&!report.failure;
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-wardrobe-cap-matrix.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}
