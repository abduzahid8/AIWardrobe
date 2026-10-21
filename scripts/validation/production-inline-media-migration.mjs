/** Fixture-only migration proof. Never selects existing customer inline images. */
import 'dotenv/config';import {createClient} from '@supabase/supabase-js';import {readFile,writeFile} from 'node:fs/promises';import {randomUUID,createHash} from 'node:crypto';
import {loadWardrobeSync} from './load-wardrobe-sync.mjs';
import {migrateInlineMedia,planInlineMedia,restoreInlineMedia} from '../maintenance/inline-wardrobe-media.mjs';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;if(!key)throw Error('Fixture credential missing');
const options={auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:init?.signal?AbortSignal.any([init.signal,AbortSignal.timeout(20000)]):AbortSignal.timeout(20000)})}};
const syncAudit=process.env.INLINE_MEDIA_SYNC_AUDIT==='1';
const rollbackAudit=process.env.INLINE_MEDIA_ROLLBACK_AUDIT==='1';
if(syncAudit&&rollbackAudit)throw Error('Choose one fixture variant');
const admin=createClient(base,key,options),users=[],paths=[],report={measuredAt:new Date().toISOString(),scope:'Only two tagged disposable accounts and repository PNG; no existing customer rows migrated',checks:[],cleanup:[]};
const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('AssertionError');};
try{
 for(let i=0;i<2;i++){
  const tag=randomUUID(),email=`readiness-${tag}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
  const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:tag}});if(created.error||!created.data.user)throw Error('FixtureCreateFailed');
  const user={id:created.data.user.id,client:createClient(base,anon,options)};users.push(user);
  const login=await user.client.auth.signInWithPassword({email,password});if(login.error)throw Error('FixtureLoginFailed');
 }
 const bytes=await readFile(new URL('../../assets/images/basic_outfit_1.png',import.meta.url)),image=`data:image/png;base64,${bytes.toString('base64')}`;
 const owner=users[0],other=users[1];report.fixtureBytes=bytes.length;
 async function seed(){const id=randomUUID();const r=await owner.client.from('clothing_items').insert({id,user_id:owner.id,name:'Readiness inline migration fixture',category:'top',image_url:image}).select('id,user_id,image_url,updated_at').single();if(r.error)throw Error('FixtureInsertFailed');return r.data;}
 const row=await seed(),plan=await planInlineMedia(row,base);paths.push(plan.path);
 const migrated=await migrateInlineMedia(admin,row,base);check('fixture_migrated',migrated.status==='migrated');
 const ownDownload=await owner.client.storage.from('wardrobe-images').download(plan.path);const downloaded=ownDownload.data?Buffer.from(await ownDownload.data.arrayBuffer()):null;
 check('owner_download_preserves_exact_original_bytes',!ownDownload.error&&downloaded?.equals(bytes));
 check('sha256_matches_original',migrated.digest===createHash('sha256').update(bytes).digest('hex'));
 const foreign=await other.client.storage.from('wardrobe-images').download(plan.path);check('other_owner_cannot_download',!!foreign.error);
 const publicRead=await fetch(migrated.uri.replace('/authenticated/','/public/'),{signal:AbortSignal.timeout(20000)});await publicRead.arrayBuffer();check('public_endpoint_denied',!publicRead.ok);
 const saved=await owner.client.from('clothing_items').select('id,user_id,image_url,name,wear_count,updated_at').eq('id',row.id).single();
 check('reference_and_metadata_preserved',!saved.error&&saved.data.image_url===migrated.uri&&saved.data.name==='Readiness inline migration fixture'&&saved.data.wear_count===0);
 report.inlineUriBytes=Buffer.byteLength(image);report.storageReferenceBytes=Buffer.byteLength(migrated.uri);
 if(syncAudit){
  const sync=await loadWardrobeSync(owner.client,base,owner.id);report.executedSyncSourceHashes=sync.hashes;
  const cached={id:row.id,userId:owner.id,imageUrl:image,category:'top',primaryColor:'blue',seasons:['summer'],occasions:['casual'],wearCount:0,isFavorite:true,name:'Readiness inline migration fixture',createdAt:new Date().toISOString(),updatedAt:new Date(Date.now()+60000).toISOString()};
  const action={id:randomUUID(),type:'add_item',payload:cached,createdAt:new Date().toISOString()};
  paths.push(`${owner.id}/${row.id}/original.png`);
  const result=await sync.processPendingActions([action],[cached],owner.id);
  check('legacy_pending_action_confirmed',result.processedIds.includes(action.id));
  const synced=await owner.client.from('clothing_items').select('image_url,is_favorite').eq('id',row.id).single();
  check('stale_inline_cache_does_not_restore_inline_database_payload',!synced.error&&sync.media.wardrobeMediaReference(synced.data.image_url)?.owner===owner.id&&synced.data.is_favorite===true);
  check('local_queue_receives_durable_private_reference',result.updatedItems[0]?.imageUrl===synced.data.image_url);
  const syncedImage=await owner.client.storage.from('wardrobe-images').download(`${owner.id}/${row.id}/original.png`);
  check('reuploaded_cached_image_preserves_original_bytes',!syncedImage.error&&Buffer.from(await syncedImage.data.arrayBuffer()).equals(bytes));
 }
 if(rollbackAudit){
  const edit=await owner.client.from('clothing_items').update({name:'Readiness rollback concurrent edit'}).eq('id',row.id);if(edit.error)throw Error('FixtureEditFailed');
  const old=await restoreInlineMedia(admin,saved.data,migrated,base);check('rollback_rejects_post_migration_competing_edit',old.status==='conflict');
  const current=await owner.client.from('clothing_items').select('id,user_id,image_url,updated_at,name').eq('id',row.id).single();
  check('refused_rollback_retains_private_reference_and_edit',!current.error&&current.data.image_url===migrated.uri&&current.data.name==='Readiness rollback concurrent edit');
  const restored=await restoreInlineMedia(admin,current.data,migrated,base);
  const final=await owner.client.from('clothing_items').select('image_url,name').eq('id',row.id).single();
  check('current_version_rollback_restores_exact_inline_bytes_and_keeps_edit',restored.status==='restored'&&!final.error&&final.data.image_url===image&&final.data.name==='Readiness rollback concurrent edit');
  const backup=await owner.client.storage.from('wardrobe-images').download(plan.path);
  check('rollback_retains_verified_private_backup',!backup.error&&Buffer.from(await backup.data.arrayBuffer()).equals(bytes));
 }
 const stale=await seed(),stalePlan=await planInlineMedia(stale,base);paths.push(stalePlan.path);
 const edited=await owner.client.from('clothing_items').update({name:'Readiness concurrent edit'}).eq('id',stale.id);if(edited.error)throw Error('FixtureEditFailed');
 const collision=await migrateInlineMedia(admin,stale,base);check('competing_edit_not_overwritten',collision.status==='conflict');
 const retained=await owner.client.from('clothing_items').select('image_url,name').eq('id',stale.id).single();check('conflicted_row_keeps_original_and_new_name',!retained.error&&retained.data.image_url===image&&retained.data.name==='Readiness concurrent edit');
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 if(paths.length){const removed=await admin.storage.from('wardrobe-images').remove(paths);report.cleanup.push({objectsRemoved:!removed.error});if(removed.error)process.exitCode=1;}
 for(const user of users){await user.client.auth.signOut({scope:'global'}).catch(()=>{});const removed=await admin.auth.admin.deleteUser(user.id);report.cleanup.push({accountDeleted:!removed.error});if(removed.error)process.exitCode=1;}
 report.passed=report.checks.length===((syncAudit||rollbackAudit)?12:8)&&report.checks.every(x=>x.passed)&&report.cleanup.every(x=>Object.values(x).every(Boolean))&&!report.failure;
 await writeFile(`docs/reports/readiness-evidence/${syncAudit?'production-inline-media-sync-compatibility':rollbackAudit?'production-inline-media-rollback':'production-inline-media-migration'}.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}
