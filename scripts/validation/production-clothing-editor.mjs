/** Runs the actual editor service against disposable ordinary production users. */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { randomUUID } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { transpileModule, ModuleKind } from 'typescript';
const url=process.env.EXPO_PUBLIC_SUPABASE_URL, anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if(!process.env.SUPABASE_SERVICE_KEY)throw Error('Fixture setup credential required');
const options={auth:{persistSession:false,autoRefreshToken:false}};
const admin=createClient(url,process.env.SUPABASE_SERVICE_KEY,options);
const users=[], report={scope:'Actual clothing editor service with disposable production accounts; no existing-user changes',checks:[],cleanup:[]};
let client;
const code=transpileModule(await readFile('src/services/clothingEditorService.ts','utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
const module={};
const syncModule={};
const syncCode=transpileModule(await readFile('store/wardrobeSyncService.ts','utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
new Function('exports','require',syncCode)(syncModule,path=>path.endsWith('supabase')?{get supabase(){return client;}}:{authEvents:{getRevision:()=>0}});
new Function('exports','require',code)(module,path=>path.endsWith('supabase')?{get supabase(){return client;}}:{authEvents:{getRevision:()=>0}});
const check=(name,passed)=>report.checks.push({name,passed});
try{
 for(let i=0;i<2;i++){
  const email=`readiness-${randomUUID()}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
  const {data,error}=await admin.auth.admin.createUser({email,password,email_confirm:true,user_metadata:{readiness_test:true,username:'Editor audit'}});
  if(error)throw error;users.push(data.user.id);
  const c=createClient(url,anon,options);const login=await c.auth.signInWithPassword({email,password});if(login.error)throw login.error;
  if(i===0)client=c;else users.otherClient=c;
 }
 const owner=client,id=randomUUID();
 const created=await owner.from('clothing_items').insert({id,user_id:users[0],category:'top',sub_category:'t-shirt',name:'Disposable editor audit',image_url:'https://example.invalid/editor.png'});
 if(created.error)throw created.error;
 const saved=await module.saveClothingEdits(id,{category:'bottom',subCategory:'bottoms',primaryColor:'Black',colorHex:'#000000',seasons:['winter'],occasions:['work']});
 check('actual_service_confirms_owner_save',saved.id===id);
 const read=await owner.from('clothing_items').select('category,sub_category,type,primary_color,color_hex,seasons,occasions').eq('id',id).single();
 check('all_edited_fields_persist',!read.error&&read.data.category==='bottom'&&read.data.sub_category==='bottoms'&&read.data.type==='bottoms'&&read.data.primary_color==='Black'&&read.data.color_hex==='#000000'&&read.data.seasons[0]==='winter'&&read.data.occasions[0]==='work');
 client=users.otherClient;
 let denied=false;try{await module.saveClothingEdits(id,{primaryColor:'forged'});}catch{denied=true;}check('other_account_edit_rejected',denied);
 const unchanged=await owner.from('clothing_items').select('primary_color').eq('id',id).single();check('owner_item_unchanged_after_attack',unchanged.data?.primary_color==='Black');
 client=owner;let missing=false;try{await module.saveClothingEdits(randomUUID(),{primaryColor:'Blue'});}catch{missing=true;}check('nonexistent_item_not_reported_saved',missing);
 client=createClient(url,anon,options);let unauth=false;try{await module.saveClothingEdits(id,{primaryColor:'Blue'});}catch{unauth=true;}check('unauthenticated_editor_rejected',unauth);
 if(process.env.OUTFIT_PAGE_CHECKS==='1') {
  client=owner;
  const rows=Array.from({length:1001},()=>({id:randomUUID(),user_id:users[0],items:[{id:'test-top',imageUrl:'https://example.invalid/outfit.png'}],date:'2026-10-16',occasion:'casual'}));
  for(let offset=0;offset<rows.length;offset+=250){const added=await owner.from('saved_outfits').insert(rows.slice(offset,offset+250));if(added.error)throw added.error;}
  const savedModule={};
  const code=transpileModule(await readFile('src/services/savedOutfitService.ts','utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
  new Function('exports','require',code)(savedModule,path=>path.endsWith('supabase')?{get supabase(){return client;}}:{authEvents:{getRevision:()=>0}});
  const started=Date.now(), loaded=await savedModule.listCloudOutfits();
  const expected=new Set(rows.map(row=>row.id));
  report.collectionReadMeasurement={expectedRows:1001,returnedRows:loaded.length,durationMs:Date.now()-started};
  check('actual_service_loads_all_1001_saved_outfits',loaded.length===1001&&new Set(loaded.map(row=>row.id)).size===1001&&loaded.every(row=>expected.has(row.id)));
  client=users.otherClient;
  const isolated=await savedModule.listCloudOutfits();
  check('saved_collection_owner_isolation',isolated.length===0);
 }
 if(process.env.OUTFIT_CHECKS==='1') {
  const dateModule={}, savedModule={};
  const compile=async path=>transpileModule(await readFile(path,'utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
  new Function('exports',await compile('src/utils/calendarDate.ts'))(dateModule);
  new Function('exports','require',await compile('src/services/savedOutfitService.ts'))(savedModule,path=>path.endsWith('supabase')?{get supabase(){return client;}}:path.endsWith('calendarDate')?dateModule:{authEvents:{getRevision:()=>0}});
  client=owner;
  const outfitId=randomUUID(), input={id:outfitId,date:'2026-10-16',occasion:'casual',items:[{id,imageUrl:'https://example.invalid/editor.png'}],caption:'Disposable saved outfit audit'};
  await savedModule.saveCloudOutfit(input);
  const saved=await owner.from('saved_outfits').select('id,items,caption').eq('id',outfitId).single();
  check('actual_outfit_save_persists',!saved.error&&saved.data?.id===outfitId&&saved.data.items[0].id===id);
  await savedModule.saveCloudOutfit(input);
  const repeated=await owner.from('saved_outfits').select('id').eq('id',outfitId);
  check('outfit_save_replay_has_one_row',repeated.data?.length===1);
  client=users.otherClient;
  const hidden=await client.from('saved_outfits').select('id').eq('id',outfitId);
  check('other_account_cannot_read_saved_outfit',hidden.data?.length===0);
  let blocked=false;try{await savedModule.saveCloudOutfit({...input,caption:'forged'});}catch{blocked=true;}
  check('other_account_cannot_overwrite_saved_outfit',blocked);
  await savedModule.removeCloudOutfit(outfitId);
  const preserved=await owner.from('saved_outfits').select('caption').eq('id',outfitId).single();
  check('other_account_delete_preserves_owner_outfit',preserved.data?.caption===input.caption);
  client=owner;await savedModule.removeCloudOutfit(outfitId);
  const removed=await owner.from('saved_outfits').select('id').eq('id',outfitId);
  check('owner_unsave_removes_cloud_row',removed.data?.length===0);
  await savedModule.removeCloudOutfit(outfitId);check('unsave_replay_idempotent',true);
  client=createClient(url,anon,options);let unauth=false;try{await savedModule.saveCloudOutfit(input);}catch{unauth=true;}
  check('unauthenticated_outfit_save_rejected',unauth);
 }
 if(process.env.READ_CHECKS==='1') {
  client=owner;
  const rows=Array.from({length:500},()=>({id:randomUUID(),user_id:users[0],category:'top',name:'Disposable pagination audit',image_url:'https://example.invalid/pagination.png'}));
  for(let offset=0;offset<rows.length;offset+=250){const inserted=await owner.from('clothing_items').insert(rows.slice(offset,offset+250));if(inserted.error)throw inserted.error;}
  const started=Date.now();
  const loaded=await syncModule.fetchItemsFromServer();
  report.readMeasurement={expectedRows:501,returnedRows:loaded?.length??null,durationMs:Date.now()-started};
  check('actual_service_loads_all_501_owner_rows',loaded?.length===501&&loaded.every(item=>item.userId===users[0])&&rows.every(row=>loaded.some(item=>item.id===row.id)));
  client=users.otherClient;
  const isolated=await syncModule.fetchItemsFromServer();
  check('other_account_reads_no_owner_rows',Array.isArray(isolated)&&isolated.length===0);
 }
 if(process.env.MUTATION_CHECKS==='1') {
  client=owner;
  const action=value=>({id:'favorite-replay',type:'update_item',payload:{itemId:id,isFavorite:value},createdAt:new Date().toISOString()});
  const liked=await syncModule.processPendingActions([action(true)],[],users[0]);
  const likedRow=await owner.from('clothing_items').select('is_favorite').eq('id',id).single();
  check('favorite_true_persists',liked.processedIds.length===1&&likedRow.data?.is_favorite===true);
  await syncModule.processPendingActions([action(false)],[],users[0]);
  await syncModule.processPendingActions([action(false)],[],users[0]);
  const unliked=await owner.from('clothing_items').select('is_favorite').eq('id',id).single();
  check('repeated_false_is_idempotent',unliked.data?.is_favorite===false);
  client=users.otherClient;
  const deniedFavorite=await syncModule.processPendingActions([action(true)],[],users[1]);
  const stillFalse=await owner.from('clothing_items').select('is_favorite').eq('id',id).single();
  check('cross_account_favorite_not_acknowledged',deniedFavorite.processedIds.length===0&&stillFalse.data?.is_favorite===false);
  await module.deleteClothingItem(id);
  const preserved=await owner.from('clothing_items').select('id').eq('id',id);
  check('cross_account_delete_preserves_owner_item',preserved.data?.length===1);
  client=owner;
  await module.deleteClothingItem(id);
  const removed=await owner.from('clothing_items').select('id').eq('id',id);
  check('owner_delete_persists',!removed.error&&removed.data.length===0);
  await module.deleteClothingItem(id);
  check('owner_delete_replay_is_idempotent',true);
 }

}finally{
 for(const id of users){const {error}=await admin.auth.admin.deleteUser(id);report.cleanup.push({accountDeleted:!error});}
 await writeFile(process.env.OUTFIT_PAGE_CHECKS==='1'?'docs/reports/readiness-evidence/production-saved-outfit-pagination.json':process.env.OUTFIT_CHECKS==='1'?'docs/reports/readiness-evidence/production-saved-outfits.json':process.env.READ_CHECKS==='1'?'docs/reports/readiness-evidence/production-wardrobe-pagination.json':process.env.MUTATION_CHECKS==='1'?'docs/reports/readiness-evidence/production-wardrobe-mutations.json':'docs/reports/readiness-evidence/production-clothing-editor.json',JSON.stringify(report,null,2)+'\n');
 process.stdout.write(JSON.stringify(report)+'\n');
}
if(report.checks.length!==(6+(process.env.MUTATION_CHECKS==='1'?6:0)+(process.env.READ_CHECKS==='1'?2:0)+(process.env.OUTFIT_CHECKS==='1'?8:0)+(process.env.OUTFIT_PAGE_CHECKS==='1'?2:0))||report.checks.some(c=>!c.passed)||report.cleanup.some(c=>!c.accountDeleted))process.exitCode=1;
