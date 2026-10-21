/** Actual store/review code, real OS files and a fresh process; cloud confirmation stubbed. */
import {readFile,writeFile,mkdir,readdir,stat,rename,rm,mkdtemp} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
import {transpileModule,ModuleKind,ScriptTarget} from 'typescript';
import {createStore} from 'zustand/vanilla';
import assert from 'node:assert/strict';
globalThis.__DEV__=false;
const owner='11111111-1111-4111-8111-111111111111';
async function runtime(root,replay=false,confirmedRows=[]){
 const hashes={},modules={};let cloudCalls=0;
 const compile=async file=>{const source=await readFile(new URL('../../'+file,import.meta.url),'utf8');hashes[file]=createHash('sha256').update(source).digest('hex');return transpileModule(source,{compilerOptions:{module:ModuleKind.CommonJS,target:ScriptTarget.ES2022}}).outputText;};
 const documents=join(root,'Documents');
 const fs={documentDirectory:pathToFileURL(documents).href+'/',
  getInfoAsync:async uri=>{try{const s=await stat(fileURLToPath(uri));return{exists:true,isDirectory:s.isDirectory(),size:s.size};}catch(e){if(e.code==='ENOENT')return{exists:false};throw e;}},
  makeDirectoryAsync:uri=>mkdir(fileURLToPath(uri),{recursive:true}),writeAsStringAsync:(uri,text)=>writeFile(fileURLToPath(uri),text),readAsStringAsync:uri=>readFile(fileURLToPath(uri),'utf8'),
  moveAsync:({from,to})=>rename(fileURLToPath(from),fileURLToPath(to)),deleteAsync:uri=>rm(fileURLToPath(uri),{recursive:true,force:true}),readDirectoryAsync:uri=>readdir(fileURLToPath(uri))};
 const reviews={},events={},media={},slice={};
 new Function('exports',await compile('store/authEvents.ts'))(events);events.authEvents.emitLogin(owner);
 new Function('exports','require',await compile('src/services/scanReviews.ts'))(reviews,name=>{if(name==='expo-file-system/legacy')return fs;if(name==='react-native')return{Platform:{OS:'ios'}};if(name.endsWith('/authEvents'))return events;throw Error('Unexpected review dependency');});
 new Function('exports',await compile('src/utils/confirmedOutfitMedia.ts'))(media);
 new Function('exports','require',await compile('store/slices/wardrobeSyncSlice.ts'))(slice,name=>{
  if(name.endsWith('/logger'))return{__esModule:true,default:{debug(){}}};
  if(name.endsWith('/lib/supabase'))return{supabase:{}};
  if(name.endsWith('/wardrobeSyncService'))return{fetchItemsFromServer:async()=>confirmedRows,fetchWearLogsFromServer:async()=>[],reconcileCloudItems:remote=>remote,processPendingActions:async actions=>{cloudCalls++;if(replay)throw Error('Replay must not repeat cloud writes');return{processedIds:actions.map(a=>a.id),updatedItems:actions.map(a=>a.payload)};}};
  if(name.endsWith('/helpers'))return{calculateStreak:()=>0};
  if(name.endsWith('/auth'))return{__esModule:true,default:{getState:()=>({user:{id:owner}})}};
  if(name.endsWith('/authEvents'))return events;
  if(name.endsWith('/confirmedOutfitMedia'))return media;
  if(name.endsWith('/scanReviews'))return{findConfirmedScanReviewItems:(...args)=>reviews.findConfirmedScanReviewItems(...args),acknowledgeSavedScanReviewItems:async(...args)=>{if(!replay)throw Error('Injected filesystem cleanup interruption');return reviews.acknowledgeSavedScanReviewItems(...args);}};
  throw Error('Unexpected sync dependency');
 });
 const state=createStore()((set,get,api)=>({...slice.createSyncSlice(set,get,api),items:[],outfits:[],wearLogs:[]}));
 Object.assign(state,{persist:{getOptions:()=>({name:'wardrobe',version:1,partialize:s=>({items:s.items,pendingActions:s.pendingActions,pendingReviewAcknowledgements:s.pendingReviewAcknowledgements,wardrobeCapacityIssue:s.wardrobeCapacityIssue}),storage:{setItem:async(_key,value)=>writeFile(join(root,'wardrobe.json'),JSON.stringify(value))}})}});
 return{state,reviews,hashes,cloudCalls:()=>cloudCalls};
}
if(process.argv[2]==='--replay'){
 const root=process.argv[3],manifest=JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
 const app=await runtime(root,true,[{id:manifest.itemIds[0],userId:owner,imageUrl:'https://example.invalid/already-confirmed.jpg'}]),saved=JSON.parse(await readFile(join(root,'wardrobe.json'),'utf8'));
 app.state.setState(saved.state);
 assert.deepEqual(app.state.getState().pendingActions,[]);assert.deepEqual(app.state.getState().pendingReviewAcknowledgements,[]);assert.deepEqual(app.state.getState().items,[]);
 await app.state.getState().rehydrateFromCloud(true);
 assert.equal(app.cloudCalls(),0);assert.deepEqual(app.state.getState().pendingReviewAcknowledgements,[]);
 const review=await app.reviews.loadScanReview(owner,manifest.reviewId);
 assert.deepEqual(review.detectedItems.map(i=>i.reviewItemId),manifest.itemIds.slice(1));
 assert.equal(createHash('sha256').update(Buffer.from(review.detectedItems[0].frameImage.split(',')[1],'base64')).digest('hex'),manifest.imageHash);
 assert.deepEqual(JSON.parse(await readFile(join(root,'wardrobe.json'),'utf8')).state.pendingReviewAcknowledgements,[]);
 process.stdout.write(JSON.stringify({legacyReviewReconciledFromAuthoritativeFetch:true,noRepeatedCloudWrite:true,onlyConfirmedCandidateRemoved:true,unsavedImageBytesExact:true,journalClearPersisted:true,sourceHashes:app.hashes})+'\n');
}else{
 const root=await mkdtemp(join(tmpdir(),'aiwardrobe-review-journal-'));let report;
 try{
  const app=await runtime(root),bytes=await readFile('scripts/validation/fixtures/opaque-photo/photo-1.jpg');
  const imageHash=createHash('sha256').update(bytes).digest('hex');
  const saved=await app.reviews.commitScanReview(owner,{detectedItems:[0,1].map(i=>({itemType:i?'pants':'jacket',frameImage:'data:image/jpeg;base64,'+bytes.toString('base64')}))},'queue1');
  const itemIds=saved.detectedItems.map(i=>i.reviewItemId),garment={id:itemIds[0],userId:owner,imageUrl:'https://example.invalid/already-confirmed.jpg',updatedAt:new Date().toISOString()};
  app.state.setState({items:[garment],pendingActions:[{id:'confirmed-action',type:'add_item',payload:garment,createdAt:garment.updatedAt}]});
  const original=console.error;let cleanupErrors=0;console.error=()=>{cleanupErrors++;};
  try{await app.state.getState().syncToServer();}finally{console.error=original;}
  assert.equal(cleanupErrors,1);assert.equal(app.cloudCalls(),1);
  const disk=JSON.parse(await readFile(join(root,'wardrobe.json'),'utf8'));
  assert.deepEqual(disk.state.pendingActions,[]);assert.deepEqual(disk.state.pendingReviewAcknowledgements,[itemIds[0]]);
  assert.equal((await app.reviews.loadScanReview(owner,saved.reviewId)).detectedItems.length,2);
  // Represent the old client format: no cleanup journal and no cached garments.
  delete disk.state.pendingReviewAcknowledgements;disk.state.items=[];
  await writeFile(join(root,'wardrobe.json'),JSON.stringify(disk));
  await writeFile(join(root,'manifest.json'),JSON.stringify({reviewId:saved.reviewId,itemIds,imageHash}));
  const next=JSON.parse((await promisify(execFile)(process.execPath,[fileURLToPath(import.meta.url),'--replay',root])).stdout);
  report={passed:true,measuredAt:new Date().toISOString(),scope:'Actual sync slice and review-storage modules through OS filesystem adapters, injected cleanup failure and a fresh Node process; pre-journal snapshot with empty cache, authoritative cloud-read fixture and no repeat cloud write; no native/UI/production-capacity claim',cloudAckAndJournalPersistedBeforeCleanup:true,reviewRetainedAfterCleanupFailure:true,...next};
 }finally{await rm(root,{recursive:true,force:true});}
 report.fixtureFilesRemoved=true;
 await writeFile('docs/reports/readiness-evidence/scan-review-legacy-recovery.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}
