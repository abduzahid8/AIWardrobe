/** Real OS files/fresh processes through production review storage; no native bridge claim. */
import {readFile,writeFile,mkdir,readdir,stat,rename,rm,mkdtemp} from 'node:fs/promises';
import {join} from 'node:path';import {tmpdir} from 'node:os';import {pathToFileURL,fileURLToPath} from 'node:url';
import {execFile} from 'node:child_process';import {promisify} from 'node:util';import {createHash} from 'node:crypto';
import {transpileModule,ModuleKind} from 'typescript';import assert from 'node:assert/strict';
const owner='11111111-1111-4111-8111-111111111111',other='22222222-2222-4222-8222-222222222222';
async function load(documents){
 const compile=async path=>transpileModule(await readFile(new URL(path,import.meta.url),'utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
 const events={},reviews={};new Function('exports',await compile('../../store/authEvents.ts'))(events);
 const fs={documentDirectory:pathToFileURL(documents).href+'/',
  getInfoAsync:async uri=>{try{const s=await stat(fileURLToPath(uri));return{exists:true,isDirectory:s.isDirectory(),size:s.size};}catch(e){if(e.code==='ENOENT')return{exists:false};throw e;}},
  makeDirectoryAsync:uri=>mkdir(fileURLToPath(uri),{recursive:true}),writeAsStringAsync:(uri,text)=>writeFile(fileURLToPath(uri),text),readAsStringAsync:uri=>readFile(fileURLToPath(uri),'utf8'),
  moveAsync:({from,to})=>rename(fileURLToPath(from),fileURLToPath(to)),deleteAsync:uri=>rm(fileURLToPath(uri),{recursive:true,force:true}),readDirectoryAsync:uri=>readdir(fileURLToPath(uri))};
 new Function('exports','require',await compile('../../src/services/scanReviews.ts'))(reviews,name=>{if(name==='expo-file-system/legacy')return fs;if(name==='react-native')return{Platform:{OS:'ios'}};if(name==='../../store/authEvents')return events;throw Error('Unexpected dependency');});
 events.authEvents.emitLogin(owner);return{reviews,events};
}
if(process.argv[2]==='--read'||process.argv[2]==='--finish'){
 const root=process.argv[3],manifest=JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
 const {reviews,events}=await load(join(root,'new','Documents'));
 const restored=await reviews.loadScanReview(owner,manifest.id);
 const expected=process.argv[2]==='--finish'?manifest.itemIds.slice(1):manifest.itemIds;
 assert.deepEqual(restored.detectedItems.map(item=>item.reviewItemId),expected);
 for(const item of restored.detectedItems)assert.equal(createHash('sha256').update(Buffer.from(item.frameImage.split(',')[1],'base64')).digest('hex'),manifest.imageHash);
 if(process.argv[2]==='--read'){
  const folder=join(root,'new','Documents','aiwardrobe-scan-reviews',owner);
  const corruptPath=join(folder,'33333333-3333-4333-8333-333333333333.json');
  await writeFile(corruptPath,'{');let unavailable=0;
  assert.equal((await reviews.listScanReviews(owner,count=>{unavailable=count;}))[0].id,manifest.id);
  assert.equal(unavailable,1);assert.equal(await readFile(corruptPath,'utf8'),'{');
  await rm(corruptPath);
  const reviewPath=join(folder,manifest.id+'.json'),original=await readFile(reviewPath,'utf8'),damaged=JSON.parse(original);
  damaged.result.detectedItems[1].reviewItemId=damaged.result.detectedItems[0].reviewItemId;
  await writeFile(reviewPath,JSON.stringify(damaged));
  await assert.rejects(reviews.acknowledgeScanReviewItem(owner,manifest.id,manifest.itemIds[0]),/Duplicate/);
  assert.deepEqual((await readdir(folder)).filter(name=>name.endsWith('.saved')),[]);
  await writeFile(reviewPath,original);
  await reviews.acknowledgeScanReviewItem(owner,manifest.id,manifest.itemIds[0]);
  process.stdout.write(JSON.stringify({firstRestartExactImages:true,stableItemIDs:true,partialAcknowledgement:true,corruptRecordIsolatedAndRetained:true,duplicateIDsCannotAcknowledge:true})+'\n');
 }else{
  await reviews.acknowledgeScanReviewItem(owner,manifest.id,manifest.itemIds[1]);
  assert.deepEqual(await reviews.listScanReviews(owner),[]);
  assert.deepEqual(await reviews.findScanReviewBySource(owner,'queue1'),{sourceQueueId:'queue1',detectedItems:[]});
  events.authEvents.emitLogin(other);assert.throws(()=>reviews.loadScanReview(owner,manifest.id),/account changed/);
  events.authEvents.emitLogin(owner);await reviews.clearAllScanReviews(events.authEvents.getRevision());
  process.stdout.write(JSON.stringify({secondRestartOnlyUnsavedItem:true,completedSourceRememberedWithoutAI:true,otherOwnerRejected:true,accountCleanup:true})+'\n');
 }
}else{
 const root=await mkdtemp(join(tmpdir(),'aiwardrobe-review-restart-'));
 try{
  const bytes=await readFile('scripts/validation/fixtures/opaque-photo/photo-1.jpg'),imageHash=createHash('sha256').update(bytes).digest('hex');
  const {reviews}=await load(join(root,'old','Documents'));
  const result={detectedItems:[0,1].map(index=>({itemType:index?'pants':'jacket',color:'beige',style:'casual',description:'Exact provider result',position:index?'lower':'upper',confidence:0,confidenceLevel:'low',detectionSources:['fixture'],frameImage:'data:image/jpeg;base64,'+bytes.toString('base64')}))};
  const saved=await reviews.commitScanReview(owner,result,'queue1');
  await writeFile(join(root,'manifest.json'),JSON.stringify({id:saved.reviewId,itemIds:saved.detectedItems.map(item=>item.reviewItemId),imageHash}));
  await rename(join(root,'old'),join(root,'new'));
  const execute=promisify(execFile);
  const first=JSON.parse((await execute(process.execPath,[fileURLToPath(import.meta.url),'--read',root])).stdout);
  const second=JSON.parse((await execute(process.execPath,[fileURLToPath(import.meta.url),'--finish',root])).stdout);
  const report={passed:true,...first,...second,containerRebased:true,imageBytes:bytes.length,imageHash,scope:'Actual production review storage through OS filesystem adapter and two fresh processes; no native/UI/provider accuracy claim'};
  await writeFile('docs/reports/readiness-evidence/scan-review-restart.json',JSON.stringify(report,null,2)+'\n');process.stdout.write(JSON.stringify(report)+'\n');
 }finally{await rm(root,{recursive:true,force:true});}
}
