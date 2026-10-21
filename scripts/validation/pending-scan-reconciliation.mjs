/** Actual OS-file and fresh-process deletion-intent recovery; Expo-shaped FS adapter. */
import {readFile,writeFile,mkdir,readdir,stat,rename,copyFile,rm,mkdtemp} from 'node:fs/promises';
import {join,basename} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {transpileModule,ModuleKind} from 'typescript';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const owner='11111111-1111-4111-8111-111111111111',other='22222222-2222-4222-8222-222222222222';
const exists=async path=>{try{await stat(path);return true;}catch(error){if(error.code==='ENOENT')return false;throw error;}};
async function load(documents){
 const compile=async path=>transpileModule(await readFile(new URL(path,import.meta.url),'utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
 const events={},media={};new Function('exports',await compile('../../store/authEvents.ts'))(events);
 const fs={documentDirectory:pathToFileURL(documents).href+'/',
  getInfoAsync:async uri=>{try{const info=await stat(fileURLToPath(uri));return{exists:true,isDirectory:info.isDirectory(),size:info.size};}catch(error){if(error.code==='ENOENT')return{exists:false};throw error;}},
  makeDirectoryAsync:uri=>mkdir(fileURLToPath(uri),{recursive:true}),copyAsync:({from,to})=>copyFile(fileURLToPath(from),fileURLToPath(to)),
  moveAsync:({from,to})=>rename(fileURLToPath(from),fileURLToPath(to)),deleteAsync:uri=>rm(fileURLToPath(uri),{recursive:true,force:true}),
  writeAsStringAsync:(uri,text)=>writeFile(fileURLToPath(uri),text),readDirectoryAsync:uri=>readdir(fileURLToPath(uri))};
 new Function('exports','require',await compile('../../src/services/pendingScanMedia.ts'))(media,name=>{if(name==='expo-file-system/legacy')return fs;if(name==='../../store/authEvents')return events;throw Error('Unexpected dependency: '+name);});
 events.authEvents.emitLogin(owner);return{media,events,fs};
}
if(process.argv[2]==='--restart'){
 const root=process.argv[3],manifest=JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
 const documents=join(root,'new','Documents'),{media,events,fs}=await load(documents);
 const folder=join(documents,'aiwardrobe-scan-queue',owner),rebased=uri=>join(folder,basename(fileURLToPath(uri)));
 const marker=uri=>join(folder,'.'+basename(fileURLToPath(uri))+'.remove');
 const refs=()=>[manifest.keep];
 await media.reconcilePendingScanMedia(owner,refs);
 assert.ok(await exists(rebased(manifest.keep)));assert.ok(!(await exists(marker(manifest.keep))));
 assert.ok(!(await exists(rebased(manifest.drop))));assert.ok(!(await exists(marker(manifest.drop))));
 assert.ok(!(await exists(join(folder,manifest.partial))));assert.ok(await exists(rebased(manifest.unmarked)));
 assert.ok(await exists(join(documents,'aiwardrobe-scan-queue',other,'foreign.jpg')));assert.ok(await exists(join(folder,'unrelated.txt')));
 await media.removePendingScanMedia(owner,manifest.keep,refs);assert.ok(await exists(rebased(manifest.keep)));
 await media.reconcilePendingScanMedia(owner,refs); // Idempotent second pass.
 const doomed=join(folder,'44444444-4444-4444-8444-444444444444.jpg');await writeFile(doomed,'owned');await writeFile(join(folder,'.'+basename(doomed)+'.remove'),'remove');
 const remove=fs.deleteAsync;let fail=true;
 fs.deleteAsync=async uri=>{if(fail&&uri===pathToFileURL(doomed).href){fail=false;throw Error('injected deletion failure');}return remove(uri);};
 await assert.rejects(media.reconcilePendingScanMedia(owner,refs),/injected deletion failure/);
 assert.ok(await exists(join(folder,'.'+basename(doomed)+'.remove')));
 await media.reconcilePendingScanMedia(owner,refs);assert.ok(!(await exists(doomed)));
 const interrupted=join(folder,'.55555555-5555-4555-8555-555555555555.pending');await writeFile(interrupted,'partial');
 const read=fs.readDirectoryAsync;fs.readDirectoryAsync=async uri=>{events.authEvents.emitLogin(other);return read(uri);};
 await media.reconcilePendingScanMedia(owner,refs);assert.ok(await exists(interrupted));
 events.authEvents.emitLogin(owner);await rm(folder,{recursive:true,force:true});
 await media.preparePendingScanRemoval(owner,manifest.keep);await media.removePendingScanMedia(owner,manifest.keep);
 process.stdout.write(JSON.stringify({sharedReferenceRetainedDuringFinalRemoval:true,missingFolderRemovalSafe:true,freshProcess:true,containerRebase:true,referencedFileRetained:true,unreferencedDeletionIntentCompleted:true,partialCopyRemoved:true,unmarkedCaptureRetained:true,otherOwnerAndUnrelatedFilesRetained:true,idempotent:true,failedDeletionIntentRetainedAndRetried:true,accountChangeStopsDeletion:true})+'\n');
}else{
 const root=await mkdtemp(join(tmpdir(),'aiwardrobe-scan-journal-'));
 try{
  const documents=join(root,'old','Documents'),{media}=await load(documents);
  const source=join(root,'capture.jpg');await writeFile(source,'synthetic image copy payload');
  const keep=await media.stagePendingScanMedia(owner,pathToFileURL(source).href,'image');
  const drop=await media.stagePendingScanMedia(owner,pathToFileURL(source).href,'image');
  const unmarked=await media.stagePendingScanMedia(owner,pathToFileURL(source).href,'image');
  await media.preparePendingScanRemoval(owner,keep);await media.preparePendingScanRemoval(owner,drop);
  await media.reconcilePendingScanMedia(owner,()=>[]); // Active transactions cannot be swept between intent and queue acknowledgement.
  assert.ok(await exists(fileURLToPath(keep)));assert.ok(await exists(fileURLToPath(drop)));
  const folder=join(documents,'aiwardrobe-scan-queue',owner),partial='.33333333-3333-4333-8333-333333333333.pending';
  await writeFile(join(folder,partial),'incomplete');await writeFile(join(folder,'unrelated.txt'),'not managed');
  await mkdir(join(documents,'aiwardrobe-scan-queue',other),{recursive:true});await writeFile(join(documents,'aiwardrobe-scan-queue',other,'foreign.jpg'),'foreign');
  await writeFile(join(root,'manifest.json'),JSON.stringify({keep,drop,unmarked,partial}));
  await rename(join(root,'old'),join(root,'new'));
  const result=await promisify(execFile)(process.execPath,[fileURLToPath(import.meta.url),'--restart',root]);
  const report={passed:true,moduleSha256:createHash('sha256').update(await readFile('src/services/pendingScanMedia.ts')).digest('hex'),activeTransactionsProtected:true,...JSON.parse(result.stdout),scope:'Real OS files and fresh Node process through actual production reconciliation module; not native Expo/device or complete app crash recovery'};
  await writeFile('docs/reports/readiness-evidence/pending-scan-reconciliation.json',JSON.stringify(report,null,2)+'\n');process.stdout.write(JSON.stringify(report)+'\n');
 }finally{await rm(root,{recursive:true,force:true});}
}
