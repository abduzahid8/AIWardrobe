/** Actual OS filesystem/process-restart check through an Expo-shaped adapter.
 * This is not an on-device Expo FileSystem test. No network/provider calls. */
import { readFile, writeFile, mkdir, stat, rename, rm, mkdtemp, truncate } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { encode, decode } from 'base64-arraybuffer';
import { transpileModule, ModuleKind } from 'typescript';
import assert from 'node:assert/strict';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const owner = '11111111-1111-4111-8111-111111111111', imageId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const exists = async path => { try { await stat(path); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
async function loadService(documents) {
  const compile = async path => transpileModule(await readFile(new URL(path, import.meta.url),'utf8'), { compilerOptions: { module: ModuleKind.CommonJS } }).outputText;
  const events = {}, media = {}, pending = {};
  new Function('exports',await compile('../../store/authEvents.ts'))(events);
  const mediaCode = await compile('../../src/services/wardrobeMedia.ts');
  new Function('exports','require',mediaCode)(media,name => {
    if(name==='base64-arraybuffer')return {encode,decode};
    if(name==='../../lib/supabase')return {supabase:{}};
    if(name==='../../store/authEvents')return events;
    if(name==='../config/env')return {Config:{supabase:{url:'https://storage.test'}}};
    throw new Error('Unexpected media dependency');
  });
  const fs = {
    documentDirectory:pathToFileURL(documents).href+'/', EncodingType:{Base64:'base64'},
    makeDirectoryAsync:uri=>mkdir(fileURLToPath(uri),{recursive:true}),
    getInfoAsync:async uri=>{try{const info=await stat(fileURLToPath(uri));return {exists:true,isDirectory:info.isDirectory(),size:info.size};}catch(error){if(error.code==='ENOENT')return {exists:false};throw error;}},
    readAsStringAsync:async uri=>(await readFile(fileURLToPath(uri))).toString('base64'),
    writeAsStringAsync:(uri,value)=>writeFile(fileURLToPath(uri),Buffer.from(value,'base64'),{mode:0o600}),
    moveAsync:({from,to})=>rename(fileURLToPath(from),fileURLToPath(to)),
    deleteAsync:uri=>rm(fileURLToPath(uri),{recursive:true,force:true}),
  };
  const pendingCode=await compile('../../src/services/pendingWardrobeMedia.ts');
  new Function('exports','require',pendingCode)(pending,name=>{
    if(name==='expo-file-system/legacy')return fs;
    if(name==='base64-arraybuffer')return {encode,decode};
    if(name==='../../store/authEvents')return events;
    if(name==='./wardrobeMedia')return media;
    throw new Error('Unexpected pending-media dependency');
  });
  globalThis.fetch=async uri=>{
    assert.equal(new URL(uri).protocol,'file:','Durability test cannot make network requests');
    return new Response(await readFile(fileURLToPath(uri)),{headers:{'content-type':'image/png'}});
  };
  events.authEvents.emitLogin(owner);
  return {pending,codeSha256:hash(pendingCode)};
}
if(process.argv[2]==='--restore'){
  const root=process.argv[3],manifest=JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
  const {pending}=await loadService(join(root,'restored','Documents'));
  assert.equal(await exists(fileURLToPath(manifest.originalPickerUri)),false);
  assert.equal(await exists(fileURLToPath(manifest.stagedUri)),false);
  const restored=pending.resolvePendingWardrobeImage(manifest.stagedUri,owner);
  assert.ok(restored.includes('/restored/Documents/'));
  const bytes=await readFile(fileURLToPath(restored));assert.equal(hash(bytes),manifest.sha256);
  assert.equal(await pending.stageNativeWardrobeImage(owner,imageId,manifest.stagedUri),restored);
  assert.equal(pending.resolvePendingWardrobeImage(manifest.stagedUri,'22222222-2222-4222-8222-222222222222'),undefined);
  await pending.removePendingWardrobeImage(owner,manifest.stagedUri);assert.equal(await exists(fileURLToPath(restored)),false);
  process.stdout.write(JSON.stringify({bytes:bytes.length,originalPickerEvicted:true,oldContainerAbsent:true,restoredByteHashMatches:true,retryAfterProcessRestart:true,otherOwnerBlocked:true,managedCleanupPassed:true})+'\n');
}else{
  const root=await mkdtemp(join(tmpdir(),'aiwardrobe-pending-media-check-'));
  const report={scope:'Actual source logic and OS file rename/read across a new Node process; Expo-shaped filesystem adapter, not device runtime',passed:false};
  try{
    const picker=join(root,'picker');await mkdir(picker);
    const bytes=await readFile(new URL('../../assets/images/basic_outfit_1.png',import.meta.url));
    const originalPickerUri=pathToFileURL(join(picker,'original.png')).href;
    await writeFile(fileURLToPath(originalPickerUri),bytes,{mode:0o600});
    const {pending,codeSha256}=await loadService(join(root,'first','Documents'));
    const excessive = join(picker,'oversized.png');
    await writeFile(excessive,'');await truncate(excessive,64*1024*1024);
    const normalFetch=globalThis.fetch;
    let attemptedRead=false;
    globalThis.fetch=async()=>{attemptedRead=true;throw new Error('Oversized source must not be read');};
    try {
      await assert.rejects(pending.stageNativeWardrobeImage(owner,imageId,pathToFileURL(excessive).href),error=>error.code==='invalid_size');
      assert.equal(attemptedRead,false);
      report.oversizedPreflight={logicalBytes:(await stat(excessive)).size,bodyReadAttempted:false,rejected:true};
    } finally {globalThis.fetch=normalFetch;}
    let activeReads=0,peakReads=0,readCalls=0;
    globalThis.fetch=async uri=>{
      activeReads++;readCalls++;peakReads=Math.max(peakReads,activeReads);
      try{
        const response=await normalFetch(uri);
        return {headers:response.headers,arrayBuffer:async()=>{
          try{await new Promise(resolve=>setTimeout(resolve,5));return await response.arrayBuffer();}
          finally{activeReads--;}
        }};
      }catch(error){activeReads--;throw error;}
    };
    try{
      const ids=Array.from({length:8},(_,index)=>`b0000000-0000-4000-8000-${index.toString(16).padStart(12,'0')}`);
      const results=await Promise.allSettled(ids.map(id=>pending.stageNativeWardrobeImage(owner,id,originalPickerUri)));
      assert.ok(results.every(result=>result.status==='fulfilled'));
      const uris=results.map(result=>result.value);
      for(const uri of uris)assert.equal(hash(await readFile(fileURLToPath(uri))),hash(bytes));
      assert.equal(hash(await readFile(fileURLToPath(originalPickerUri))),hash(bytes));
      assert.equal(peakReads,1);assert.equal(readCalls,8);assert.equal(activeReads,0);
      await Promise.all(uris.map(uri=>pending.removePendingWardrobeImage(owner,uri)));
      report.concurrentStaging={submitted:8,completed:8,fixtureBytes:bytes.length,totalCommittedBytes:bytes.length*8,peakBodyReads:peakReads,readCalls,unchangedSource:true,allCommittedHashesMatch:true,scope:'Actual service through Node OS-file adapter; five-ms read delay exposes overlap, not native RSS profiling'};
    }finally{globalThis.fetch=normalFetch;}
    const stagedUri=await pending.stageNativeWardrobeImage(owner,imageId,originalPickerUri);
    assert.equal(hash(await readFile(fileURLToPath(stagedUri))),hash(bytes));
    await writeFile(join(root,'manifest.json'),JSON.stringify({originalPickerUri,stagedUri,sha256:hash(bytes)}),{mode:0o600});
    await rm(picker,{recursive:true,force:true});await rename(join(root,'first'),join(root,'restored'));
    const {stdout}=await promisify(execFile)(process.execPath,[fileURLToPath(import.meta.url),'--restore',root],{timeout:30000});
    report.sourceJavaScriptSha256=codeSha256;report.restoration=JSON.parse(stdout);report.passed=true;
  }catch(error){report.failure=error.name;process.exitCode=1;}
  finally{await rm(root,{recursive:true,force:true});report.temporaryFilesRemoved=true;
    await writeFile(new URL('../../docs/reports/readiness-evidence/pending-media-os-durability.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
    process.stdout.write(JSON.stringify(report)+'\n');}
}
