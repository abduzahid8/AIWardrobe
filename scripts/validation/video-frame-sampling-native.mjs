/** Actual iOS-simulator AVFoundation codec probe, plus production TS sampling function.
 * This does not execute the Expo JS/native bridge or the app's picker/provider flow.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import { transpileModule, ModuleKind } from 'typescript';
const run = promisify(execFile), root = await mkdtemp(join(tmpdir(),'aiwardrobe-video-sampling-'));
let device;
const command = async (file,args,timeout=60000) => (await run(file,args,{timeout,maxBuffer:2*1024*1024})).stdout;
try {
  const runtimeList = JSON.parse(await command('xcrun',['simctl','list','runtimes','-j']));
  const runtime = runtimeList.runtimes.find(item=>item.isAvailable && item.name==='iOS 26.5');
  assert.ok(runtime,'Required iOS26.5 runtime unavailable');
  device = (await command('xcrun',['simctl','create','AIWardrobe-Video-Codec-Audit','com.apple.CoreSimulator.SimDeviceType.iPhone-16-Pro',runtime.identifier])).trim();
  assert.match(device,/^[A-F0-9-]{36}$/);
  await command('xcrun',['simctl','boot',device]);
  await command('xcrun',['simctl','bootstatus',device,'-b'],120000);
  const sdk=(await command('xcrun',['--sdk','iphonesimulator','--show-sdk-path'])).trim();
  const binary=join(root,'probe');
  await command('xcrun',['--sdk','iphonesimulator','swiftc','-parse-as-library','-target','arm64-apple-ios18.0-simulator','-sdk',sdk,'scripts/validation/video-frame-sampling-native.swift','-o',binary],120000);
  const clip=join(root,'three-scenes.mp4');
  await command('ffmpeg',['-v','error','-f','lavfi','-i','color=c=0xFF0000:s=160x120:r=5:d=10','-f','lavfi','-i','color=c=0x00FF00:s=160x120:r=5:d=10','-f','lavfi','-i','color=c=0x0000FF:s=160x120:r=5:d=10','-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]','-map','[v]','-c:v','libx264','-pix_fmt','yuv420p',clip]);
  const probe=async times=>JSON.parse(await command('xcrun',['simctl','spawn',device,binary,clip,...(times?[times.join(',')]:[])]));
  const metadata=await probe();
  const source=await readFile('src/services/videoFrameSampling.ts','utf8');
  const exports={};
  new Function('exports','require',transpileModule(source,{compilerOptions:{module:ModuleKind.CommonJS}}).outputText)(exports,()=>({}));
  const times=exports.videoSampleTimes(metadata.durationSeconds);
  const current=await probe(times), previous=await probe([0,1000,2000]);
  const dominant=rgb=>rgb.indexOf(Math.max(...rgb));
  assert.deepEqual(times,[3000,15000,27000]);
  assert.deepEqual(current.samples.map(sample=>dominant(sample.rgb)),[0,1,2]);
  assert.deepEqual(previous.samples.map(sample=>dominant(sample.rgb)),[0,0,0]);
  const report={passed:true,runtime:runtime.name,scope:'Actual iOS AVFoundation duration and exact-time H264 frame decoding with production TS time selection; no Expo bridge or provider/UI claim',metadata,current,previous,lateScenesObserved:true};
  await writeFile('docs/reports/readiness-evidence/video-frame-sampling-native.json',JSON.stringify(report,null,2)+'\n');
  process.stdout.write(JSON.stringify(report)+'\n');
} finally {
  if(device){await command('xcrun',['simctl','shutdown',device]).catch(()=>{});await command('xcrun',['simctl','delete',device]);}
  await rm(root,{recursive:true,force:true});
}
