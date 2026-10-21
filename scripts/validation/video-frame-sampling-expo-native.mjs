/** Real Expo JS/native-bridge probe in an isolated simulator with generated video only. */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, writeFile, cp, copyFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import jpeg from 'jpeg-js';
const run=promisify(execFile),root=await mkdtemp(join(tmpdir(),'aiwardrobe-expo-video-'));
const command=async(file,args,timeout=60000)=>(await run(file,args,{timeout,maxBuffer:4*1024*1024})).stdout;
let device;
try {
  const app=join(root,'AIWardrobe.app');
  await cp('/tmp/aiwardrobe-budget-release-00nhnjcc/AIWardrobe.app',app,{recursive:true});
  await command('node',['node_modules/@expo/cli/build/bin/cli','export:embed','--entry-file','scripts/validation/native-video-sampling-app.tsx','--platform','ios','--dev','false','--bytecode','--bundle-output',join(app,'main.jsbundle'),'--assets-dest',app],120000);
  const runtimes=JSON.parse(await command('xcrun',['simctl','list','runtimes','-j']));
  const runtime=runtimes.runtimes.find(item=>item.isAvailable&&item.name==='iOS 26.5');assert.ok(runtime);
  device=(await command('xcrun',['simctl','create','AIWardrobe-Expo-Video-Audit','com.apple.CoreSimulator.SimDeviceType.iPhone-16-Pro',runtime.identifier])).trim();assert.match(device,/^[A-F0-9-]{36}$/);
  await command('xcrun',['simctl','boot',device]);await command('xcrun',['simctl','bootstatus',device,'-b'],120000);
  await command('xcrun',['simctl','install',device,app]);
  const container=(await command('xcrun',['simctl','get_app_container',device,'com.aiwardrobe','data'])).trim();
  const documents=join(container,'Documents');await mkdir(documents,{recursive:true});
  await command('ffmpeg',['-v','error','-f','lavfi','-i','color=c=0xFF0000:s=160x120:r=5:d=10','-f','lavfi','-i','color=c=0x00FF00:s=160x120:r=5:d=10','-f','lavfi','-i','color=c=0x0000FF:s=160x120:r=5:d=10','-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]','-map','[v]','-c:v','libx264','-pix_fmt','yuv420p',join(documents,'codec-audit.mp4')]);
  await command('xcrun',['simctl','launch',device,'com.aiwardrobe']);
  let result;
  for(let attempt=0;attempt<90;attempt++){
    try { result=JSON.parse(await readFile(join(documents,'native-video-sampling-result.json'),'utf8'));break; }
    catch(error){if(error.code!=='ENOENT')throw error;await new Promise(resolve=>setTimeout(resolve,1000));}
  }
  assert.ok(result,'Native probe did not finish within90seconds');
  assert.equal(result.success,true);
  await command('xcrun',['simctl','terminate',device,'com.aiwardrobe']);
  await command('xcrun',['simctl','launch',device,'com.aiwardrobe']);
  let reviewRestart;
  for(let i=0;i<90;i++){
    try{reviewRestart=JSON.parse(await readFile(join(documents,'native-review-restart-result.json'),'utf8'));break;}
    catch(error){if(error.code!=='ENOENT')throw error;await new Promise(resolve=>setTimeout(resolve,1000));}
  }
  assert.ok(reviewRestart,'Native review restart did not finish');assert.equal(reviewRestart.success,true);assert.equal(reviewRestart.exactImages,true);assert.equal(reviewRestart.stableIDs,true);result.reviewRestart=reviewRestart;
  const evidence='docs/reports/readiness-evidence/expo-native-video-sampling';await mkdir(evidence,{recursive:true});
  for(const frame of result.frames??[])await copyFile(join(documents,frame.file),join(evidence,frame.file));
  if(result.success){
    result.dominantChannels=[];
    for(const frame of result.frames){
      const image=jpeg.decode(await readFile(join(evidence,frame.file)));
      const sums=[0,0,0];
      for(let i=0;i<image.data.length;i+=4)for(let channel=0;channel<3;channel++)sums[channel]+=image.data[i+channel];
      result.dominantChannels.push(sums.indexOf(Math.max(...sums)));
    }
    assert.deepEqual(result.dominantChannels,[0,1,2]);
    result.allThreeScenesVerified=true;
  }
  result.scope='Actual production readVideoSampleTimes + ExpoVideo/ExpoVideoThumbnails JS/native bridge in isolated iOS26.5 app; generated H264 clip; no picker, provider or complete shipping-app flow claim';
  await writeFile(join(evidence,'result.json'),JSON.stringify(result,null,2)+'\n');
  assert.equal(result.success,true);assert.deepEqual(result.times,[3000,15000,27000]);assert.deepEqual(result.repeatedTimes,result.times);
  assert.equal(result.frames.length,3);assert.ok(result.frames.every(frame=>frame.base64Length>0&&frame.width===160&&frame.height===120));
  process.stdout.write(JSON.stringify(result)+'\n');
}finally{
  if(device){await command('xcrun',['simctl','shutdown',device]).catch(()=>{});await command('xcrun',['simctl','delete',device]);}
  await rm(root,{recursive:true,force:true});
}
