/** Isolated Android emulator probe of actual Expo video JS/native APIs. No user accounts. */
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import jpeg from 'jpeg-js';
const run=promisify(execFile),root=await mkdtemp(join(tmpdir(),'aiwardrobe-android-video-'));
const sdk='/opt/homebrew/share/android-commandlinetools',adb=join(sdk,'platform-tools/adb');
const env={...process.env,ANDROID_HOME:sdk,ANDROID_AVD_HOME:join(root,'avds'),JAVA_HOME:'/tmp/aiwardrobe-jdk17/jdk-17.0.20.1+1/Contents/Home'};
const command=async(file,args,timeout=60000)=>(await run(file,args,{env,timeout,maxBuffer:4*1024*1024})).stdout;
const serial='emulator-5584', avd='AIWardrobeVideoNativeAudit';
let emulator,emulatorExit,deviceStarted=false,emulatorLog='';
const adbCommand=(args,timeout=30000)=>command(adb,['-s',serial,...args],timeout);
try {
  const devices=await command(adb,['devices']);assert.ok(!devices.includes(serial),'Audit emulator port already in use');
  await mkdir(env.ANDROID_AVD_HOME,{recursive:true});
  await command(join(sdk,'cmdline-tools/latest/bin/avdmanager'),['create','avd','-n',avd,'-k','system-images;android-34;google_apis;arm64-v8a','-d','pixel_7','-p',join(root,'device.avd')]);
  const configPath=join(root,'device.avd','config.ini');
  const config=await readFile(configPath,'utf8');
  await writeFile(configPath,config.replace(/^disk\.dataPartition\.size=.*$/m,'disk.dataPartition.size=1024M'));
  const assets=join(root,'assets');await mkdir(assets);
  await command('node',['node_modules/@expo/cli/build/bin/cli','export:embed','--entry-file','scripts/validation/native-video-sampling-app.tsx','--platform','android','--dev','false','--bytecode','--bundle-output',join(assets,'index.android.bundle'),'--assets-dest',join(root,'bundle-assets')],120000);
  const unsigned=join(root,'unsigned.apk'),aligned=join(root,'aligned.apk'),apk=join(root,'probe.apk');
  // Repackage a copy; the shipping release APK is never modified.
  await command('python3',['-c',`import zipfile,sys,re
src,dst,bundle=sys.argv[1:]
with zipfile.ZipFile(src) as old,zipfile.ZipFile(dst,'w') as new:
 for item in old.infolist():
  if item.filename=='assets/index.android.bundle' or re.match(r'META-INF/(?:.*\\.(?:SF|RSA|DSA|EC)|MANIFEST\\.MF)$',item.filename,re.I):continue
  new.writestr(item,old.read(item.filename))
 new.write(bundle,'assets/index.android.bundle')`,'android/app/build/outputs/apk/release/app-release.apk',unsigned,join(assets,'index.android.bundle')]);
  await command(join(sdk,'build-tools/36.0.0/zipalign'),['-P','16','-f','4',unsigned,aligned]);
  await command(join(sdk,'build-tools/36.0.0/apksigner'),['sign','--ks','android/app/debug.keystore','--ks-key-alias','androiddebugkey','--ks-pass','pass:android','--key-pass','pass:android','--out',apk,aligned]);
  emulator=spawn(join(sdk,'emulator/emulator'),['-avd',avd,'-port','5584','-no-window','-no-audio','-no-snapshot','-partition-size','1024','-gpu','swiftshader'],{env,stdio:['ignore','pipe','pipe']});
  emulatorExit=new Promise(resolve=>emulator.once('close',resolve));
  for(const stream of [emulator.stdout,emulator.stderr])stream.on('data',chunk=>{emulatorLog=(emulatorLog+chunk).slice(-12000);});
  deviceStarted=true;
  let connected=false;
  for(let i=0;i<90;i++){
    if(emulator.exitCode!==null)throw Error('Audit emulator exited before connecting');
    try { if((await adbCommand(['get-state'],3000)).trim()==='device'){connected=true;break;} } catch {}
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  assert.ok(connected,'Audit emulator connection timeout');
  let booted=false;
  for(let i=0;i<90;i++){
    if(emulator.exitCode!==null)throw Error('Audit emulator exited');
    if((await adbCommand(['shell','getprop','sys.boot_completed'])).trim()==='1'){booted=true;break;}
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  assert.ok(booted,'Audit Android emulator boot timeout');
  await adbCommand(['root']);await adbCommand(['wait-for-device']);
  await adbCommand(['install','-r',apk],60000);
  const documents='/data/user/0/com.aiwardrobe/files';
  const packageInfo=await adbCommand(['shell','cmd','package','list','packages','-U','com.aiwardrobe']);
  const uid=/package:com\.aiwardrobe\s+uid:(\d+)/.exec(packageInfo)?.[1];assert.ok(uid,'Missing audit package UID: '+packageInfo.trim());
  await adbCommand(['shell','mkdir','-p',documents]);
  const clip=join(root,'codec-audit.mp4');
  await command('ffmpeg',['-v','error','-f','lavfi','-i','color=c=0xFF0000:s=160x120:r=5:d=10','-f','lavfi','-i','color=c=0x00FF00:s=160x120:r=5:d=10','-f','lavfi','-i','color=c=0x0000FF:s=160x120:r=5:d=10','-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]','-map','[v]','-c:v','libx264','-pix_fmt','yuv420p',clip]);
  await adbCommand(['push',clip,documents+'/codec-audit.mp4']);
  await adbCommand(['shell','chown','-R',`${uid}:${uid}`,documents]);
  await adbCommand(['shell','restorecon','-R',documents]);
  await adbCommand(['shell','am','start','-n','com.aiwardrobe/.MainActivity']);
  let result;
  for(let i=0;i<90;i++){
    try { result=JSON.parse(await adbCommand(['shell','cat',documents+'/native-video-sampling-result.json']));break; }
    catch { await new Promise(resolve=>setTimeout(resolve,1000)); }
  }
  assert.ok(result,'Android native probe did not write a result');
  assert.equal(result.success,true);
  await adbCommand(['shell','am','force-stop','com.aiwardrobe']);
  await adbCommand(['shell','am','start','-n','com.aiwardrobe/.MainActivity']);
  let reviewRestart;
  for(let i=0;i<90;i++){
    try{reviewRestart=JSON.parse(await adbCommand(['shell','cat',documents+'/native-review-restart-result.json']));break;}
    catch{await new Promise(resolve=>setTimeout(resolve,1000));}
  }
  assert.ok(reviewRestart,'Android review restart did not finish');assert.equal(reviewRestart.success,true);assert.equal(reviewRestart.exactImages,true);assert.equal(reviewRestart.stableIDs,true);result.reviewRestart=reviewRestart;
  const evidence='docs/reports/readiness-evidence/android-native-video-sampling';await mkdir(evidence,{recursive:true});
  result.dominantChannels=[];
  for(const frame of result.frames??[]){
    await adbCommand(['pull',documents+'/'+frame.file,join(evidence,frame.file)]);
    const image=jpeg.decode(await readFile(join(evidence,frame.file))),sums=[0,0,0];
    for(let i=0;i<image.data.length;i+=4)for(let channel=0;channel<3;channel++)sums[channel]+=image.data[i+channel];
    result.dominantChannels.push(sums.indexOf(Math.max(...sums)));
  }
  result.scope='Actual production duration/exact-frame APIs and review file persistence across process restart; isolated Android API34 arm64 emulator, generated H264; no picker/provider/full app flow claim';
  await writeFile(join(evidence,'result.json'),JSON.stringify(result,null,2)+'\n');
  assert.equal(result.success,true);assert.deepEqual(result.times,[3000,15000,27000]);assert.deepEqual(result.repeatedTimes,result.times);assert.deepEqual(result.dominantChannels,[0,1,2]);
  process.stdout.write(JSON.stringify(result)+'\n');
}catch(error){await writeFile('/tmp/aiwardrobe-android-video-emulator.log',emulatorLog);throw error;}
finally{
  if(deviceStarted){await adbCommand(['emu','kill']).catch(()=>{});await Promise.race([emulatorExit,new Promise(resolve=>setTimeout(resolve,10000))]);if(emulator.exitCode===null)emulator.kill('SIGTERM');}
  await rm(root,{recursive:true,force:true});
}
