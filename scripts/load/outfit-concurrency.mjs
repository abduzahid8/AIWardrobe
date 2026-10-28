/** Bounded real outfit API burst. Parent credentials stay in memory/child environment. */
import {spawn} from 'node:child_process';
import {readFile,writeFile,open} from 'node:fs/promises';
import {randomUUID} from 'node:crypto';
const stages=[2,5,10];
const fullDeck=process.env.OUTFIT_BURST_FULL_DECK==='1';
const requestedLimit=fullDeck?5:1;
if(process.argv.includes('--plan')){console.log(JSON.stringify({stages,maxRequests:17,requestedOutfitsPerRequest:requestedLimit,displayedOutfitsPerRequest:fullDeck?3:1,maximumLiveFixtures:10,escalation:'Stop on any invalid outcome, missing AI success or failed cleanup',scope:'Short same-shape burst; not sustained throughput or10000DAUcertification'},null,2));process.exit(0);}
if(!process.env.SUPABASE_SERVICE_KEY)throw Error('Fixture credential unavailable');
const runId=randomUUID().replaceAll('-','').slice(0,12),reportName=`outfit-concurrency-${runId}.json`;
const folder=new URL('../../docs/reports/readiness-evidence/',import.meta.url);
const report={measuredAt:new Date().toISOString(),runId,scope:`Live NVIDIA/Edge/Auth/catalog; ${fullDeck?'three displayed Work/Old Money outfits (limit five)':'one Work/Old Money outfit'} per request with no budget/weather. Same controlled catalog pool. No native,sustained-throughput or10kDAUclaim.`,plannedStages:stages,stages:[]};
let cancelled=false;
const waitingChildren=new Set();
const cancel=()=>{cancelled=true;for(const child of waitingChildren)if(child.connected)child.send('cancel');};
process.once('SIGINT',cancel);process.once('SIGTERM',cancel);
const save=()=>writeFile(new URL(reportName,folder),JSON.stringify(report,null,2)+'\n');
for(const concurrency of stages){
 if(cancelled)break;
 const children=[],active=new Set();let maxInflight=0,started=0,ready=0,failedPreparation=false;
 const stage={concurrency,outcomes:[],maxClientInflight:0};report.stages.push(stage);await save();
 for(let index=0;index<concurrency;index++){
  if(cancelled){failedPreparation=true;break;}
  const filename=`outfit-burst-${runId}-${concurrency}-${index}.json`;
  const log=await open(`/tmp/aiwardrobe-outfit-burst-${runId}-${concurrency}-${index}.log`,'wx',0o600);
  const child=spawn(process.execPath,['scripts/validation/production-old-money-work.mjs'],{env:{...process.env,OLD_MONEY_REPORT:filename,OLD_MONEY_LIMIT:String(requestedLimit),OLD_MONEY_REQUIRE_DECK:fullDeck?'1':'0',OUTFIT_START_BARRIER:'1'},stdio:['ignore',log.fd,log.fd,'ipc']});await log.close();
  const item={child,index,filename,ready:false,exited:false};children.push(item);
  child.on('message',message=>{
   if(message?.type==='ready'){
    item.ready=true;ready++;waitingChildren.add(child);
    if(ready===concurrency&&!failedPreparation&&!cancelled){started=Date.now();for(const c of children){waitingChildren.delete(c.child);c.child.send('start');}}
    else if(failedPreparation||cancelled)child.send('cancel');
   }
   if(message?.type==='generation-start'){active.add(index);maxInflight=Math.max(maxInflight,active.size);}
   if(message?.type==='generation-end')active.delete(index);
  });
  item.done=new Promise(resolve=>{
   const finish=code=>{
    if(item.exited)return;
    item.exited=true;waitingChildren.delete(child);active.delete(index);item.code=code;
    if(!started){failedPreparation=true;for(const c of children)if(c.ready&&!c.exited&&c.child.connected)c.child.send('cancel');}
    resolve();
   };
   child.once('exit',finish);
   child.on('error',()=>{if(!child.pid)finish(null);});
  });
 }
 await Promise.all(children.map(c=>c.done));
 for(const item of children){
  try{
   const outcome=JSON.parse(await readFile(new URL(item.filename,folder),'utf8'));
   stage.outcomes.push({index:item.index,report:item.filename,fixtureUserId:outcome.fixtureUserId??null,exitCode:item.code,passed:outcome.passed===true,status:outcome.response?.status??null,source:outcome.response?.source??null,provider:outcome.response?.provider??null,latencyMs:outcome.response?.latencyMs??null,outfitCount:outcome.response?.outfitCount??0,visibleDeck:outcome.visibleDeck??null,providerAttempts:outcome.response?.providerAttempts??[],cleanup:outcome.cleanup??{},failure:outcome.failure??null});
  }catch{stage.outcomes.push({index:item.index,exitCode:item.code,passed:false,failure:'No readable child report'});}
 }
 stage.maxClientInflight=maxInflight;stage.generationWindowMs=started?Date.now()-started:null;
 stage.validResponses=stage.outcomes.filter(o=>o.passed).length;
 stage.aiResponses=stage.outcomes.filter(o=>o.passed&&o.source==='ai'&&o.provider?.startsWith('nvidia:')).length;
 stage.passed=!failedPreparation&&maxInflight===concurrency&&stage.validResponses===concurrency&&stage.aiResponses===concurrency;
 await save();console.log(JSON.stringify({stage:concurrency,maxInflight,valid:stage.validResponses,ai:stage.aiResponses,passed:stage.passed}));
 if(!stage.passed){report.stoppedEscalation='A stage failed; larger burst not started';process.exitCode=1;break;}
}
report.completedAt=new Date().toISOString();report.cancelled=cancelled;report.passed=!cancelled&&report.stages.length===stages.length&&report.stages.every(s=>s.passed);if(!report.passed)process.exitCode=1;await save();console.log(JSON.stringify({report:reportName,passed:report.passed}));
