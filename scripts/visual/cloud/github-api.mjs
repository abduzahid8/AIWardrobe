/** Uses the existing OS credential helper; credentials never reach stdout. */
import cp from 'node:child_process';
const raw=cp.execFileSync('git',['credential-osxkeychain','get'],{input:'protocol=https\nhost=github.com\n\n',encoding:'utf8'});
const credentials=Object.fromEntries(raw.trim().split('\n').map(line=>{const i=line.indexOf('=');return [line.slice(0,i),line.slice(i+1)]}));
if(!credentials.password)throw Error('GitHub credential unavailable');
export async function github(path,init={}){
 const r=await fetch('https://api.github.com/repos/abduzahid8/AIWardrobe'+path,{...init,headers:{Authorization:'Bearer '+credentials.password,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28',...init.headers},signal:AbortSignal.timeout(30000)});
 if(!r.ok)throw Error('GitHub HTTP '+r.status+' '+path.split('?')[0]);return r;
}
