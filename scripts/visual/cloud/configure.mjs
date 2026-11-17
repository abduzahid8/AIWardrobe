import fs from 'node:fs';import crypto from 'node:crypto';import cp from 'node:child_process';
import {management} from '../../stylist5/private-api.mjs';import {github} from './github-api.mjs';
const dir=process.env.STYLIST5_PRIVATE_DIR;if(!dir)throw Error('private release directory required');
const file=dir+'/worker-token';let token;
if(fs.existsSync(file))token=fs.readFileSync(file,'utf8');else{token='svw1.'+crypto.randomBytes(32).toString('hex');fs.writeFileSync(file,token,{mode:0o600})}
await management('/secrets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify([{name:'STYLIST_VISUAL_WORKER_TOKEN',value:token}])});
const key=await(await github('/actions/secrets/public-key')).json();
for(const [name,value] of [['STYLIST_VISUAL_WORKER_TOKEN',token],['STYLIST_VISUAL_WORKER_URL','https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/catalog-visual-worker']]){
 const encrypted=cp.execFileSync(process.env.STYLIST5_PYTHON||'python3',['-c',"import sys,json,base64;from nacl.public import SealedBox,PublicKey;p=json.load(sys.stdin);print(base64.b64encode(SealedBox(PublicKey(base64.b64decode(p['key']))).encrypt(p['value'].encode())).decode())"],{input:JSON.stringify({key:key.key,value}),encoding:'utf8'}).trim();
 await github('/actions/secrets/'+name,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({encrypted_value:encrypted,key_id:key.key_id})});
}
console.log({scopedWorkerSecretConfigured:true,githubWorkerSecretsConfigured:true,supabaseServiceKeyExported:false});
