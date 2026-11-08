import {serve} from 'https://deno.land/std@0.168.0/http/server.ts';
import {createClient} from 'https://esm.sh/@supabase/supabase-js@2.95.3';
const headers={'Content-Type':'application/json','Cache-Control':'no-store'};
const reply=(status:number,body:unknown)=>new Response(JSON.stringify(body),{status,headers});
async function sameToken(a:string,b:string){
 const [x,y]=await Promise.all([a,b].map(s=>crypto.subtle.digest('SHA-256',new TextEncoder().encode(s))));
 const left=new Uint8Array(x),right=new Uint8Array(y);let difference=0;
 for(let i=0;i<left.length;i++)difference|=left[i]^right[i];return difference===0;
}
serve(async request=>{
 const expected=Deno.env.get('STYLIST_VISUAL_WORKER_TOKEN');
 if(!expected)return reply(503,{error:'worker_not_configured'});
 const supplied=(request.headers.get('Authorization')||'').replace(/^Bearer\s+/i,'');
 if(supplied.length>128||!await sameToken(supplied,expected))return reply(401,{error:'worker_auth_required'});
 if(request.method!=='POST')return reply(405,{error:'method_not_allowed'});
 try{
  const reader=request.body?.getReader();if(!reader)return reply(400,{error:'invalid_worker_payload'});
  const parts:Uint8Array[]=[];let bytes=0;
  try{for(;;){const chunk=await reader.read();if(chunk.done)break;bytes+=chunk.value.length;if(bytes>524288)return reply(413,{error:'worker_payload_too_large'});parts.push(chunk.value)}}finally{await reader.cancel().catch(()=>{});reader.releaseLock()}
  const all=new Uint8Array(bytes);let offset=0;for(const part of parts){all.set(part,offset);offset+=part.length}
  const body=JSON.parse(new TextDecoder().decode(all));
  if(!['seed','lease','complete','state'].includes(body.action)||!/^[-a-f0-9]{36}$/.test(body.worker||''))return reply(400,{error:'invalid_worker_payload'});
  const admin=createClient(Deno.env.get('SUPABASE_URL')!,Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,{auth:{persistSession:false,autoRefreshToken:false}});
  const {data,error}=await admin.rpc('catalog_visual_batch',{p_action:body.action,p_worker:body.worker,p_payload:body.payload||{}}).abortSignal(AbortSignal.timeout(20000));
  if(error)return reply(error.code==='22023'?400:503,{error:'worker_operation_failed',code:error.code});
  return reply(200,{data});
 }catch{return reply(400,{error:'invalid_worker_payload'})}
});
