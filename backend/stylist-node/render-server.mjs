/** Free-plan Render fallback for isolated Stylist 6 staging.
 * Reuses the exact Vercel HMAC/JWT/quota guarded POST() handler. No S5
 * routes, secrets, production migrations, nor unsigned generation API.
 */
import { createServer } from 'node:http';
import { POST } from './dist/stylist6-http-route.mjs';
import { STYLIST_TRANSPORT_MAX_BODY_BYTES } from './transport/signature.mjs';

export const RENDER_STYLIST6_ROUTE='/api/stylist6-generate-outfits';
const HEALTH_ROUTE='/health';

function json(res,status,code){
  res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});
  res.end(JSON.stringify({success:false,outfits:[],code,error:code,generationVersion:'stylist-6-cloud-staging-1'}));
}
export function createStylist6RenderServer(post=POST){
  return createServer(async(req,res)=>{
    const route=(req.url||'').split('?')[0];
    if(req.method==='GET' && route===HEALTH_ROUTE){
      res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});
      res.end(JSON.stringify({ok:true,engine:'stylist-6-render-staging',
        computeConfigured:Boolean(process.env.STYLIST6_NODE_SHARED_SECRET &&
          process.env.SUPABASE_URL && process.env.SUPABASE_ANON_KEY &&
          process.env.SUPABASE_SERVICE_ROLE_KEY && process.env.NVIDIA_API_KEY)}));
      return;
    }
    if(route!==RENDER_STYLIST6_ROUTE){json(res,404,'not_found');return;}
    if(req.method!=='POST'){json(res,405,'method_not_allowed');return;}
    const expected=Number(req.headers['content-length']||0);
    if(expected>STYLIST_TRANSPORT_MAX_BODY_BYTES){json(res,413,'invalid_generation_request');return;}
    try{
      const chunks=[];let size=0;
      for await(const part of req){
        size+=part.length;
        if(size>STYLIST_TRANSPORT_MAX_BODY_BYTES){json(res,413,'invalid_generation_request');return;}
        chunks.push(part);
      }
      const body=Buffer.concat(chunks,size);
      const headers=new Headers();
      for(const [key,val] of Object.entries(req.headers))if(val!==undefined)
        headers.set(key,Array.isArray(val)?val.join(', '):String(val));
      // In-process trusted Render entrypoint only. The user cannot choose
      // this budget via headers or body, and the HMAC/JWT handler still owns
      // verification and billing. The Vercel route keeps the default 50s.
      const response=await post(new Request('http://localhost'+RENDER_STYLIST6_ROUTE,
        {method:'POST',headers,body}),{renderCanary:true});
      const data=Buffer.from(await response.arrayBuffer());
      const contentType=response.headers.get('content-type')||'application/json';
      const outHeaders={'Content-Type':contentType,'Cache-Control':'no-store'};
      const timing=response.headers.get('server-timing');
      if(timing)outHeaders['Server-Timing']=timing;
      res.writeHead(response.status,outHeaders);res.end(data);
    }catch{
      if(!res.headersSent)json(res,503,'stylist_compute_unavailable');
      else res.destroy();
    }
  });
}
if(process.argv[1]?.endsWith('/render-server.mjs')){
  const port=Number(process.env.PORT||10000);
  if(!Number.isInteger(port)||port<1||port>65535)throw Error('invalid_port');
  // Private Render service logs: missing variable NAMES only, no values.
  // The public /health endpoint intentionally exposes only one boolean.
  const required=['STYLIST6_NODE_SHARED_SECRET','SUPABASE_URL','SUPABASE_ANON_KEY',
    'SUPABASE_SERVICE_ROLE_KEY','NVIDIA_API_KEY'];
  const missing=required.filter(name=>!process.env[name]);
  if(missing.length)console.warn('[stylist6-render] missing_environment_names:',missing.join(','));
  createStylist6RenderServer().listen(port,'0.0.0.0',()=>{
    console.log('[stylist6-render] ready '+port);
  });
}
