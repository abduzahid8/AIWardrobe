/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(process.env.NVIDIA_CHAT_SOURCE || path.resolve(__dirname, '../../backend/supabase/functions/ai-chat/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
function load({denial = null, key = 'fixture-key', status = 200, text = 'Pair the navy shirt with beige trousers and brown leather shoes.'} = {}) {
  let handler;
  const admission=jest.fn(async()=>denial);
  const fetchMock = jest.fn(async () => new Response(JSON.stringify({choices:[{message:{content:text}}]}), {status}));
  const rows = [{key:'nvidia_token',value:key}];
  const chain = {select:()=>chain,in:async()=>({data:rows}),eq:()=>chain,order:()=>chain,limit:async()=>({data:[{tier:'premium',status:'active'}]})};
  const client = {auth:{getUser:async()=>({data:{user:{id:'fixture-owner'}}})},from:()=>chain};
  new Function('require','exports','Deno','fetch',compiled)(name=> {
    if(name.includes('rateLimit')) return {enforceRateLimit:admission};
    if(name.includes('server.ts')) return {serve:fn=>{handler=fn;}};
    return {createClient:()=>client};
  },{}, {env:{get:name=>name.startsWith('SUPABASE_')?'fixture-config':undefined}},fetchMock);
  return {handler,fetchMock,admission};
}
const request=()=>new Request('https://example.test',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify({query:'Style a navy shirt and beige trousers.'})});
test('paid users use the server-stored NVIDIA key and receive its actual response',async()=>{
 const {handler,fetchMock}=load();const response=await handler(request());
 expect(response.status).toBe(200);expect(await response.json()).toMatchObject({provider:'nvidia',text:expect.stringContaining('navy')});
 expect(fetchMock).toHaveBeenCalledTimes(1);expect(fetchMock.mock.calls[0][0]).toBe('https://integrate.api.nvidia.com/v1/chat/completions');
 expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe('Bearer fixture-key');
});
test.each([429,503])('NVIDIA HTTP %s produces an unavailable result without a fabricated answer',async status=>{
 const {handler,fetchMock}=load({status});const response=await handler(request());
 expect(response.status).toBe(503);expect((await response.json()).text).toBeUndefined();expect(fetchMock).toHaveBeenCalledTimes(1);
});
test('missing NVIDIA configuration makes no provider request',async()=>{
 const {handler,fetchMock}=load({key:''});expect((await handler(request())).status).toBe(503);expect(fetchMock).not.toHaveBeenCalled();
});

test.each([403,429,503])('chat admission denial %s prevents provider work',async status=>{
 const {handler,fetchMock,admission}=load({denial:new Response('{"error":"denied"}',{status})});
 const response=await handler(request());expect(response.status).toBe(status);
 expect(admission).toHaveBeenCalledWith(expect.any(Request),expect.objectContaining({feature:'ai-chat',limit:20}));
 expect(fetchMock).not.toHaveBeenCalled();
});
