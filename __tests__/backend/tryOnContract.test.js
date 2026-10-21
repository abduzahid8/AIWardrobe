/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(process.env.TRY_ON_SOURCE || path.resolve(__dirname, '../../backend/supabase/functions/try-on/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;

function loadHandler(denied = null, {authenticated=true,providerConfigured=false}={}) {
  let handler;
  const mockRateLimit = jest.fn().mockResolvedValue(denied);
  const mockFrom=jest.fn(()=>({select:()=>({eq:(_,key)=>({single:async()=>({data:providerConfigured&&key==='nvidia_token'?{value:'test-provider-key'}:null})})})}));
  const mockComposite=jest.fn(()=> 'data:image/png;base64,Q09NUE9TSVRF');
  const mockClient = jest.fn(() => ({
    auth: { getUser: async () => ({ data: { user: authenticated?{ id: 'test-user' }:null } }) },
    from: mockFrom,
  }));
  const mockFetch = jest.fn(async()=>new Response(JSON.stringify({artifacts:[{base64:'generated-image'}]}),{status:200}));
  new Function('require', 'exports', 'Deno', 'fetch', compiled)(name => {
    if (name.includes('tryOnComposite')) return { composeTryOnImages: mockComposite };
    if (name.includes('rateLimit')) return { enforceRateLimit: mockRateLimit };
    if (name.includes('server.ts')) return { serve: fn => { handler = fn; } };
    if (name.includes('supabase-js')) return { createClient: mockClient };
    throw new Error(`Unexpected import ${name}`);
  }, {}, { env: { get: () => 'test-configuration' } }, mockFetch);
  return { handler, mockClient, mockFetch, mockRateLimit, mockFrom, mockComposite };
}

test('missing provider configuration is 503 and never a successful garment-image mock', async () => {
  const { handler, mockFetch } = loadHandler();
  const result = await handler(new Request('https://example.test', { method: 'POST', body: JSON.stringify({ person_image: 'person', garment_image: 'garment' }) }));
  expect(result.status).toBe(503);
  const body = await result.json();
  expect(body.success).toBe(false);
  expect(body.resultImage).toBeUndefined();
  expect(mockFetch).not.toHaveBeenCalled();
});

test('denied admission stops before config, body processing, or paid work', async () => {
  const denied = new Response('{}', { status: 429 });
  const { handler, mockClient, mockFetch } = loadHandler(denied);
  expect(await handler(new Request('https://example.test', { method: 'POST' }))).toBe(denied);
  expect(mockClient).not.toHaveBeenCalled();
  expect(mockFetch).not.toHaveBeenCalled();
});

test('CORS preflight does not consume quota', async () => {
  const { handler, mockRateLimit } = loadHandler();
  expect((await handler(new Request('https://example.test', { method: 'OPTIONS' }))).status).toBe(200);
  expect(mockRateLimit).not.toHaveBeenCalled();
});

const request=body=>new Request('https://example.test',{method:'POST',body:JSON.stringify(body)});
test.each([401,403,429,503])('denial %s prevents all config, composition and provider work',async status=>{
 const denied=new Response('{}',{status});const {handler,mockFrom,mockFetch,mockComposite}=loadHandler(denied);expect(await handler(request({person_image:'person',garment_image:'garment'}))).toBe(denied);expect(mockFrom).not.toHaveBeenCalled();expect(mockFetch).not.toHaveBeenCalled();expect(mockComposite).not.toHaveBeenCalled();
});
test.each([{person_image:null,garment_image:'garment'},{person_image:{},garment_image:'garment'},{person_image:'',garment_image:'garment'},{person_image:'a'.repeat(14*1024*1024+1),garment_image:'garment'},{person_image:'person',garment_image:'garment',garment_type:'unsupported'}])('invalid input is400 before configuration/provider reads',async body=>{
 const {handler,mockFrom,mockFetch}=loadHandler();expect((await handler(request(body))).status).toBe(400);expect(mockFrom).not.toHaveBeenCalled();expect(mockFetch).not.toHaveBeenCalled();
});
test('wrong method is405 without consuming admission',async()=>{
 const {handler,mockRateLimit}=loadHandler();expect((await handler(new Request('https://example.test'))).status).toBe(405);expect(mockRateLimit).not.toHaveBeenCalled();
});
test('provider input includes BOTH references and retains documented NVIDIA endpoint and parameters',async()=>{
 const {handler,mockFetch,mockComposite,mockRateLimit}=loadHandler(null,{providerConfigured:true});const response=await handler(request({person_image:'person',garment_image:'garment'}));expect(response.status).toBe(200);expect(await response.json()).toMatchObject({success:true,methodUsed:'nvidia_kontext'});expect(mockRateLimit).toHaveBeenCalledWith(expect.any(Request),expect.objectContaining({feature:'try-on',limit:5}));expect(mockComposite).toHaveBeenCalledWith('person','garment');expect(mockFetch.mock.calls[0][0]).toBe('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev');expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toMatchObject({image:'data:image/png;base64,Q09NUE9TSVRF',cfg_scale:4,steps:30,samples:1});expect(mockFetch.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
});
