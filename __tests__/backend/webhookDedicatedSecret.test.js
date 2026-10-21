/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');
const source=fs.readFileSync(process.env.WEBHOOK_SECRET_SOURCE||path.resolve(__dirname,'../../backend/supabase/functions/revenuecat-webhook/index.ts'),'utf8');
function load(values){let handler;const createClient=jest.fn();new Function('require','exports','Deno',ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(name=>name.includes('server.ts')?{serve:fn=>handler=fn}:{createClient},{},{env:{get:name=>values[name]}});return {handler,createClient};}
const request=token=>new Request('https://fixture.invalid',{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:'{}'});
test('a public SDK key cannot authenticate when no dedicated webhook secret is configured',async()=>{
 const {handler,createClient}=load({REVENUECAT_API_KEY:'appl_public-fixture'});const response=await handler(request('appl_public-fixture'));expect(response.status).toBe(503);expect(createClient).not.toHaveBeenCalled();
});
test.each(['appl_public-fixture','goog_public-fixture','test_public-fixture'])('SDK key %s is not a valid dedicated webhook secret',async key=>{
 const {handler,createClient}=load({REVENUECAT_API_KEY:'server-api-fixture',REVENUECAT_WEBHOOK_SECRET:key});const response=await handler(request(key));expect(response.status).toBe(503);expect(createClient).not.toHaveBeenCalled();
});
test('reusing the API credential as webhook secret is rejected',async()=>{
 const {handler,createClient}=load({REVENUECAT_API_KEY:'same-fixture',REVENUECAT_WEBHOOK_SECRET:'same-fixture'});const response=await handler(request('same-fixture'));expect(response.status).toBe(503);expect(createClient).not.toHaveBeenCalled();
});
test('a valid dedicated secret reaches payload validation, not a DB write for an empty body',async()=>{
 const {handler,createClient}=load({REVENUECAT_API_KEY:'api-fixture',REVENUECAT_WEBHOOK_SECRET:'dedicated-secret-fixture'});const response=await handler(request('dedicated-secret-fixture'));expect(response.status).toBe(400);expect(createClient).not.toHaveBeenCalled();
});
