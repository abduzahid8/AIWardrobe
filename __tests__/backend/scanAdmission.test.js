/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');
const entry=process.env.SCAN_ADMISSION_SOURCE||path.resolve(__dirname,'../../backend/supabase/functions/ai-process/index.ts');
function load(denial=null){
 let handler;
 const admission=jest.fn(async()=>denial);
 const readConfig=jest.fn(async()=>({data:{value:null,diagnostic:'sensitive-fixture'},error:null}));
 const config={select:()=>config,eq:()=>config,single:readConfig};
 const createClient=jest.fn(()=>({from:()=>config}));
 const logs=[];
 const compiled=ts.transpileModule(fs.readFileSync(entry,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 new Function('require','exports','Deno','console',compiled)(name=>{
  if(name.includes('server.ts'))return {serve:fn=>handler=fn};
  if(name.includes('rateLimit'))return {enforceRateLimit:admission};
  if(name.includes('supabase-js'))return {createClient};
  if(name==='node:buffer')return require(name);
  return {};
 },{}, {env:{get:()=> 'fixture-config'}},{log:(...args)=>logs.push(args),error:()=>{},warn:()=>{}});
 return {handler,admission,readConfig,logs};
}
const req=body=>new Request('https://fixture.invalid',{method:'POST',headers:{Authorization:'Bearer fixture'},body:JSON.stringify(body)});
test.each([403,429,503])('scan admission denial %s cannot reach configuration or providers',async status=>{
 const {handler,admission,readConfig}=load(new Response('{"error":"denied"}',{status}));
 const response=await handler(req({image:'AA==',operation:'classify'}));
 expect(response.status).toBe(status);expect(admission).toHaveBeenCalledWith(expect.any(Request),expect.objectContaining({feature:'ai-process',limit:20}));expect(readConfig).not.toHaveBeenCalled();
});
test.each([{},{image:12,operation:'classify'},{image:'AA==',operation:'unknown'}])('invalid scan input is rejected before consuming quota: %j',async body=>{
 const {handler,admission,readConfig}=load();const response=await handler(req(body));expect(response.status).toBe(400);expect(admission).not.toHaveBeenCalled();expect(readConfig).not.toHaveBeenCalled();
});
test('malformed JSON does not consume quota',async()=>{
 const {handler,admission}=load();const response=await handler(new Request('https://fixture.invalid',{method:'POST',body:'{'}));expect(response.status).toBe(400);expect(admission).not.toHaveBeenCalled();
});
test('provider configuration values are absent from diagnostic logs',async()=>{
 const {handler,logs}=load();await handler(req({image:'AA==',operation:'classify'}));expect(JSON.stringify(logs)).not.toContain('sensitive-fixture');
});
