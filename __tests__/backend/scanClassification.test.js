/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');
const entry=process.env.SCAN_CLASSIFICATION_SOURCE||path.resolve(__dirname,'../../backend/supabase/functions/ai-process/index.ts');
const valid={category:'shirt',section:'tops',style:'casual',color:'white',material:'cotton',description:'White cotton shirt.',confidence:0.91};
function load(provider){let handler;const admission=jest.fn(async()=>null);const fetchMock=jest.fn(provider);const config={select:()=>config,eq:()=>config,single:async()=>({data:{value:'fixture-token'},error:null})};
 const compiled=ts.transpileModule(fs.readFileSync(entry,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 new Function('require','exports','Deno','console','fetch',compiled)(name=>{
  if(name.includes('server.ts'))return {serve:fn=>handler=fn};if(name.includes('rateLimit'))return {enforceRateLimit:admission};if(name.includes('supabase-js'))return {createClient:()=>({from:()=>config})};if(name==='node:buffer')return require(name);return {};
 },{},{env:{get:()=> 'fixture-config'}},{log:()=>{},error:()=>{},warn:()=>{}},fetchMock);
 return {handler,fetchMock,admission};}
const request=()=>new Request('https://fixture.invalid',{method:'POST',headers:{Authorization:'Bearer fixture'},body:JSON.stringify({image:'AA==',operation:'classify'})});
const envelope=content=>new Response(JSON.stringify({choices:[{message:{content:typeof content==='string'?content:JSON.stringify(content)}}]}));
test.each([401,429,500])('provider HTTP %s cannot become a successful fabricated garment',async status=>{
 const {handler}=load(async()=>new Response('private-provider-detail',{status}));const response=await handler(request());const body=await response.json();expect(response.status).toBe(503);expect(body.success).toBe(false);expect(body.classification).toBeNull();expect(body.error).toBe('ai_classification_unavailable');expect(JSON.stringify(body)).not.toContain('private-provider-detail');
});
test.each(['', 'not JSON', {}, {...valid,color:'invented-color'}, {...valid,description:''}, {...valid,category:'invented-category'}])('invalid output %j cannot be accepted',async content=>{
 const {handler}=load(async()=>envelope(content));const response=await handler(request());expect(response.status).toBe(503);expect((await response.json()).success).toBe(false);
});
test('valid classification preserves reported attributes and confidence',async()=>{
 const {handler}=load(async()=>envelope(valid));const response=await handler(request());expect(response.status).toBe(200);const body=await response.json();expect(body.classification).toMatchObject({category:'shirt',section:'tops',confidence:0.91,attributes:{color:'white',style:'casual',material:'cotton'}});
});
test('contradictory section is derived from the recognized garment category',async()=>{
 const {handler}=load(async()=>envelope({...valid,section:'shoes'}));const response=await handler(request());expect(response.status).toBe(200);expect((await response.json()).classification.section).toBe('tops');
});
test('missing confidence is unknown rather than invented certainty',async()=>{
 const {confidence,...without}=valid;const {handler}=load(async()=>envelope(without));const response=await handler(request());expect((await response.json()).classification.confidence).toBe(0);
});
test.each(['fetch','body'])('classification deadline covers a stalled %s',async phase=>{
 jest.useFakeTimers();
 try{
  const {handler,fetchMock}=load(async(_url,options)=>{
   const stalled=()=>new Promise((_resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true}));
   return phase==='fetch'?stalled():{ok:true,text:stalled};
  });
  const pending=handler(request());
  for(let n=0;n<12;n++)await Promise.resolve();
  expect(fetchMock).toHaveBeenCalledTimes(1);
  await jest.advanceTimersByTimeAsync(45000);
  const response=await pending;expect(response.status).toBe(503);
  expect((await response.json()).classification).toBeNull();
 }finally{jest.useRealTimers();}
});
test('the actual photo is bound to the final user message after the JSON example',async()=>{
 const {handler,fetchMock}=load(async()=>envelope(valid));await handler(request());
 const sent=JSON.parse(fetchMock.mock.calls[0][1].body);
 expect(sent.messages.at(-1)).toMatchObject({role:'user',content:expect.arrayContaining([{type:'image_url',image_url:{url:'data:image/jpeg;base64,AA=='}}])});
 expect(sent.messages.slice(0,-1).some(message=>message.role==='assistant'&&JSON.parse(message.content).category)).toBe(true);
});

test.each(['null','unknown','fabric','NONE'])('unknown material sentinel %s remains null rather than a claimed fabric',async material=>{
 const {handler}=load(async()=>envelope({...valid,material}));const response=await handler(request());expect(response.status).toBe(200);expect((await response.json()).classification.attributes.material).toBeNull();
});

const batchRequest = frames => new Request('https://fixture.invalid', {method:'POST',headers:{Authorization:'Bearer fixture'},body:JSON.stringify({operation:'classify_frames',frames})});
test('all video frames are classified in order under one quota admission', async () => {
 const {handler,fetchMock,admission}=load(async()=>envelope(valid));
 const response=await handler(batchRequest(['AA==','BB==','CC==']));
 const body=await response.json();expect(response.status).toBe(200);
 expect(admission).toHaveBeenCalledTimes(1);expect(fetchMock).toHaveBeenCalledTimes(3);
 expect(body.frames.map(frame=>frame.frameIndex)).toEqual([0,1,2]);
 const urls=fetchMock.mock.calls.map(call=>JSON.parse(call[1].body).messages.at(-1).content[0].image_url.url);
 expect(urls).toEqual(['AA==','BB==','CC=='].map(frame=>'data:image/jpeg;base64,'+frame));
});
test('a failed later frame cannot return a partial video scan as successful', async () => {
 let calls=0;const {handler}=load(async()=>++calls===1?envelope(valid):new Response('unavailable',{status:429}));
 const response=await handler(batchRequest(['AA==','BB==']));const body=await response.json();
 expect(response.status).toBe(503);expect(body.success).toBe(false);expect(body.frames).toBeUndefined();
});
test.each([[],['AA==',null],['AA==','BB==','CC==','DD=='],[' ']].map(frames=>[frames]))('invalid video batch is rejected before admission %j',async frames=>{
 const {handler,admission,fetchMock}=load(async()=>envelope(valid));const response=await handler(batchRequest(frames));
 expect(response.status).toBe(400);expect(admission).not.toHaveBeenCalled();expect(fetchMock).not.toHaveBeenCalled();
});
test('video total payload limit is enforced before admission',async()=>{
 const {handler,admission}=load(async()=>envelope(valid));
 const response=await handler(batchRequest(['A'.repeat(8*1024*1024),'B'.repeat(8*1024*1024)]));
 expect(response.status).toBe(400);expect(admission).not.toHaveBeenCalled();
});
