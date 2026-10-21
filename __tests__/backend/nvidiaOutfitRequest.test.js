/** @jest-environment node */
const fs=require('fs'), path=require('path'), ts=require('typescript');
const root=process.env.NVIDIA_REQUEST_SOURCE_ROOT || path.resolve(__dirname,'../../backend/supabase/functions');
const strip=source=>source.replace(/^import .*\n/gm,'').replace(/\bexport\s+/g,'');
const source=['_shared/modelOutfitResponse.ts','generate-outfits/index.ts'].map(file=>strip(fs.readFileSync(path.join(root,file),'utf8'))).join('\n');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
test.each(['meta/llama-3.2-11b-vision-instruct','nvidia/nemotron-3.5-lightning-30b-a3b'])('NVIDIA request for %s returns parsed IDs with appropriate thinking control',async model=>{
 const fetchMock=jest.fn(async()=>new Response(JSON.stringify({choices:[{message:{content:'{"outfits":[{"items":[{"id":"fixture-top"},{"id":"fixture-bottom"},{"id":"fixture-shoes"}]}]}'}}]})));
 const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {callNvidia};')(()=>{},()=>{}, {env:{get:()=>undefined}},fetchMock);
 const result=await api.callNvidia('fixture-key',model,'system','candidates');
 expect(result[0].items.map(item=>item.id)).toEqual(['fixture-top','fixture-bottom','fixture-shoes']);
 const body=JSON.parse(fetchMock.mock.calls[0][1].body);
 if(model.includes('nemotron'))expect(body.chat_template_kwargs).toEqual({enable_thinking:false});
 else expect(body.chat_template_kwargs).toBeUndefined();
});


test('NVIDIA deadline remains active while the response body is being read',async()=>{
 jest.useFakeTimers();
 try {
  let signal;
  const fetchMock=jest.fn(async(_url,options)=>{
   signal=options.signal;
   return {ok:true,status:200,json:()=>new Promise((_,reject)=>{
    signal.addEventListener('abort',()=>reject(new Error('body aborted')),{once:true});
   })};
  });
  const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {callNvidia};')(()=>{},()=>{}, {env:{get:()=>undefined}},fetchMock);
  let settled=false;const trace=[];
  const pending=api.callNvidia('fixture-key','meta/llama-3.2-11b-vision-instruct','system','candidates',0,trace).then(value=>{settled=true;return value;});
  await jest.advanceTimersByTimeAsync(45000);
  expect(signal.aborted).toBe(true);
  expect(settled).toBe(true);
  expect(await pending).toBeNull();
  expect(trace).toEqual([expect.objectContaining({status:'timeout'})]);
  expect(jest.getTimerCount()).toBe(0);
 } finally {jest.useRealTimers();}
});

test.each([
 ['tops','upper_body','кроссовки ttnm light_runn_ltnu 10249949 01','shoes'],
 ['tops','upper_body','Жилет SLVSS FEATHERWEIGHT NYLON CHNNL','outerwear'],
 ['tops','upper_body','Джинсы SLIM','bottom'],
 ['bottoms','lower_body','Футболка CLASSIC TEE','top'],
 ['tops','upper_body','Белая рубашка','top'],
])('Russian catalog garment %s/%s/%s resolves to %s', (category,type,name,expected)=>{
 const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {macroCategory};')(()=>{},()=>{}, {env:{get:()=>undefined}},jest.fn());
 expect(api.macroCategory(category,type,name)).toBe(expected);
});

test.each(['headers','body'])('deadline settles even if stalled %s ignores abort',async stage=>{
 jest.useFakeTimers();
 try{
  const never=()=>new Promise(()=>{});
  const fetchMock=jest.fn(stage==='headers'?never:async()=>({ok:true,status:200,json:never}));
  const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {callNvidia};')(()=>{},()=>{},{env:{get:()=>undefined}},fetchMock);
  let settled=false;const trace=[];
  api.callNvidia('fixture-key','fixture-model','system','candidates',0,trace).then(()=>{settled=true;});
  await jest.advanceTimersByTimeAsync(45001);
  expect(settled).toBe(true);expect(trace[0].status).toBe('timeout');
  expect(fetchMock.mock.calls[0][1].signal.aborted).toBe(true);
  expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});
test.each([['connection',()=>Promise.reject(new TypeError('fetch failed')),'network_error'],['json',async()=>({ok:true,status:200,json:async()=>{throw new SyntaxError('bad JSON');}}),'invalid_response']])('distinguishes %s errors from timeouts',async(_stage,transport,status)=>{
 const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {callNvidia};')(()=>{},()=>{},{env:{get:()=>undefined}},transport);
 const trace=[];expect(await api.callNvidia('fixture-key','fixture-model','system','candidates',0,trace)).toBeNull();
 expect(trace[0].status).toBe(status);
});

test('provider prompt never asks for off-style shoes or dropping a required climate layer',()=>{
 const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {buildSystemPrompt};')(()=>{},()=>{},{env:{get:()=>undefined}},jest.fn());
 const prompt=api.buildSystemPrompt(1,'old_money',true,{},'office',{amount:300,currency:'USD'});
 expect(prompt).not.toContain('LEAST off-aesthetic');
 expect(prompt).not.toContain('unless it is the only option');
 expect(prompt).not.toContain('drop the formal outerwear');
 expect(prompt).toContain('Return fewer outfits or an empty outfits array rather than relaxing the request');
});

test('multiword style bans preserve their scope instead of banning permitted footwear',()=>{
 const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {extractRejectKeywords};')(()=>{},()=>{},{env:{get:()=>undefined}},jest.fn());
 const banned=api.extractRejectKeywords(['NEVER include: loafers with tie laces, chunky sneakers, driving moccasins.']);
 expect(banned).toEqual(['loafers with tie laces','chunky sneakers','driving moccasins']);
 expect(banned).not.toContain('loafers');
 expect(banned).not.toContain('sneaker');
});

test.each([false,true])('final NVIDIA output example contains every required slot (layered=%s)',async layered=>{
 const fetchMock=jest.fn(async()=>new Response(JSON.stringify({choices:[{message:{content:'{"outfits":[{"items":[]}]}'}}]})));
 const api=new Function('serve','createClient','Deno','fetch',compiled+'\nreturn {callNvidia};')(()=>{},()=>{},{env:{get:()=>undefined}},fetchMock);
 await api.callNvidia('fixture-key','fixture-model','system','candidates',0,[],layered);
 const system=JSON.parse(fetchMock.mock.calls[0][1].body).messages[0].content;
 const example=JSON.parse(system.split('OUTPUT CONTRACT: Return ')[1].split('. Every item')[0]);
 expect(example.outfits[0].items.map(item=>item.id)).toEqual((layered?['BASE TOP','OUTERWEAR','BOTTOM','SHOES']:['TOP','BOTTOM','SHOES']).map(slot=>`EXACT ${slot} CANDIDATE ID`));
 expect(system).toContain('Never replace a required slot with another top');
 expect(system).toContain('A dress replaces top and bottom');
});
