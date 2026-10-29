/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript'),api={};
const root=process.env.OUTFIT_REQUEST_SOURCE_ROOT||path.resolve(__dirname,'../../backend/supabase/functions');
new Function('exports',ts.transpileModule(fs.readFileSync(path.join(root,'_shared/outfitGenerationRequest.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(api);
const req=body=>new Request('https://example.invalid',{method:'POST',body:typeof body==='string'?body:JSON.stringify(body)});
it.each([null,[],true,'not json'])('rejects non-object or malformed bodies: %j',async body=>{
 await expect(api.readOutfitGenerationRequest(req(body))).rejects.toMatchObject({status:400});
});
test('accepts the maximum selected pool without dropping IDs or request constraints',async()=>{
 const body={prompt:'x'.repeat(990)+' budget $5',limit:15,selectedItemIds:Array.from({length:200},(_,i)=>'item-'+i),preferences:{avoidColors:['red']}};
 expect(api.validateOutfitGenerationRequest(body)).toBe(body);
 expect(await api.readOutfitGenerationRequest(req(body))).toEqual(body);
});
test('preserves a legacy inline image at the measured existing maximum size',async()=>{
 const imageUrl='data:image/jpeg;base64,'+'a'.repeat(4047638);
 const body={wardrobeItems:[{id:'legacy-garment',imageUrl,name:'Shirt',brand:null,pattern:null,material:null,styleTags:null}]};
 const parsed=await api.readOutfitGenerationRequest(req(body));expect(parsed.wardrobeItems[0].imageUrl).toBe(imageUrl);
});
test('rejects declared excessive body size before reading it',async()=>{
 const request=new Request('https://example.invalid',{method:'POST',headers:{'content-length':String(api.MAX_GENERATION_BODY_BYTES+1)},body:'{}'});
 await expect(api.readOutfitGenerationRequest(request)).rejects.toMatchObject({status:413});
});
test('enforces actual chunked bytes even without Content-Length',async()=>{
 const body=new ReadableStream({start(controller){controller.enqueue(new Uint8Array(api.MAX_GENERATION_BODY_BYTES+1));}});
 const request=new Request('https://example.invalid',{method:'POST',body,duplex:'half'});
 await expect(api.readOutfitGenerationRequest(request)).rejects.toMatchObject({status:413});
});
test('a stalled body settles by deadline even if source cancellation hangs',async()=>{
 jest.useFakeTimers();
 try{
  const body=new ReadableStream({pull:()=>new Promise(()=>{}),cancel:()=>new Promise(()=>{})});
  const pending=api.readOutfitGenerationRequest(new Request('https://example.invalid',{method:'POST',body,duplex:'half'}));
  const rejected=expect(pending).rejects.toMatchObject({status:408});await jest.advanceTimersByTimeAsync(15000);await rejected;expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});
test('bounds normalized physical-product history independently from raw IDs',()=>{
 const valid={excludeProductKeys:['n:brown leather loafers','i:cdn.test/shoes.jpg']};expect(api.validateOutfitGenerationRequest(valid)).toBe(valid);
 for(const excludeProductKeys of [Array(451).fill('n:shoe'),['i:'+ 'x'.repeat(2048)],[{}]])expect(()=>api.validateOutfitGenerationRequest({excludeProductKeys})).toThrow('invalid_generation_request');
});
