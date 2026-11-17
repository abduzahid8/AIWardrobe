/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');
const {rankNvidiaLooks,nvidiaRankingBody}=require('../../backend/supabase/functions/_shared/nvidiaLookRanking');
const root=process.env.NVIDIA_REQUEST_SOURCE_ROOT||path.resolve(__dirname,'../../backend/supabase/functions');
const source=fs.readFileSync(path.join(root,'generate-outfits/index.ts'),'utf8').replace(/^import .*\n/gm,'').replace(/\bexport\s+/g,'');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
const api=new Function('serve','createClient','Deno',compiled+'\nreturn {macroCategory};')(()=>{},()=>{},{env:{get:()=>undefined}});
test.each(['meta/llama-3.2-11b-vision-instruct','nvidia/nemotron-3.5-lightning-30b-a3b'])('NVIDIA %s ranks real complete-look IDs once using supported JSON mode',async model=>{
 const ids=['look127','look208','look009'];
 const fetchMock=jest.fn(async()=>new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({rankedLookIds:[...ids].reverse()})}}]})));
 const result=await rankNvidiaLooks('fixture-key',model,'Rank existing looks',{looks:ids.map(id=>({id}))},ids,fetchMock);
 expect(result.ids).toEqual([...ids].reverse());expect(result.attempt.status).toBe('ok');expect(fetchMock).toHaveBeenCalledTimes(1);
 const body=JSON.parse(fetchMock.mock.calls[0][1].body);
 expect(body.response_format).toEqual({type:'json_object'});expect(body.messages[0].content).toContain(JSON.stringify(ids));
 if(model.includes('nemotron'))expect(body.chat_template_kwargs).toEqual({enable_thinking:false});else expect(body.chat_template_kwargs).toBeUndefined();
});
test('a transport success with incomplete ranking remains a truthful deterministic fallback',async()=>{
 const fetchMock=jest.fn(async()=>new Response(JSON.stringify({choices:[{message:{content:'{"rankedLookIds":["look001","look002"]}'}}]})));
 const result=await rankNvidiaLooks('fixture-key','fixture-model','Rank',{},['look127','look208','look009'],fetchMock);
 expect(result.ids).toBeNull();expect(result.attempt).toMatchObject({status:'rank_contract_mismatch',expected:3,received:2,unknown:2});
 expect(JSON.stringify(result)).not.toContain('look001');expect(JSON.stringify(result)).not.toContain('fixture-key');
});
test.each([
 ['tops','upper_body','кроссовки ttnm light_runn_ltnu 10249949 01','shoes'],
 ['tops','upper_body','Жилет SLVSS FEATHERWEIGHT NYLON CHNNL','outerwear'],
 ['tops','upper_body','Джинсы SLIM','bottom'],
 ['bottoms','lower_body','Футболка CLASSIC TEE','top'],
 ['tops','upper_body','Белая рубашка','top'],
])('Russian catalog garment %s/%s/%s resolves to %s',(category,type,name,expected)=>{
 expect(api.macroCategory(category,type,name)).toBe(expected);
});
test('rank prompts cannot request product substitution or ungrounded example IDs',()=>{
 const prompt=nvidiaRankingBody('fixture-model','Rank',{},['look127','look009']).messages[0].content;
 expect(prompt).toContain('No product IDs');expect(prompt).not.toContain('look001');
});
