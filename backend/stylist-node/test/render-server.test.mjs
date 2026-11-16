import test from 'node:test';
import assert from 'node:assert/strict';
import { createStylist6RenderServer } from '../render-server.mjs';
async function fixture(post){
  const server=createStylist6RenderServer(post);
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url=`http://127.0.0.1:${server.address().port}`;
  return {url,close:()=>new Promise(resolve=>server.close(resolve))};
}
test('portable Render adapter exposes only isolated signed S6 route and basic health',async()=>{
  let calls=0;
  const f=await fixture(async(request)=>{
    calls++;
    assert.equal(request.headers.get('x-aiw-signature'),'test-sig');
    assert.equal(await request.text(),'{}');
    return Response.json({success:true,outfits:[]},{status:200});
  });
  try{
    const h=await fetch(f.url+'/health');assert.equal(h.status,200);
    assert.equal((await h.json()).engine,'stylist-6-render-staging');
    assert.equal((await fetch(f.url+'/api/generate-outfits',{method:'POST'})).status,404);
    assert.equal((await fetch(f.url+'/api/stylist6-generate-outfits')).status,405);
    assert.equal(calls,0);
    const actual=await fetch(f.url+'/api/stylist6-generate-outfits',{
      method:'POST',headers:{'x-aiw-signature':'test-sig'},body:'{}'});
    assert.equal(actual.status,200);assert.equal(calls,1);
  }finally{await f.close();}
});
test('portable Render adapter refuses oversize inputs before invoking compute',async()=>{
  let calls=0;
  const f=await fixture(async()=>{calls++;return Response.json({})});
  try{
    const r=await fetch(f.url+'/api/stylist6-generate-outfits',{
      method:'POST',body:'x'.repeat(1024*1024+1)});
    assert.equal(r.status,413);assert.equal(calls,0);
  }finally{await f.close();}
});
