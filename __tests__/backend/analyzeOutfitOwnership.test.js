/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(process.env.ANALYZE_OUTFIT_SOURCE || path.resolve(__dirname, '../../backend/supabase/functions/analyze-outfit/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
function load({ authenticated = true, denial = null, wardrobe = null, shop = [], providerContent = null, finishReason = 'stop', providerStatus = 200, detected = [{category: 'shirt', primaryColor: 'white'}] } = {}) {
  let handler;
  const admission = jest.fn(async () => denial);
  let clientNumber = 0;
  const queries = [];
  const fetchMock = jest.fn(async () => new Response(JSON.stringify({ choices: [{ message: { content: providerContent ?? JSON.stringify(detected.map(item=>({section:({jacket:'outerwear',coat:'outerwear',cardigan:'outerwear',pants:'bottoms',trousers:'bottoms',jeans:'bottoms',shorts:'bottoms',skirt:'bottoms',dress:'dresses',sneakers:'shoes',boots:'shoes',sandals:'shoes'}[item.category]||'tops'),specificType:item.category,primaryColor:'white',colorHex:'#ffffff',style:'casual',material:null,pattern:'solid',fit:'regular',description:'A visible garment.',...item}))) }, finish_reason: finishReason }] }), {status:providerStatus}));
  const createClient = () => {
    const number = clientNumber++;
    return {
      auth: { getUser: async () => ({ data: { user: authenticated ? { id: 'authenticated-owner' } : null }, error: null }) },
      from: table => {
        const query = { client: number, table, filters: {} };
        queries.push(query);
        const chain = {
          select: () => chain,
          eq: (key, value) => { query.filters[key] = value; return chain; },
          single: async () => ({ data: { value: 'test-provider-key' }, error: null }),
          limit: async () => ({ data: table === 'clothing_items' ? wardrobe || [{ id: query.filters.user_id === 'authenticated-owner' ? 'owned-item' : 'other-users-private-item', category: 'shirt', primary_color: 'white' }] : shop, error: null }),
        };
        return chain;
      },
    };
  };
  new Function('require', 'exports', 'Deno', 'fetch', compiled)(name => {
    if (name.includes('rateLimit')) return { enforceRateLimit: admission };
    if (name.includes('server.ts')) return { serve: fn => { handler = fn; } };
    return { createClient };
  }, {}, { env: { get: () => 'test-configuration' } }, fetchMock);
  return { handler, queries, fetchMock, admission };
}
const request = body => new Request('https://example.test', { method: 'POST', headers: { Authorization: 'Bearer user-jwt' }, body: JSON.stringify(body) });

test('payload userId cannot select another owner through the service role', async () => {
  const { handler, queries } = load();
  const response = await handler(request({ image: 'base64-image', userId: 'victim-owner' }));
  expect(response.status).toBe(200);
  const wardrobeQuery = queries.find(query => query.table === 'clothing_items');
  expect(wardrobeQuery).toMatchObject({ client: 1, filters: { user_id: 'authenticated-owner' } });
  const body = await response.json();
  expect(body.recommendations[0].similarFromWardrobe.map(item => item.id)).toEqual(['owned-item']);
});

test('authentication failure stops before provider work or wardrobe reads', async () => {
  const { handler, queries, fetchMock } = load({ authenticated: false });
  expect((await handler(request({ image: 'base64-image', userId: 'victim-owner' }))).status).toBe(401);
  expect(fetchMock).not.toHaveBeenCalled();
  expect(queries).toEqual([]);
});

test.each([null, {}, '', 'a'.repeat(14 * 1024 * 1024 + 1)])('rejects invalid image shape or size before external work', async image => {
  const { handler, fetchMock } = load();
  expect((await handler(request({ image }))).status).toBe(400);
  expect(fetchMock).not.toHaveBeenCalled();
});

test.each([403,429,503])('analysis admission denial %s prevents wardrobe and provider access',async status=>{
 const {handler,queries,fetchMock,admission}=load({denial:new Response('{"error":"denied"}',{status})});
 const response=await handler(request({image:'base64-image',userId:'victim-owner'}));expect(response.status).toBe(status);
 expect(admission).toHaveBeenCalledWith(expect.any(Request),expect.objectContaining({feature:'analyze-outfit',limit:10}));
 expect(queries).toEqual([]);expect(fetchMock).not.toHaveBeenCalled();
});
test('the actual outfit image stays in the final user message after the JSON-array example',async()=>{
 const {handler,fetchMock}=load();await handler(request({image:'actual-outfit-photo'}));
 const body=JSON.parse(fetchMock.mock.calls[0][1].body);
 expect(body.messages.at(-1)).toMatchObject({role:'user',content:expect.arrayContaining([{type:'image_url',image_url:{url:'data:image/jpeg;base64,actual-outfit-photo'}}])});
 expect(body.messages.slice(0,-1).some(message=>message.role==='assistant'&&Array.isArray(JSON.parse(message.content)))).toBe(true);
});

async function recommendations(config) {
 const {handler}=load(config);const response=await handler(request({image:'photo'}));expect(response.status).toBe(200);return (await response.json()).recommendations[0];
}
test('matching color, style and pattern never makes trousers similar to a jacket',async()=>{
 const result=await recommendations({detected:[{category:'jacket',primaryColor:'beige',style:'casual',pattern:'solid'}],wardrobe:[{id:'wrong-slot',category:'pants',color:'beige',style:'casual',pattern:'solid'}],shop:[{id:'wrong-shop-slot',garment_type:'pants',name:'beige casual pants'}]});
 expect(result.similarFromWardrobe).toEqual([]);expect(result.similarFromShop).toEqual([]);
});
test('missing clothing types never earn an empty-string category match',async()=>{
 const result=await recommendations({detected:[{category:'shirt',primaryColor:'white'}],wardrobe:[{id:'unknown',color:'white'}],shop:[{id:'unknown-shop',name:'white cotton piece'}]});
 expect(result.similarFromWardrobe).toEqual([]);expect(result.similarFromShop).toEqual([]);
});
test('t-shirt is not a substring match for shirt; typed broad categories use their specific garment type',async()=>{
 const result=await recommendations({wardrobe:[{id:'tshirt',category:'t-shirt',color:'white'},{id:'shirt',category:'tops',type:'shirt',primary_color:'white'}],shop:[{id:'shop-tshirt',garment_type:'t-shirt',name:'white tshirt'},{id:'shop-shirt',category:'tops',garment_type:'shirt',name:'white shirt'}]});
 expect(result.similarFromWardrobe.map(i=>i.id)).toEqual(['shirt']);expect(result.similarFromShop.map(i=>i.id)).toEqual(['shop-shirt']);
});
test('primary_color participates in ranking and legitimate trouser aliases match',async()=>{
 const result=await recommendations({detected:[{category:'pants',primaryColor:'green'}],wardrobe:[{id:'wrong-color',category:'pants',primary_color:'red'},{id:'same-color',category:'trousers',primary_color:'green'}]});
 expect(result.similarFromWardrobe.map(i=>i.id)).toEqual(['same-color','wrong-color']);
});

test('broad catalog labels use explicit English/Russian garment names, never shared color alone',async()=>{
 const result=await recommendations({detected:[{category:'sneakers',primaryColor:'white'}],shop:[{id:'ru',garment_type:'upper_body',category:'tops',name:'Белые кроссовки'},{id:'en',garment_type:'shoes',category:'shoes',name:'White textile trainers'},{id:'wrong',garment_type:'upper_body',category:'tops',name:'White shirt'},{id:'ambiguous',garment_type:'upper_body',name:'Shirt with sneakers print'}]});
 expect(result.similarFromShop.map(i=>i.id).sort()).toEqual(['en','ru']);
});
test('dress shirts remain shirts and contradictory specific catalog types are withheld',async()=>{
 const result=await recommendations({shop:[{id:'dress-shirt',garment_type:'upper_body',name:'White dress shirt'},{id:'contradiction',garment_type:'pants',name:'White shirt'}]});
 expect(result.similarFromShop.map(i=>i.id)).toEqual(['dress-shirt']);
});

test.each(['Linen dress','Summer dresses','Белое платье'])('broad dress catalog title is recognized: %s',async name=>{
 const result=await recommendations({detected:[{category:'dress',primaryColor:'white'}],shop:[{id:'dress',garment_type:'upper_body',category:'tops',name}]});
 expect(result.similarFromShop.map(i=>i.id)).toEqual(['dress']);
});

test.each(['not JSON','[{"category":"shirt"},]','[{"category":"shirt"}, {"category":','[null]','[1]','[{}]','[{"category":"shirt","primaryColor":"white"}]'])('invalid or incomplete provider output is an explicit retryable failure: %s',async providerContent=>{
 const {handler,queries}=load({providerContent});const response=await handler(request({image:'photo'}));
 expect(response.status).toBe(503);expect(await response.json()).toMatchObject({success:false,error:'ai_analysis_unavailable'});expect(queries.some(q=>q.table==='clothing_items'||q.table==='shop_catalog')).toBe(false);
});
test('a genuine empty JSON array remains distinct from invalid provider output',async()=>{
 const {handler}=load({providerContent:'[]'});const response=await handler(request({image:'photo'}));expect(response.status).toBe(200);expect(await response.json()).toMatchObject({success:false,detectedItems:[]});
});
test('truncated completion is not presented as a complete outfit even if it contains parseable JSON',async()=>{
 const {handler}=load({finishReason:'length'});expect((await handler(request({image:'photo'}))).status).toBe(503);
});
test.each([429,503])('provider HTTP %s is a retryable analysis failure',async providerStatus=>{
 const {handler}=load({providerStatus});const response=await handler(request({image:'photo'}));expect(response.status).toBe(503);expect(await response.json()).toMatchObject({success:false,error:'ai_analysis_unavailable'});
});

test.each([{section:'bottoms'},{primaryColor:{}},{colorHex:'beige'},{style:'invented'},{material:{}},{pattern:[]},{fit:'unknown'},{description:''}])('invalid structured garment field is rejected: %j',async override=>{
 const {handler}=load({detected:[{category:'shirt',primaryColor:'white',...override}]});const response=await handler(request({image:'photo'}));expect(response.status).toBe(503);expect(await response.json()).toMatchObject({success:false,error:'ai_analysis_unavailable'});
});

test('single-garment mode actually requests one item instead of a full outfit',async()=>{
 const {handler,fetchMock}=load();expect((await handler(request({image:'photo',mode:'single'}))).status).toBe(200);const prompt=JSON.parse(fetchMock.mock.calls[0][1].body).messages.at(-1).content.find(v=>v.type==='text').text;expect(prompt).toContain('exactly one');expect(prompt).not.toContain('Identify ALL visible clothing items');
});
test('unknown analysis modes are rejected before provider work',async()=>{
 const {handler,fetchMock}=load();expect((await handler(request({image:'photo',mode:'unsupported'}))).status).toBe(400);expect(fetchMock).not.toHaveBeenCalled();
});
test('multiple detected items cannot masquerade as a successful single-garment analysis',async()=>{
 const {handler}=load({detected:[{category:'shirt'},{category:'pants'}]});expect((await handler(request({image:'photo',mode:'single'}))).status).toBe(503);
});

test('explicit unknown fit is retained as null without inventing regular fit',async()=>{
 const {handler}=load({detected:[{category:'shirt',fit:null}]});const response=await handler(request({image:'photo'}));expect(response.status).toBe(200);const data=await response.json();expect(data.detectedItems[0].fit).toBeNull();
});

test('hosted rejection diagnostics record field names without model text or image input',async()=>{
 const warn=jest.spyOn(console,'warn').mockImplementation(()=>{});
 try{
  const {handler}=load({detected:[{category:'shirt',style:'PRIVATE-MODEL-CONTENT'}]});const response=await handler(request({image:'PRIVATE-IMAGE-CONTENT',userId:'PRIVATE-OWNER'}));expect(response.status).toBe(503);expect(response.headers.get('X-Analysis-Rejection')).toBe('invalid_item_fields');expect(response.headers.get('X-Analysis-Fields')).toBe('style');expect(JSON.stringify([...response.headers])).not.toMatch(/PRIVATE-/);
  expect(warn).toHaveBeenCalledWith('analysis_rejected',JSON.stringify({reason:'invalid_item_fields',items:[{index:0,fields:['style']}]}));
  expect(JSON.stringify(warn.mock.calls)).not.toMatch(/PRIVATE-/);
 }finally{warn.mockRestore();}
});

test.each([['tops','white shirt','shirt','tops'],['tops','beige jacket','jacket','outerwear'],['outerwear','wool coat','coat','outerwear'],['bottoms','blue jeans','jeans','bottoms'],['accessories','black belt','belt','accessories'],['shoes','black loafers','loafer','shoes']])('broad model label %s is resolved from explicit garment description %s',async(category,specificType,expectedCategory,expectedSection)=>{
 const {handler}=load({detected:[{category,section:category,specificType}]});const response=await handler(request({image:'photo'}));expect(response.status).toBe(200);expect((await response.json()).detectedItems[0]).toMatchObject({category:expectedCategory,section:expectedSection});
});
test.each(['beige casual item','shirt and pants set','white shirt'])('ambiguous or contradictory broad-bottom label is rejected: %s',async specificType=>{
 const {handler}=load({detected:[{category:'bottoms',section:'bottoms',specificType}]});expect((await handler(request({image:'photo'}))).status).toBe(503);
});

test('an explicit unknown pattern is retained as null, without fabricating solid fabric',async()=>{
 const {handler}=load({detected:[{category:'jacket',pattern:null}]});const response=await handler(request({image:'photo',mode:'single'}));expect(response.status).toBe(200);const data=await response.json();expect(data.detectedItems[0].pattern).toBeNull();expect(data.itemCount).toBe(1);
});

test.each([['Plain','solid'],['no_pattern','solid'],['solid-colored','solid'],['Stripes','striped'],['Checkered','plaid'],['floral','other'],['textured','other'],['Unknown',null],['n/a',null]])('known pattern wording %s preserves its meaning as %s',async(pattern,expected)=>{
 const {handler}=load({detected:[{category:'jacket',pattern}]});const response=await handler(request({image:'photo',mode:'single'}));expect(response.status).toBe(200);expect(response.headers.get('X-Analysis-Contract')).toBe('20261019-patterns-v4');expect((await response.json()).detectedItems[0].pattern).toBe(expected);
});
test('unrecognized pattern text still fails instead of fabricating a known pattern',async()=>{
 const {handler}=load({detected:[{category:'jacket',pattern:'invented_pattern'}]});expect((await handler(request({image:'photo'}))).status).toBe(503);
});

test.each([[undefined,'missing'],['solid|striped','copied_option_list'],['','empty_string'],[[],'array'],[{},'object']])('pattern failure shape is diagnostic without retaining free text: %j',async(pattern,expected)=>{
 const {handler}=load({detected:[{category:'jacket',pattern}]});const response=await handler(request({image:'photo'}));expect(response.status).toBe(503);expect(response.headers.get('X-Analysis-Pattern-Form')).toBe(expected);
});

test.each(['null','NULL',' null '])('quoted null pattern is decoded as explicit unknown: %j',async pattern=>{
 const {handler}=load({detected:[{category:'jacket',pattern}]});const response=await handler(request({image:'photo',mode:'single'}));expect(response.status).toBe(200);expect((await response.json()).detectedItems[0].pattern).toBeNull();
});
