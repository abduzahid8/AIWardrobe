/** Actual installed SDK header precedence; no network request is made. */
const {createClient}=jest.requireActual('@supabase/supabase-js') as typeof import('@supabase/supabase-js');

function fixture(){
 const request=jest.fn(async (_input:RequestInfo|URL,_init?:RequestInit)=>({
  ok:true,headers:new Headers({'Content-Type':'application/json'}),json:async()=>({success:true}),
 } as Response));
 const client=createClient('https://example.supabase.co','public-test-key',{
  accessToken:async()=>{await Promise.resolve();return 'newer-owner-token';},
  auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
  global:{fetch:request},
 });
 return {client,request};
}

test('invoke-level captured user JWT survives SDK asynchronous token lookup',async()=>{
 const {client,request}=fixture();
 const result=await client.functions.invoke('delete-account',{method:'POST',headers:{Authorization:'Bearer initiating-owner-token'}});
 expect(result.error).toBeNull();
 const headers=new Headers(request.mock.calls[0][1]?.headers);
 expect(headers.get('Authorization')).toBe('Bearer initiating-owner-token');
 expect(headers.get('apikey')).toBe('public-test-key');
});

test('without an explicit JWT the SDK instead uses its latest token',async()=>{
 const {client,request}=fixture();
 const result=await client.functions.invoke('delete-account',{method:'POST'});
 expect(result.error).toBeNull();
 expect(new Headers(request.mock.calls[0][1]?.headers).get('Authorization')).toBe('Bearer newer-owner-token');
});
