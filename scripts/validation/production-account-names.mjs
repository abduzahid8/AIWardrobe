/** Live Auth/profile requests confined to disposable accounts; credentials never logged. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!service)throw Error('Setup credential required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
const publicHeaders={apikey:anon,'Content-Type':'application/json'};
const call=async(path,headers,method='GET',body)=>{const r=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(30000),redirect:'error'});return{status:r.status,ok:r.ok,data:await r.json().catch(()=>null)};};
const report={measuredAt:new Date().toISOString(),scope:'Disposable production Auth-trigger, authenticated profile/ownership and anonymous availability HTTP checks; no existing-user changes or external provider consent',checks:[],cleanup:[]},users=[];
const check=(name,r,passed)=>report.checks.push({name,status:r.status,passed});
try {
 for(let i=0;i<2;i++){
  const suffix=randomUUID(),email=`names-${suffix}@example.invalid`,password=randomUUID()+'Aa!9',handle='audit_'+suffix.replaceAll('-','').slice(0,20);
  const created=await call('/auth/v1/admin/users',admin,'POST',{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:i===0?{handle:handle.toUpperCase(),display_name:'Shared Display Name'}:{full_name:'Shared Display Name'}});
  if(!created.ok||!created.data?.id)throw Error('Account fixture failed');
  const user={id:created.data.id,handle};users.push(user);
  const login=await call('/auth/v1/token?grant_type=password',publicHeaders,'POST',{email,password});
  if(!login.ok||!login.data?.access_token)throw Error('Fixture login failed');
  user.headers={...publicHeaders,Authorization:`Bearer ${login.data.access_token}`};
  const profile=await call(`/rest/v1/profiles?id=eq.${user.id}&select=handle,display_name,username`,user.headers);
  const row=profile.data?.[0];user.handle=row?.handle;
  check(i?'oauth_name_separate_from_generated_handle':'requested_handle_canonicalized',profile,profile.ok&&row?.display_name==='Shared Display Name'&&(i?/^user_[a-z0-9_]{25}$/.test(row?.handle):row?.handle===handle));
 }
 const [owner,other]=users;
 let r=await call('/rest/v1/rpc/is_username_available',publicHeaders,'POST',{check_username:owner.handle.toUpperCase()});check('anonymous_lookup_detects_case_variant',r,r.ok&&r.data===false);
 r=await call('/rest/v1/rpc/is_username_available',publicHeaders,'POST',{check_username:'bad handle'});check('anonymous_lookup_rejects_invalid_format',r,r.ok&&r.data===false);
 r=await call(`/rest/v1/profiles?id=eq.${owner.id}`,{...owner.headers,Prefer:'return=representation'},'PATCH',{display_name:'Apple New Display Name'});
 check('display_edit_keeps_unique_handle',r,r.ok&&r.data?.[0]?.handle===owner.handle&&r.data?.[0]?.username==='Apple New Display Name');
 r=await call(`/rest/v1/profiles?id=eq.${other.id}`,{...other.headers,Prefer:'return=representation'},'PATCH',{display_name:'Apple New Display Name'});
 check('display_names_may_be_shared',r,r.ok&&r.data?.[0]?.handle===other.handle);
 r=await call(`/rest/v1/profiles?id=eq.${other.id}`,other.headers,'PATCH',{handle:owner.handle.toUpperCase()});check('case_variant_duplicate_handle_rejected',r,r.status===409&&r.data?.code==='23505');
 r=await call(`/rest/v1/profiles?id=eq.${owner.id}`,owner.headers,'PATCH',{handle:'bad handle'});check('invalid_profile_handle_rejected',r,r.status===400&&r.data?.code==='23514');
 r=await call(`/rest/v1/profiles?id=eq.${other.id}`,{...owner.headers,Prefer:'return=representation'},'PATCH',{display_name:'Cross-owner write'});check('cross_owner_name_write_denied',r,r.ok&&Array.isArray(r.data)&&r.data.length===0);
 r=await call(`/rest/v1/profiles?id=eq.${owner.id}`,{...owner.headers,Prefer:'return=representation'},'PATCH',{username:'Legacy Display Edit'});check('legacy_display_edit_keeps_unique_handle',r,r.ok&&r.data?.[0]?.handle===owner.handle&&r.data?.[0]?.display_name==='Legacy Display Edit');
 const renamed='audit_'+randomUUID().replaceAll('-','').slice(0,20);
 r=await call(`/rest/v1/profiles?id=eq.${owner.id}`,{...owner.headers,Prefer:'return=representation'},'PATCH',{handle:renamed.toUpperCase()});check('handle_edit_canonicalizes_without_changing_name',r,r.ok&&r.data?.[0]?.handle===renamed&&r.data?.[0]?.display_name==='Legacy Display Edit');
 r=await call('/rest/v1/rpc/is_username_available',publicHeaders,'POST',{check_username:owner.handle});check('released_handle_available',r,r.ok&&r.data===true);
} catch(error){report.failure=error.message;process.exitCode=1;} finally{
 for(const user of users){const r=await call(`/auth/v1/admin/users/${user.id}`,admin,'DELETE');report.cleanup.push({deleted:r.ok,status:r.status});if(!r.ok)process.exitCode=1;}
 await writeFile('docs/reports/readiness-evidence/production-account-names.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(!report.checks.length||report.checks.some(c=>!c.passed))process.exitCode=1;
