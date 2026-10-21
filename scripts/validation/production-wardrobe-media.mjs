/** Verify the new private media boundary before enabling garment-save cutover. */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { decode } from 'base64-arraybuffer';
import { transpileModule, ModuleKind } from 'typescript';
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID, createHash } from 'node:crypto';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL, anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY, key = process.env.SUPABASE_SERVICE_KEY;
if (!base || !anon || !key) throw new Error('Fixture setup credential required');
const options = { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } };
const admin = createClient(base,key,options), users = [], paths = [];
const report = { measuredAt: new Date().toISOString(), scope: 'Actual wardrobeMedia source; two disposable accounts, unchanged 2.9MB repository PNG; no app cutover/provider calls', checks: [], cleanup: [] };
const compile = async path => transpileModule(await readFile(new URL(path,import.meta.url),'utf8'),{ compilerOptions: { module: ModuleKind.CommonJS } }).outputText;
const events = {}; new Function('exports',await compile('../../store/authEvents.ts'))(events);
const code = await compile('../../src/services/wardrobeMedia.ts');
report.executedServiceJavaScriptSha256 = createHash('sha256').update(code).digest('hex');
const bytes = await readFile(new URL('../../assets/images/basic_outfit_1.png',import.meta.url));
report.fixtureBytes = bytes.length;
const checksum = value => createHash('sha256').update(value).digest('hex');
report.fixtureSha256 = checksum(bytes);
function check(name, passed) { report.checks.push({name,passed:Boolean(passed)}); if (!passed) throw new Error('AssertionError'); }
async function download(url,headers) {
  const response = await fetch(url,{headers,redirect:'error',signal:AbortSignal.timeout(20000)});
  const body = Buffer.from(await response.arrayBuffer());
  return {ok:response.ok,status:response.status,checksum:checksum(body),bytes:body.length};
}
try {
  for (let index=0;index<2;index++) {
    const tag=randomUUID(),email=`readiness-${tag}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
    const created=await admin.auth.admin.createUser({email,password,email_confirm:true,user_metadata:{username:`readiness_${tag}`,readiness_test:true}});
    if(created.error||!created.data.user)throw new Error('FixtureSetupError');
    const user={id:created.data.user.id,client:createClient(base,anon,options)};users.push(user);
    const login=await user.client.auth.signInWithPassword({email,password});
    if(login.error||!login.data.session)throw new Error('FixtureLoginError');
    user.accessToken=login.data.session.access_token;
    const module={};new Function('exports','require',code)(module,name=>{
      if(name==='base64-arraybuffer')return {decode};
      if(name==='../../lib/supabase')return {supabase:user.client};
      if(name==='../../store/authEvents')return events;
      if(name==='../config/env')return {Config:{supabase:{url:base,anonKey:anon}}};
      throw new Error('Unexpected dependency');
    });user.media=module;
  }
  const [owner,other]=users,id=randomUUID(),path=`${owner.id}/${id}/original.png`;paths.push(path);
  events.authEvents.emitLogin(owner.id);
  const input=`data:image/png;base64,${bytes.toString('base64')}`;
  const uri=await owner.media.uploadWardrobeImage(owner.id,id,input);
  check('upload_returns_canonical_private_reference',uri.includes('/object/authenticated/wardrobe-images/')&&!uri.includes('?'));
  check('retry_retains_stable_reference',(await owner.media.uploadWardrobeImage(owner.id,id,input))===uri);
  const ownerRead=await download(uri,owner.media.wardrobeMediaHeaders(uri,owner.id,owner.accessToken));
  check('authenticated_download_preserves_all_bytes',ownerRead.ok&&ownerRead.bytes===bytes.length&&ownerRead.checksum===report.fixtureSha256);
  check('public_endpoint_cannot_read',!(await download(uri.replace('/authenticated/','/public/'))).ok);
  check('unauthenticated_endpoint_cannot_read',!(await download(uri)).ok);
  const foreign=await other.client.storage.from('wardrobe-images').download(path);
  check('other_owner_cannot_download',Boolean(foreign.error));
  const overwrite=await other.client.storage.from('wardrobe-images').upload(path,new Uint8Array([137,80,78,71,13,10,26,10]),{contentType:'image/png',upsert:true});
  check('other_owner_cannot_overwrite',Boolean(overwrite.error));
  const deniedList=await other.client.storage.from('wardrobe-images').list(`${owner.id}/${id}`);
  check('other_owner_cannot_list',!deniedList.error&&Array.isArray(deniedList.data)&&deniedList.data.length===0);
  const signed=await owner.media.createWardrobeImageAccessUrl(owner.id,uri);
  const signedRead=await download(signed);
  check('explicit_temporary_access_preserves_bytes',signedRead.ok&&signedRead.checksum===report.fixtureSha256);
  const removed=await owner.client.functions.invoke('delete-account',{method:'POST',body:{}});
  check('account_delete_cleans_private_bucket',!removed.error&&removed.data?.success===true);
  const absent=await admin.storage.from('wardrobe-images').list(`${owner.id}/${id}`);
  check('deleted_owner_media_absent',!absent.error&&absent.data.length===0);
  check('deleted_object_invalidates_signed_link',!(await download(signed)).ok);
  const intact=await admin.auth.admin.getUserById(other.id);
  check('other_account_preserved',!intact.error&&intact.data.user?.id===other.id);
}catch(error){report.failure=error.name;process.exitCode=1;}
finally{
  if(paths.length){const removed=await admin.storage.from('wardrobe-images').remove(paths);report.cleanup.push({fixtureFilesRemoved:!removed.error});if(removed.error)process.exitCode=1;}
  for(const user of users){
    let removed;for(let attempt=0;attempt<3;attempt++){try{removed=await admin.auth.admin.deleteUser(user.id);}catch{removed={error:true};}if(!removed.error||removed.error.status===404)break;}
    const passed=!removed.error||removed.error.status===404;report.cleanup.push({temporaryAccountDeleted:passed});if(!passed)process.exitCode=1;
  }
  report.passed=report.checks.length===13&&report.checks.every(item=>item.passed)&&report.cleanup.every(item=>Object.values(item).every(Boolean))&&!report.failure;
  if(!report.passed)process.exitCode=1;
  await writeFile(new URL('../../docs/reports/readiness-evidence/production-wardrobe-media.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
  process.stdout.write(JSON.stringify(report)+'\n');
}
