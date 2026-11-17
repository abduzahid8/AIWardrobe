/**
 * Offline, ADMIN-ONLY and EXPLICITLY LICENSED merchandise photo ingestion.
 *
 * Never run during an authenticated styling request. No implicit merchant
 * copyright/hosting permission: the administrator MUST provide independent
 * per-SKU rights evidence in a manifest; a reachable image is NOT a license.
 *
 * --manifest /secure/rights-grants.json [--apply]
 * Default is DRY RUN. --apply uses SUPABASE_SERVICE_ROLE_KEY on the admin
 * host only; no key is ever embedded in the mobile or server response.
 * Output is source-bounded and content-addressed, never replaces the
 * original catalog image_url, price, source_url or product identity.
 */
import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import sharp from 'sharp';
import {HOST_ALLOWLIST,isInferenceAuthorized} from './image_access.mjs';
import {productPhotoSourceHash,CATALOG_PHOTO_MAX_BYTES} from
  '../../backend/stylist6/catalog_photo_cache.mjs';

const MAX_ORIGINAL_BYTES=4*1024*1024;
export const LICENSED_IMAGE_INGEST_VERSION='s6-admin-licensed-photo-ingest-v1';
const GRANT_BASES=new Set(['owner_upload','licensed_provider','merchant_contract']);
function validHost(url){
  try{
    const parsed=new URL(url);
    return parsed.protocol==='https:'&&!parsed.username&&!parsed.password&&!parsed.port&&
      HOST_ALLOWLIST.has(parsed.hostname)&&isInferenceAuthorized(url)&&
      !parsed.hash;
  }catch{return false;}
}
export function checkPhotoGrant(grant,now=Date.now()){
  if(!grant||typeof grant!=='object'||!grant.productId||
     typeof grant.sourceImageUrl!=='string'||!validHost(grant.sourceImageUrl)||
     !GRANT_BASES.has(grant.rightsBasis)||
     typeof grant.rightsReference!=='string'||grant.rightsReference.length<8||
     grant.hostingPermission!==true||grant.externalAiInferencePermission!==true)
    return {ok:false,reason:'license_or_source_missing'};
  const sourceHash=productPhotoSourceHash(grant.productId,grant.sourceImageUrl);
  const grantedAt=Date.parse(grant.rightsGrantedAt);
  const expires=grant.rightsExpiresAt==null?Infinity:Date.parse(grant.rightsExpiresAt);
  if(!sourceHash||!Number.isFinite(grantedAt)||grantedAt>now+60000||
     now>=expires||expires<=grantedAt)
    return {ok:false,reason:'rights_dates_or_source_invalid'};
  return {ok:true,sourceHash};
}
async function boundedSourcePhoto(url,fetchImpl){
  const response=await fetchImpl(url,{
    method:'GET',redirect:'error',signal:AbortSignal.timeout(6000),
    headers:{Accept:'image/*','User-Agent':'AIWardrobe-Stylist6-Cloud/1.0'},
  });
  if(!response?.ok||!response.body?.getReader)throw Error('source_unavailable');
  const len=Number(response.headers?.get?.('content-length'));
  if(Number.isFinite(len)&&len>MAX_ORIGINAL_BYTES)throw Error('source_oversize');
  const reader=response.body.getReader();
  let bytes=0;const chunks=[];
  try{
    while(true){
      const {done,value}=await reader.read();
      if(done)break;
      if(!value?.byteLength)continue;
      bytes+=value.byteLength;
      if(bytes>MAX_ORIGINAL_BYTES)throw Error('source_oversize');
      chunks.push(Buffer.from(value));
    }
  }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
  if(bytes<1000)throw Error('source_media_invalid');
  return Buffer.concat(chunks,bytes);
}
export async function ingestLicensedProductPhoto(grant,{
  fetchImpl=fetch,loadProduct,uploadObject,saveProvenance,
  now=Date.now(),apply=false,
}={}){
  const approval=checkPhotoGrant(grant,now);
  if(!approval.ok)return {status:'rejected',reason:approval.reason};
  if(typeof loadProduct!=='function'||typeof uploadObject!=='function'||
     typeof saveProvenance!=='function')throw Error('admin_providers_required');
  const row=await loadProduct(grant.productId);
  if(!row||row.is_active!==true||row.id!==grant.productId||
     row.image_url!==grant.sourceImageUrl||
     typeof row.source_url!=='string'||!validHost(row.image_url)||
     !/^https:\/\//.test(row.source_url)||
     !(Number(row.price)>0)||!/^[A-Z]{3}$/.test(row.currency||''))
    return {status:'rejected',reason:'source_catalog_drift_or_invalid'};
  if(!apply)return {status:'dry_run',productId:row.id,
    sourceHash:approval.sourceHash};
  const raw=await boundedSourcePhoto(row.image_url,fetchImpl);
  const metadata=await sharp(raw,{limitInputPixels:4_200_000,
    failOn:'warning'}).metadata();
  if(!metadata?.width||!metadata?.height||
     metadata.width<180||metadata.height<180||
     metadata.pages>1)throw Error('source_media_invalid');
  const jpeg=await sharp(raw,{limitInputPixels:4_200_000})
    .rotate().resize(760,760,{fit:'inside',withoutEnlargement:true})
    .jpeg({quality:76,chromaSubsampling:'4:4:4'}).toBuffer();
  if(jpeg.length<128||jpeg.length>CATALOG_PHOTO_MAX_BYTES)
    throw Error('optimized_photo_unusable');
  const sha=createHash('sha256').update(jpeg).digest('hex');
  // Product/source-addressed object path prevents two variants with
  // identical photograph bytes from fighting over a UNIQUE storage path.
  // The separate image_sha256 still authenticates exact downloaded bytes.
  const storagePath='licensed-stylist6/'+approval.sourceHash.slice(0,2)+
    '/'+approval.sourceHash+'.jpg';
  const uploaded=await uploadObject(storagePath,jpeg);
  if(uploaded!==true)throw Error('storage_upload_failed');
  // Exact original record is reloaded before marking ready. This avoids
  // publishing an image for an SKU edited during this slow network request.
  const current=await loadProduct(row.id);
  if(!current||current.is_active!==true||
     current.image_url!==row.image_url||
     current.id!==row.id)throw Error('source_changed_during_ingest');
  await saveProvenance({
    product_id:row.id,source_image_url:row.image_url,
    source_hash:approval.sourceHash,storage_bucket:'shop-catalog',
    storage_path:storagePath,image_sha256:sha,media_type:'image/jpeg',
    status:'ready',rights_basis:grant.rightsBasis,
    rights_reference:grant.rightsReference,
    cache_hosting_granted:true,external_ai_inference_granted:true,
    rights_granted_at:new Date(grant.rightsGrantedAt).toISOString(),
    rights_expires_at:grant.rightsExpiresAt?new Date(grant.rightsExpiresAt).toISOString():null,
    last_verified_at:new Date(now).toISOString(),
  });
  return {status:'ingested',productId:row.id,storagePath,sha256:sha};
}
async function main(){
  const argv=process.argv.slice(2);
  const index=argv.indexOf('--manifest');
  if(index===-1||!argv[index+1])throw Error('usage: --manifest /secure/grants.json [--apply]');
  const apply=argv.includes('--apply');
  const grants=JSON.parse(await fs.readFile(argv[index+1],'utf8'));
  if(!Array.isArray(grants)||grants.length<1||grants.length>30)
    throw Error('one_to_thirty_approved_grants_required');
  const secrets=await import('dotenv');
  secrets.config({path:process.env.AIW_STYLIST_SECRET_ENV||'.env'});
  const url=process.env.SUPABASE_URL;
  const key=process.env.SUPABASE_SERVICE_ROLE_KEY;
  if(!url||!key)throw Error('server_credentials_required');
  const {createClient}=await import('@supabase/supabase-js');
  const admin=createClient(url,key,{
    auth:{persistSession:false,autoRefreshToken:false}});
  const loadProduct=async id=>{
    const {data,error}=await admin.from('shop_catalog')
      .select('id,image_url,source_url,is_active,price,currency')
      .eq('id',id).maybeSingle();
    if(error)throw Error('source_catalog_read_failed');
    return data;
  };
  const uploadObject=async(path,bytes)=>{
    const {error}=await admin.storage.from('shop-catalog').upload(path,bytes,{
      contentType:'image/jpeg',cacheControl:'86400',upsert:false});
    if(error)throw Error('storage_upload_failed');
    return true;
  };
  const saveProvenance=async data=>{
    const {error}=await admin.from('catalog_photo_cache_v1').upsert(data,
      {onConflict:'product_id'});
    if(error)throw Error('rights_provenance_write_failed');
  };
  const report=[];
  for(const grant of grants){
    try{report.push(await ingestLicensedProductPhoto(grant,{
      loadProduct,uploadObject,saveProvenance,apply,
    }));}
    catch(error){report.push({status:'failed',reason:String(error.message).slice(0,65)});}
  }
  const summary={mode:apply?'apply':'dry-run',items:report.length,
    statuses:report.reduce((acc,x)=>(acc[x.status]=(acc[x.status]||0)+1,acc),{})};
  process.stdout.write(JSON.stringify(summary)+'\n');
  if(report.some(r=>r.status==='failed'))process.exitCode=1;
}
if(process.argv[1]&&fileURLToPath(import.meta.url)===process.argv[1])
  main().catch(error=>{process.stderr.write(String(error.message)+'\n');process.exitCode=1;});
