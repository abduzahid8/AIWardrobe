/**
 * Server-only, consent- and SOURCE-VERIFIED persistent catalog photograph.
 *
 * A cached picture is NEVER a replacement SKU: the immutable current
 * product_id + original image_url MUST equal the provenance recorded by
 * an explicitly licensed off-request ingest. The caller keeps the original
 * merchant product item, source image and shopping URL unchanged.
 *
 * Missing, revoked, stale, mismatched, tampered or unreachable cached
 * imagery falls back to the ORIGINAL image path and its existing guards.
 */
import {createHash} from 'node:crypto';

export const CATALOG_PHOTO_CACHE_VERSION='s6-licensed-original-image-cache-v1';
export const CATALOG_PHOTO_MAX_BYTES=512*1024;
const SHA=/^[a-f0-9]{64}$/;
const VALID_PATH=/^licensed-stylist6\/([a-f0-9]{2}\/)?[a-f0-9]{64}\.jpg$/;
const FIRST_PARTY_HOST='fyqpifmrsftsfqibhwhy.supabase.co';
const MAX_VERIFICATION_AGE_MS=30*24*3600*1000;
const ID=/^[\w.:-]{1,200}$/;

function validOrigin(origin){
  try{
    const v=new URL(origin);
    return v.protocol==='https:'&&v.hostname===FIRST_PARTY_HOST&&
      !v.username&&!v.password&&!v.port&&v.pathname==='/'||(
        v.protocol==='https:'&&v.hostname===FIRST_PARTY_HOST&&
        !v.username&&!v.password&&!v.port&&v.pathname===''
      );
  }catch{return false;}
}
export function productPhotoSourceHash(productId,imageUrl){
  if(typeof productId!=='string'||!ID.test(productId)||
     typeof imageUrl!=='string'||imageUrl.length>2048)return null;
  try{
    const url=new URL(imageUrl);
    if(url.protocol!=='https:'||url.username||url.password||url.port||
       url.hash)return null;
  }catch{return null;}
  return createHash('sha256').update(JSON.stringify([productId,imageUrl]))
    .digest('hex');
}
export function verifiedCatalogPhotoCache(item,record,{supabaseOrigin,now=Date.now()}={}){
  const id=String(item?.id||'');
  const sourceUrl=item?.imageUrl||item?.image_url;
  const computedHash=productPhotoSourceHash(id,sourceUrl);
  if(!computedHash||!validOrigin(supabaseOrigin)||!record||
     record.product_id!==id||record.source_image_url!==sourceUrl||
     record.source_hash!==computedHash||
     record.status!=='ready'||record.storage_bucket!=='shop-catalog'||
     record.cache_hosting_granted!==true||
     record.external_ai_inference_granted!==true||
     typeof record.rights_reference!=='string'||record.rights_reference.length<8||
     !['owner_upload','licensed_provider','merchant_contract'].includes(record.rights_basis)||
     typeof record.storage_path!=='string'||!VALID_PATH.test(record.storage_path)||
     typeof record.image_sha256!=='string'||!SHA.test(record.image_sha256)||
     record.media_type!=='image/jpeg')return null;
  const verifiedAt=Date.parse(record.last_verified_at);
  const grantedAt=Date.parse(record.rights_granted_at);
  const expires=record.rights_expires_at==null?Infinity:Date.parse(record.rights_expires_at);
  if(!Number.isFinite(verifiedAt)||!Number.isFinite(grantedAt)||
     verifiedAt>now+60000||grantedAt>now+60000||
     verifiedAt<now-MAX_VERIFICATION_AGE_MS||now>=expires)return null;
  // Fixed origin and exclusively canonical bucket/path: no user URL
  // interpolation, redirect, signed private-bucket token or SSRF.
  const objectPath=record.storage_path.split('/').map(encodeURIComponent).join('/');
  const cachedUrl='https://'+FIRST_PARTY_HOST+
    '/storage/v1/object/public/shop-catalog/'+objectPath;
  return {cachedUrl,rawSha256:record.image_sha256,productId:id,
    sourceHash:computedHash};
}
async function readBoundedJpeg(response){
  if(!response?.ok||!response.body?.getReader)throw Error('media_unavailable');
  const header=Number(response.headers?.get?.('content-length'));
  if(Number.isFinite(header)&&header>CATALOG_PHOTO_MAX_BYTES)
    throw Error('cached_media_oversized');
  const reader=response.body.getReader(),chunks=[];
  let length=0;
  try{
    while(true){
      const {done,value}=await reader.read();
      if(done)break;
      if(!value?.byteLength)continue;
      length+=value.byteLength;
      if(length>CATALOG_PHOTO_MAX_BYTES)throw Error('cached_media_oversized');
      chunks.push(Buffer.from(value));
    }
  }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
  if(length<128)throw Error('cached_media_missing');
  const bytes=Buffer.concat(chunks,length);
  if(bytes[0]!==0xff||bytes[1]!==0xd8)
    throw Error('cached_media_type_mismatch');
  return bytes;
}
/**
 * ALWAYS recheck the exact-current source binding and downloaded original
 * image SHA. A failure degrades to the EXISTING original source resolver;
 * it never authorizes a product, invents an image, or skips the photo critic.
 */
export async function resolveCatalogCachedPhoto({
  item,record,supabaseOrigin,fetchImpl=fetch,decodeImage,
  originalResolver,now=Date.now(),deadlineMs=2300,
}={}){
  if(typeof originalResolver!=='function'||typeof decodeImage!=='function')
    throw Error('original_photo_resolver_required');
  const permitted=verifiedCatalogPhotoCache(item,record,{supabaseOrigin,now});
  if(!permitted)return originalResolver(item);
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),Math.min(4000,Math.max(400,deadlineMs)));
  try{
    const response=await fetchImpl(permitted.cachedUrl,{
      method:'GET',redirect:'error',signal:controller.signal,
      headers:{Accept:'image/jpeg','User-Agent':'AIWardrobe-Stylist6-Cloud/1.0'},
    });
    const bytes=await readBoundedJpeg(response);
    const sha256=createHash('sha256').update(bytes).digest('hex');
    if(sha256!==permitted.rawSha256)throw Error('cache_checksum_mismatch');
    const fitted=await decodeImage(bytes);
    if(!Buffer.isBuffer(fitted)||!fitted.length)throw Error('decode_failed');
    return {bytes:fitted,sha256:createHash('sha256').update(fitted).digest('hex')};
  }catch{
    return originalResolver(item);
  }finally{clearTimeout(timer);controller.abort();}
}
