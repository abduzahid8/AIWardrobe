import type {ImageIdentity} from './mediaImageSource';

function sameHeaders(a: unknown,b: unknown): boolean {
  if(a===b)return true;
  if(!a||!b||typeof a!=='object'||typeof b!=='object')return false;
  const left=a as Record<string,unknown>,right=b as Record<string,unknown>,keys=Object.keys(left);
  return keys.length===Object.keys(right).length&&keys.every(key=>Object.prototype.hasOwnProperty.call(right,key)&&left[key]===right[key]);
}
function sameResource(a: unknown,b: unknown): boolean {
  if(a===b)return true;
  if(Array.isArray(a)||Array.isArray(b))return Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((value,index)=>sameResource(value,b[index]));
  if(!a||!b||typeof a!=='object'||typeof b!=='object')return false;
  const left=a as Record<string,unknown>,right=b as Record<string,unknown>;
  return ['uri','cacheKey','blurhash','thumbhash'].every(key=>left[key]===right[key])&&sameHeaders(left.headers,right.headers);
}
export interface MediaImageLifecycle {
  source: unknown; userId?: string; accessToken?: string; recyclingKey?: string|null; version: number;
}
/** Opaque view version: never serialize credentials or large inline images into React/native keys. */
export function nextMediaImageLifecycle(previous:MediaImageLifecycle|undefined,source:unknown,identity:ImageIdentity,recyclingKey?:string|null):MediaImageLifecycle {
  if(previous&&previous.userId===identity.userId&&previous.accessToken===identity.accessToken
    &&previous.recyclingKey===recyclingKey&&sameResource(previous.source,source))return previous;
  return {source,userId:identity.userId,accessToken:identity.accessToken,recyclingKey,version:previous?previous.version+1:0};
}
