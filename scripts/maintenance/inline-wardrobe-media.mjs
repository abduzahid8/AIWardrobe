/** Migrate one explicitly supplied row. No bulk scan or implicit production apply. */
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {decode} from 'base64-arraybuffer';
import {transpileModule,ModuleKind} from 'typescript';
const uuid=/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
const checksum=bytes=>createHash('sha256').update(Buffer.from(bytes)).digest('hex');
let reader;
async function readInput(uri){
 if(!reader){
  const deadlineSource=await readFile(new URL('../../src/lib/requestDeadline.ts',import.meta.url),'utf8');
  const deadline={};new Function('exports',transpileModule(deadlineSource,{compilerOptions:{module:ModuleKind.CommonJS}}).outputText)(deadline);
  const source=await readFile(new URL('../../src/services/wardrobeMedia.ts',import.meta.url),'utf8');
  const code=transpileModule(source,{compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
  const api={};new Function('exports','require',code)(api,name=>{
   if(name==='base64-arraybuffer')return {decode};
   if(name==='../../lib/supabase')return {supabase:{}};
   if(name==='../../store/authEvents')return {authEvents:{}};
   if(name==='../config/env')return {Config:{supabase:{url:'https://unused.invalid'}}};
   if(name==='../lib/requestDeadline')return deadline;
   throw Error('Unexpected media dependency');
  });reader=api.readWardrobeImageInput;
 }
 return reader(uri);
}
export async function planInlineMedia(row,baseUrl){
 if(!uuid.test(row?.id)||!uuid.test(row?.user_id)||!/^data:/i.test(row?.image_url||''))throw Error('Invalid migration row');
 if(!Object.hasOwn(row,'updated_at') || (row.updated_at!==null && (typeof row.updated_at!=='string'||!Number.isFinite(Date.parse(row.updated_at)))))throw Error('Missing concurrency version');
 const base=new URL(baseUrl);if(base.protocol!=='https:'||base.username||base.password)throw Error('Invalid project origin');
 const input=await readInput(row.image_url);
 const digest=checksum(input.bytes),path=`${row.user_id}/${row.id}/legacy-original-${digest}.${input.extension}`;
 return {...input,digest,path,uri:`${base.origin}/storage/v1/object/authenticated/wardrobe-images/${path}`};
}
export async function migrateInlineMedia(admin,row,baseUrl){
 const plan=await planInlineMedia(row,baseUrl),bucket=admin.storage.from('wardrobe-images');
 const uploaded=await bucket.upload(plan.path,plan.bytes,{contentType:plan.mimeType,upsert:false});
 if(uploaded.error&&!['409','Duplicate'].includes(String(uploaded.error.statusCode??uploaded.error.error)))throw Error('MigrationUploadFailed');
 // Verify the stored bytes before replacing the only database reference.
 const downloaded=await bucket.download(plan.path);
 if(downloaded.error||!downloaded.data)throw Error('MigrationReadbackFailed');
 const stored=await downloaded.data.arrayBuffer();
 if(stored.byteLength!==plan.bytes.byteLength||checksum(stored)!==plan.digest)throw Error('MigrationChecksumMismatch');
 // Production BEFORE UPDATE trigger refreshes updated_at for every change.
 // A competing edit must win; never overwrite it using a stale snapshot.
 let update=admin.from('clothing_items').update({image_url:plan.uri}).eq('id',row.id).eq('user_id',row.user_id);
 update=row.updated_at===null?update.is('updated_at',null):update.eq('updated_at',row.updated_at);
 const result=await update.select('id,image_url,updated_at');
 if(result.error)throw Error('MigrationUpdateFailed');
 if(result.data?.length!==1)return {status:'conflict',path:plan.path,digest:plan.digest,bytes:plan.bytes.byteLength};
 if(result.data[0].id!==row.id||result.data[0].image_url!==plan.uri)throw Error('MigrationConfirmationFailed');
 return {status:'migrated',path:plan.path,uri:plan.uri,digest:plan.digest,bytes:plan.bytes.byteLength,mimeType:plan.mimeType,updatedAt:result.data[0].updated_at};
}


/** Restore verified bytes, guarded against edits made after the supplied row snapshot. */
export async function restoreInlineMedia(admin,row,receipt,baseUrl){
 const extensions={'image/png':'png','image/jpeg':'jpg','image/webp':'webp','image/heic':'heic','image/heif':'heif'};
 const ext=extensions[receipt?.mimeType],base=new URL(baseUrl);
 if(base.protocol!=='https:'||base.username||base.password||!uuid.test(row?.id)||!uuid.test(row?.user_id)
   ||!ext||!/^[a-f0-9]{64}$/.test(receipt?.digest||'')||receipt.status!=='migrated'
   ||!Number.isSafeInteger(receipt.bytes)||receipt.bytes<1||receipt.bytes>10*1024*1024)throw Error('Invalid rollback receipt');
 const path=`${row.user_id}/${row.id}/legacy-original-${receipt.digest}.${ext}`;
 const uri=`${base.origin}/storage/v1/object/authenticated/wardrobe-images/${path}`;
 if(receipt.path!==path||receipt.uri!==uri||row.image_url!==uri)throw Error('Rollback reference mismatch');
 if(!Object.hasOwn(row,'updated_at')||(row.updated_at!==null&&(typeof row.updated_at!=='string'||!Number.isFinite(Date.parse(row.updated_at)))))throw Error('Missing concurrency version');
 const downloaded=await admin.storage.from('wardrobe-images').download(path);
 if(downloaded.error||!downloaded.data)throw Error('RollbackDownloadFailed');
 const bytes=await downloaded.data.arrayBuffer();
 if(bytes.byteLength!==receipt.bytes||checksum(bytes)!==receipt.digest)throw Error('RollbackChecksumMismatch');
 const inline=`data:${receipt.mimeType};base64,${Buffer.from(bytes).toString('base64')}`;
 await readInput(inline);
 let update=admin.from('clothing_items').update({image_url:inline}).eq('id',row.id).eq('user_id',row.user_id).eq('image_url',uri);
 update=row.updated_at===null?update.is('updated_at',null):update.eq('updated_at',row.updated_at);
 const result=await update.select('id,image_url');
 if(result.error)throw Error('RollbackUpdateFailed');
 if(result.data?.length!==1)return {status:'conflict'};
 const confirmed=await readInput(result.data[0].image_url);
 if(result.data[0].id!==row.id||checksum(confirmed.bytes)!==receipt.digest)throw Error('RollbackConfirmationFailed');
 return {status:'restored',digest:receipt.digest,bytes:receipt.bytes};
}
