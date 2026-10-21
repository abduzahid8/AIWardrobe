import test from 'node:test';import assert from 'node:assert/strict';
import {migrateInlineMedia,restoreInlineMedia} from './inline-wardrobe-media.mjs';
const row={id:'11111111-1111-4111-8111-111111111111',user_id:'22222222-2222-4222-8222-222222222222',image_url:'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQkAAAAASUVORK5CYII=',updated_at:'2026-10-20T00:00:00Z'};
function fixture({corrupt=false,conflict=false,uploadError=false}={}){
 let bytes,patched=false;const filters=[];
 const query={eq:(key,value)=>{filters.push([key,value]);return query;},is:(key,value)=>{filters.push([key,value]);return query;},select:async()=>({data:conflict?[]:[{id:row.id,image_url:query.values.image_url}]})};
 const admin={storage:{from:()=>({upload:async(_path,input,options)=>{assert.equal(options.upsert,false);bytes=input;return {error:uploadError?{statusCode:'403'}:null};},download:async()=>({data:new Blob([corrupt?new Uint8Array([1]):bytes])})})},from:()=>({update:values=>{patched=true;query.values=values;return query;}})};
 return {admin,filters,setConflict(){conflict=true;},setCorrupt(){corrupt=true;},get patched(){return patched;}};
}
test('copies exact bytes then conditionally replaces only the image reference',async()=>{
 const f=fixture(),r=await migrateInlineMedia(f.admin,row,'https://project.invalid');assert.equal(r.status,'migrated');assert.ok(r.uri.includes('/authenticated/wardrobe-images/'));assert.equal(r.digest.length,64);assert.deepEqual(f.filters,[['id',row.id],['user_id',row.user_id],['updated_at',row.updated_at]]);
});
test('checksum failure never patches the row',async()=>{const f=fixture({corrupt:true});await assert.rejects(migrateInlineMedia(f.admin,row,'https://project.invalid'),/Checksum/);assert.equal(f.patched,false);});
test('failed upload never patches the row',async()=>{const f=fixture({uploadError:true});await assert.rejects(migrateInlineMedia(f.admin,row,'https://project.invalid'),/Upload/);assert.equal(f.patched,false);});
test('competing edits produce conflict instead of success',async()=>{const f=fixture({conflict:true});assert.equal((await migrateInlineMedia(f.admin,row,'https://project.invalid')).status,'conflict');});
test('invalid source bytes never upload or patch',async()=>{const f=fixture();await assert.rejects(migrateInlineMedia(f.admin,{...row,image_url:'data:image/png;base64,YmFk'},'https://project.invalid'));assert.equal(f.patched,false);});

test('requires a valid row version before uploading',async()=>{const f=fixture();await assert.rejects(migrateInlineMedia(f.admin,{...row,updated_at:undefined},'https://project.invalid'),/concurrency/);assert.equal(f.patched,false);});

test('rollback restores original bytes and requires current image and row version',async()=>{
 const f=fixture(),receipt=await migrateInlineMedia(f.admin,row,'https://project.invalid');
 const r=await restoreInlineMedia(f.admin,{...row,image_url:receipt.uri},receipt,'https://project.invalid');assert.equal(r.status,'restored');assert.equal(r.digest,receipt.digest);
 assert.ok(f.filters.some(([field,value])=>field==='image_url'&&value===receipt.uri));
});
test('rollback rejects mismatched owner or project receipts',async()=>{
 const f=fixture(),receipt=await migrateInlineMedia(f.admin,row,'https://project.invalid');
 await assert.rejects(restoreInlineMedia(f.admin,{...row,image_url:receipt.uri},{...receipt,path:'different-owner'},'https://project.invalid'),/mismatch/);
 await assert.rejects(restoreInlineMedia(f.admin,{...row,image_url:receipt.uri},receipt,'https://other.invalid'),/mismatch/);
});
test('rollback refuses corrupted backup bytes',async()=>{
 const f=fixture(),receipt=await migrateInlineMedia(f.admin,row,'https://project.invalid');f.setCorrupt();
 await assert.rejects(restoreInlineMedia(f.admin,{...row,image_url:receipt.uri},receipt,'https://project.invalid'),/Checksum/);
});
test('rollback does not report success when a competing edit wins',async()=>{
 const f=fixture(),receipt=await migrateInlineMedia(f.admin,row,'https://project.invalid');f.setConflict();
 assert.equal((await restoreInlineMedia(f.admin,{...row,image_url:receipt.uri},receipt,'https://project.invalid')).status,'conflict');
});
