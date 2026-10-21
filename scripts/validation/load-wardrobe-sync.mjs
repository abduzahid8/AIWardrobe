/** Execute real sync/media modules with a real SDK; native filesystem unused for inline fixtures. */
import {readFile} from 'node:fs/promises';
import {transpileModule,ModuleKind,ScriptTarget} from 'typescript';
import {encode,decode} from 'base64-arraybuffer';
import {createHash} from 'node:crypto';
export async function loadWardrobeSync(client,base,owner){
 const modules={},hashes={};
 async function load(key,file){
  const source=await readFile(new URL(`../../${file}`,import.meta.url),'utf8');
  hashes[file]=createHash('sha256').update(source).digest('hex');
  const code=transpileModule(source,{compilerOptions:{module:ModuleKind.CommonJS,target:ScriptTarget.ES2022}}).outputText;
  const api={};new Function('exports','require',code)(api,name=>{
   if(name==='base64-arraybuffer')return {encode,decode};
   if(name==='expo-file-system/legacy')return {documentDirectory:null};
   if(name.endsWith('lib/supabase'))return {supabase:client};
   if(name.endsWith('config/env'))return {Config:{supabase:{url:base}}};
   for(const dependency of ['requestDeadline','authEvents','wardrobeMedia','pendingWardrobeMedia','wardrobeItemMedia'])if(name.endsWith('/'+dependency)&&modules[dependency])return modules[dependency];
   throw Error('Unsupported sync fixture dependency');
  });modules[key]=api;
 }
 await load('authEvents','store/authEvents.ts');modules.authEvents.authEvents.emitLogin(owner);
 await load('requestDeadline','src/lib/requestDeadline.ts');
 await load('wardrobeMedia','src/services/wardrobeMedia.ts');
 await load('pendingWardrobeMedia','src/services/pendingWardrobeMedia.ts');
 await load('wardrobeItemMedia','src/services/wardrobeItemMedia.ts');
 await load('wardrobeSyncService','store/wardrobeSyncService.ts');
 return {...modules.wardrobeSyncService,media:modules.wardrobeMedia,hashes};
}
