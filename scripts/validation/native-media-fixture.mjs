/** Disposable account/photo setup for manual native UI verification. Never writes a session into the app. */
import 'dotenv/config';
import {createClient} from '@supabase/supabase-js';
import {randomUUID,createHash} from 'node:crypto';
import {readFile,writeFile,unlink} from 'node:fs/promises';
import {transpileModule,ModuleKind} from 'typescript';
import {loadAppMediaForNode} from './app-media-node.mjs';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!key)throw new Error('Fixture setup credential required');
const file='/tmp/aiwardrobe-native-media-fixture.json';
const opts={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false}};
const admin=createClient(base,key,opts);
if(process.argv[2]==='setup'){
 try{await readFile(file);throw new Error('Fixture already exists; inspect and clean it before creating another');}catch(error){if(error.code!=='ENOENT')throw error;}
 const tag=randomUUID(),email=`readiness-${tag}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const made=await admin.auth.admin.createUser({email,password,email_confirm:true,user_metadata:{username:`readiness_${tag}`,readiness_test:true}});
 if(made.error||!made.data.user)throw new Error('Fixture account setup failed');
 const id=made.data.user.id,imageId=randomUUID(),path=`${id}/${imageId}/original.png`;
 // Record cleanup identity immediately, including partial setup failures.
 const fixture={id,email,password,imageId,path,createdAt:new Date().toISOString()};
 await writeFile(file,JSON.stringify(fixture),{mode:0o600,flag:'wx'});
 const client=createClient(base,anon,opts);
 const login=await client.auth.signInWithPassword({email,password});if(login.error)throw new Error('Fixture login failed');
 const events={};new Function('exports',transpileModule(await readFile(new URL('../../store/authEvents.ts',import.meta.url),'utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText)(events);
 events.authEvents.emitLogin(id);
 const app=await loadAppMediaForNode(client,events,base);
 const image=await readFile(new URL('../../assets/images/basic_outfit_1.png',import.meta.url));
 const imageUrl=await app.media.uploadWardrobeImage(id,imageId,`data:image/png;base64,${image.toString('base64')}`);
 const inserted=await client.from('clothing_items').insert({id:imageId,user_id:id,image_url:imageUrl,category:'top',sub_category:'t-shirt',type:'t-shirt',name:'Readiness native photo',primary_color:'Blue',color_hex:'#2563EB',pattern:'solid',material:'cotton',seasons:['summer'],occasions:['casual']});
 if(inserted.error)throw new Error('Fixture garment setup failed');
 await client.auth.signOut();
 fixture.imageBytes=image.length;fixture.imageSha256=createHash('sha256').update(image).digest('hex');
 await writeFile(file,JSON.stringify(fixture),{mode:0o600});
 console.log(JSON.stringify({fixtureReady:true,email,id,imageBytes:image.length}));
}else if(process.argv[2]==='cleanup'){
 const fixture=JSON.parse(await readFile(file,'utf8'));
 const user=await admin.auth.admin.getUserById(fixture.id);
 if(user.error||user.data.user?.user_metadata?.readiness_test!==true||user.data.user.email!==fixture.email)throw new Error('Fixture identity verification failed');
 const paths=[fixture.path,...(fixture.extraPaths||[])];
 if(paths.some(path=>typeof path!=='string'||!path.startsWith(`${fixture.id}/${fixture.imageId}/`)))throw Error('Invalid fixture cleanup path');
 const removed=await admin.storage.from('wardrobe-images').remove(paths);if(removed.error)throw new Error('Fixture file cleanup failed');
 const deleted=await admin.auth.admin.deleteUser(fixture.id);if(deleted.error)throw new Error('Fixture account cleanup failed');
 await unlink(file);console.log(JSON.stringify({fixtureFileRemoved:true,temporaryAccountDeleted:true,localFixtureManifestRemoved:true}));
}else throw new Error('Choose setup or cleanup');
