/** Operational cloud release with only source, authorization, schema and rollback guards. */
import fs from 'node:fs';import path from 'node:path';import cp from 'node:child_process';import crypto from 'node:crypto';import assert from 'node:assert/strict';import esbuild from 'esbuild';import ts from 'typescript';
import {management,query,quote} from '../../stylist5/private-api.mjs';
const root=path.resolve(import.meta.dirname,'../../..'),dir=process.env.STYLIST5_PRIVATE_DIR;
if(!dir)throw Error('private release directory required');
const entry='backend/supabase/functions/generate-outfits/index.ts',control='backend/supabase/functions/catalog-visual-worker/index.ts';
const base='6fe2a2dff6202df38d33b802b0f7eb44a124eaf7',oldSource='6dc1c4c40e96e954685b4e27ec5803462c3308060a1835be18ac78a9884d425d',oldFingerprint='96fc30d5b45dd16db43064f9449fb53adf4b69d1ea353f283c79e14a6e292748';
const migrations=['backend/supabase/migrations/20261108040000_stylist5_editorial.sql','backend/supabase/migrations/20261108120000_stylist5_cloud_visual_queue.sql','backend/supabase/migrations/20261108121000_stylist5_source_taxonomy.sql','backend/supabase/migrations/20261108122000_stylist5_capsule_priority.sql','backend/supabase/migrations/20261108123000_stylist5_queue_qualification.sql','backend/supabase/migrations/20261108124000_stylist5_expired_worker_leases.sql','backend/supabase/migrations/20261108125000_stylist5_top_level_lease.sql'];
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),save=(name,v)=>fs.writeFileSync(dir+'/'+name,JSON.stringify(v,null,2),{mode:0o600}),read=name=>JSON.parse(fs.readFileSync(dir+'/'+name));
const git=(...args)=>cp.execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
async function source(slug){const b=Buffer.from(await(await management('/functions/'+slug+'/body')).arrayBuffer()),key=Buffer.from('"sourcesContent":['),offset=b.indexOf(key);assert(offset>=0,'deployed source missing');const a=offset+key.length;let e=a+1;for(;e<b.length;e++){if(b[e]===92){e++;continue}if(b[e]===34)break}return JSON.parse(b.subarray(a,e+1).toString())}
async function live(){const all=await(await management('/functions')).json();return all.find(f=>f.slug==='generate-outfits')}
async function baseline(){const f=await live();const pinned=fs.existsSync(dir+'/runtime-baseline.json')?read('runtime-baseline.json'):{version:200};assert.equal(f.version,pinned.version,'baseline version drift');if(pinned.verify_jwt!==undefined)assert.equal(f.verify_jwt,pinned.verify_jwt);assert.equal(f.status,'ACTIVE');assert.equal(f.ezbr_sha256,oldFingerprint);assert.equal(hash(await source('generate-outfits')),oldSource);return f;}
function sourceGuards(){
 const declarations=s=>{const tree=ts.createSourceFile('entry.ts',s,ts.ScriptTarget.Latest,true),out=new Map();function visit(n){if(ts.isFunctionDeclaration(n)&&n.name)out.set(n.name.text,n.getText(tree));ts.forEachChild(n,visit)}visit(tree);return out};
 const before=declarations(git('show',base+':'+entry)),s=fs.readFileSync(root+'/'+entry,'utf8'),after=declarations(s);
 for(const key of ['shopPriceOf','outfitShopSpend','validateOutfitAgainstStyle','validateOutfitCompatibility','convertShopAmountEdge','budgetInCurrencyEdge']){assert(before.has(key)&&after.has(key));assert.equal(after.get(key),before.get(key),'protected source changed '+key)}
 for(const file of ['src/domain/budgetFx.ts','backend/supabase/functions/_shared/rateLimit.ts'])assert.equal(fs.readFileSync(root+'/'+file,'utf8').trim(),git('show',base+':'+file),'protected source changed');
 for(const guard of ["if (!/^Bearer\\s+\\S+$/i.test(authHeader.trim())) return authReply(401)","if (!user?.id) return authReply(401)",".eq('user_id', requestUserId)"])assert(s.includes(guard),'auth/ownership source guard missing');
 assert.equal(hash(fs.readFileSync(root+'/.v199-balanced-build/v200.ts')),oldSource);assert.equal(hash(fs.readFileSync(dir+'/v200.ts')),oldSource);
}
async function bundle(file){return(await esbuild.build({absWorkingDir:root,entryPoints:[file],bundle:true,format:'esm',platform:'neutral',target:'es2022',external:['https://*'],write:false,metafile:true,logLevel:'silent'}));}
async function security(cloud=false){const extra=cloud?",'public.catalog_visual_jobs'::regclass,'public.catalog_visual_worker_config'::regclass,'public.catalog_stylist_capsule'::regclass,'public.stylist5_taxonomy_rollback'::regclass":'';
 const r=await query(`select jsonb_build_object('rls',not exists(select 1 from pg_class where oid in ('public.shop_catalog'::regclass,'public.saved_outfits'::regclass,'public.catalog_visual_fashion'::regclass,'public.catalog_image_embeddings'::regclass,'public.stylist_look_feedback'::regclass${extra}) and not relrowsecurity),'privateEmbeddings',not has_table_privilege('authenticated','public.catalog_image_embeddings','SELECT'),'serviceOnly',not has_function_privilege('authenticated','public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)','EXECUTE'),'invoker',not(select prosecdef from pg_proc where oid='public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)'::regprocedure),'feedbackRls',(select count(*)=4 from pg_policies where schemaname='public' and tablename='stylist_look_feedback')) state`);
 assert(Object.values(r[0].state).every(v=>v===true),'database security guard failed');
 if(cloud){const x=await query("select not has_function_privilege('authenticated','public.catalog_visual_batch(text,uuid,jsonb)','EXECUTE') and not has_function_privilege('anon','public.catalog_visual_batch(text,uuid,jsonb)','EXECUTE') and not has_table_privilege('authenticated','public.catalog_visual_jobs','SELECT') and not has_table_privilege('authenticated','public.catalog_stylist_capsule','INSERT') and not(select prosecdef from pg_proc where oid='public.catalog_visual_batch(text,uuid,jsonb)'::regprocedure) private");assert(x[0].private,'worker queue privilege guard failed')}
 return r[0].state;
}
async function deploy(slug,bytes,jwt){const body=new FormData();body.append('metadata',JSON.stringify({name:slug,entrypoint_path:'index.ts',verify_jwt:jwt}));body.append('file',new Blob([bytes],{type:'application/typescript'}),'index.ts');return(await management('/functions/deploy?slug='+slug,{method:'POST',body})).json()}
const mode=process.argv[2];sourceGuards();
if(mode==='prepare'){
 await baseline();await security();const b=await bundle(entry),c=await bundle(control);
 const manifest={head:git('rev-parse','HEAD'),baselineSource:oldSource,generateSha:hash(b.outputFiles[0].contents),controlSha:hash(c.outputFiles[0].contents),inputs:[...new Set([...Object.keys(b.metafile.inputs),...Object.keys(c.metafile.inputs)])].map(p=>({path:p,sha:hash(fs.readFileSync(root+'/'+p))})),migrations:migrations.map(p=>({path:p,sha:hash(fs.readFileSync(root+'/'+p))}))};
 fs.writeFileSync(dir+'/candidate.ts',b.outputFiles[0].contents,{mode:0o600});fs.writeFileSync(dir+'/controller.ts',c.outputFiles[0].contents,{mode:0o600});save('manifest.json',manifest);console.log({sourceIntegrityPrepared:true,baseline:200,candidateSha256:manifest.generateSha});
}else if(mode==='migrate'){
 const m=read('manifest.json');await baseline();await security();
 for(const file of m.migrations){assert.equal(hash(fs.readFileSync(root+'/'+file.path)),file.sha);assert.equal(hash(cp.execFileSync('git',['show','HEAD:'+file.path],{cwd:root})),file.sha,'uncommitted migration')}
 const integrity="select md5(string_agg(md5(to_jsonb(s)::text),'' order by id)) digest from public.shop_catalog s";
 let sql=`begin isolation level repeatable read;set local lock_timeout='2s';create temporary table source_before as ${integrity};`;
 for(const file of m.migrations){const version=path.basename(file.path).split('_')[0];const exists=await query(`select count(*) n from supabase_migrations.schema_migrations where version=${quote(version)}`);if(Number(exists[0].n)!==0){const prior=read('migration-applied.json').migrations.find(p=>p.path===file.path);assert(prior&&prior.sha===file.sha,'recorded migration drift');continue;}sql+=fs.readFileSync(root+'/'+file.path,'utf8').replace(/^begin;$/m,'').replace(/^commit;$/m,'');sql+=`insert into supabase_migrations.schema_migrations(version,name,statements) values(${quote(version)},${quote(path.basename(file.path,'.sql'))},array['Guarded Stylist 5 source-safe activation']);`;}
 sql+=`do $$begin if(select digest from source_before) is distinct from (${integrity}) then raise exception 'authoritative_catalog_changed';end if;end $$;commit;`;
 await query(sql);await security(true);save('migration-applied.json',{at:new Date().toISOString(),migrations:m.migrations,sourceCatalogUnchanged:true});console.log({migrationsApplied:true,sourceCatalogUnchanged:true,security:await security(true)});
}else if(mode==='controller'){
 const m=read('manifest.json'),bytes=fs.readFileSync(dir+'/controller.ts');assert.equal(hash(bytes),m.controlSha);await security(true);
 const f=await deploy('catalog-visual-worker',bytes,false);assert.equal(hash(await source('catalog-visual-worker')),m.controlSha);save('controller-deployment.json',{version:f.version,sha:m.controlSha});console.log({cloudControlPlaneDeployed:true,version:f.version});
}else if(mode==='deploy'){
 const m=read('manifest.json');assert.equal(git('rev-parse','HEAD'),m.head,'commit drift');const branch=git('branch','--show-current');assert.equal(git('-c','credential.helper=osxkeychain','ls-remote','--heads','origin',branch).split(/\s/)[0],m.head,'push exact source first');
 for(const file of m.inputs){assert.equal(hash(fs.readFileSync(root+'/'+file.path)),file.sha);assert.equal(hash(cp.execFileSync('git',['show','HEAD:'+file.path],{cwd:root})),file.sha)}
 const old=await baseline();await security(true);assert(read('migration-applied.json').sourceCatalogUnchanged);const bytes=fs.readFileSync(dir+'/candidate.ts');assert.equal(hash(bytes),m.generateSha);
 assert(!fs.existsSync(dir+'/edge-deploy-attempt.json'),'deployment already attempted');save('edge-deploy-attempt.json',{head:m.head,at:new Date().toISOString()});
 const deployed=await deploy('generate-outfits',bytes,old.verify_jwt);assert.equal(hash(await source('generate-outfits')),m.generateSha);const f=await live();assert.equal(f.version,deployed.version);assert.equal(f.status,'ACTIVE');assert.equal(f.verify_jwt,old.verify_jwt);
 for(const token of [null,'invalid-stylist-release-auth']){const r=await fetch('https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/generate-outfits',{method:'POST',headers:{'Content-Type':'application/json',...(token?{Authorization:'Bearer '+token}:{})},body:'{}',signal:AbortSignal.timeout(20000)});await r.text();assert.equal(r.status,401,'authentication guard failed')}
 save('deployment.json',{version:f.version,status:f.status,sha:m.generateSha,commit:m.head,at:new Date().toISOString(),security:await security(true)});console.log({activeEdgeVersion:f.version,exactSourceConfirmed:true,authenticationPreserved:true,rollbackV200Preserved:true,commit:m.head});
}else if(mode==='rollback'){
 const m=read('manifest.json'),deployment=read('deployment.json'),f=await live();assert.equal(f.version,deployment.version);assert.equal(hash(await source('generate-outfits')),m.generateSha);
 await query('update public.catalog_visual_worker_config set enabled=false where singleton');
 const prior=JSON.parse(fs.readFileSync(dir+'/visual-v200.json')),sql=fs.readFileSync(dir+'/retrieval-v200.sql','utf8');
 await query(`begin;${sql}
  delete from public.catalog_visual_fashion;
  insert into public.catalog_visual_fashion select * from jsonb_populate_recordset(null::public.catalog_visual_fashion,${quote(JSON.stringify(prior))}::jsonb);
  delete from public.catalog_garment_intelligence where product_id in(select product_id from public.stylist5_taxonomy_rollback);
  insert into public.catalog_garment_intelligence select restored.* from public.stylist5_taxonomy_rollback b cross join lateral jsonb_populate_record(null::public.catalog_garment_intelligence,b.original) restored;
  commit;`);
 const restored=await deploy('generate-outfits',fs.readFileSync(dir+'/v200.ts'),f.verify_jwt);assert.equal(hash(await source('generate-outfits')),oldSource);save('rollback.json',{version:restored.version,restoredSource:oldSource,workerPaused:true});console.log({exactV200SourceRestored:true,version:restored.version});
}else throw Error('unsupported mode');
