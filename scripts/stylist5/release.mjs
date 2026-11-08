/** One guarded STYLIST 5 release. prepare is read-only; release is single-use.
 * Exact v200 code + prior retrieval and visual cache are preserved for rollback.
 * All credentials and raw catalog evidence remain outside the repository. */
import fs from 'node:fs';import path from 'node:path';import cp from 'node:child_process';import crypto from 'node:crypto';import assert from 'node:assert/strict';import esbuild from 'esbuild';import ts from 'typescript';
import {management,query,quote} from './private-api.mjs';
const root=path.resolve(import.meta.dirname,'../..'),dir=process.env.STYLIST5_PRIVATE_DIR;
if(!dir)throw Error('Set STYLIST5_PRIVATE_DIR to the private benchmark/release directory');
const report=path.join(root,'docs/reports/stylist-5'),entry='backend/supabase/functions/generate-outfits/index.ts',migration='backend/supabase/migrations/20261108040000_stylist5_editorial.sql';
const BASE='6fe2a2dff6202df38d33b802b0f7eb44a124eaf7',FINGERPRINT='96fc30d5b45dd16db43064f9449fb53adf4b69d1ea353f283c79e14a6e292748',SOURCE='6dc1c4c40e96e954685b4e27ec5803462c3308060a1835be18ac78a9884d425d';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),git=(...args)=>cp.execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
const read=name=>JSON.parse(fs.readFileSync(path.join(dir,name))),save=(name,data)=>fs.writeFileSync(path.join(dir,name),JSON.stringify(data,null,2)+'\n',{mode:0o600});
const list=async()=>(await management('/functions')).json(),summary=f=>({slug:f.slug,version:f.version,status:f.status,verify_jwt:f.verify_jwt,ezbr_sha256:f.ezbr_sha256});
async function source(){const b=Buffer.from(await(await management('/functions/generate-outfits/body')).arrayBuffer()),key=Buffer.from('"sourcesContent":['),p=b.indexOf(key);assert(p>=0,'missing deployed source');const a=p+key.length;let e=a+1;for(;e<b.length;e++){if(b[e]===92){e++;continue;}if(b[e]===34)break;}return JSON.parse(b.subarray(a,e+1).toString());}
async function build(){return esbuild.build({absWorkingDir:root,entryPoints:[entry],bundle:true,format:'esm',platform:'neutral',target:'es2022',external:['https://*'],write:false,metafile:true,logLevel:'silent'});}
const signature='public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)';
const definition=async()=>(await query(`select pg_get_functiondef(${quote(signature)}::regprocedure) definition`))[0].definition;
async function cache(){return (await query("select coalesce(jsonb_agg(to_jsonb(v) order by product_id),'[]') rows from public.catalog_visual_fashion v"))[0].rows;}
async function security(){const rows=await query(`select jsonb_build_object(
 'rls',not exists(select 1 from pg_class where oid in ('public.catalog_visual_fashion'::regclass,'public.catalog_image_embeddings'::regclass,'public.stylist_look_feedback'::regclass,'public.shop_catalog'::regclass,'public.saved_outfits'::regclass) and not relrowsecurity),
 'invoker',not(select prosecdef from pg_proc where oid=${quote(signature)}::regprocedure),
 'serviceOnly',not has_function_privilege('authenticated',${quote(signature)},'EXECUTE'),
 'privateEmbeddings',not has_table_privilege('authenticated','public.catalog_image_embeddings','SELECT'),
 'readOnlyVisual',not has_table_privilege('authenticated','public.catalog_visual_fashion','INSERT'),
 'feedbackPolicies',(select count(*)=4 from pg_policies where schemaname='public' and tablename='stylist_look_feedback')
 ) state`);assert(Object.values(rows[0].state).every(x=>x===true),'security preflight failed');return rows[0].state;}
function criticalSource(){
 const declarations=source=>{const tree=ts.createSourceFile('index.ts',source,ts.ScriptTarget.Latest,true),result=new Map();function visit(n){if(ts.isFunctionDeclaration(n)&&n.name)result.set(n.name.text,n.getText(tree));ts.forEachChild(n,visit);}visit(tree);return result;};
 const prior=declarations(git('show',BASE+':'+entry)),current=fs.readFileSync(path.join(root,entry),'utf8'),next=declarations(current);
 for(const name of ['shopPriceOf','outfitShopSpend','validateOutfitAgainstStyle','validateOutfitCompatibility','convertShopAmountEdge','budgetInCurrencyEdge']){assert(prior.has(name)&&next.has(name),'missing critical helper '+name);assert.equal(next.get(name),prior.get(name),'critical source drift '+name);}
 for(const p of ['src/domain/budgetFx.ts','backend/supabase/functions/_shared/rateLimit.ts'])assert.equal(fs.readFileSync(path.join(root,p),'utf8').trim(),git('show',BASE+':'+p),'critical source drift '+p);
 for(const condition of ["if (!/^Bearer\\s+\\S+$/i.test(authHeader.trim())) return authReply(401)","if (!user?.id) return authReply(401)",".eq('user_id', requestUserId)","status >= 500 ? 503 : 401"])assert(current.includes(condition),'missing auth/ownership guard');
 assert.equal(hash(fs.readFileSync(path.join(root,'.v199-balanced-build/v200.ts'))),SOURCE);
 assert.equal(hash(fs.readFileSync(path.join(root,'.v199-balanced-build/v199.ts'))),'205cf14073d71465b79edb6f226e13f86f47bd31cc0a8c7e953a37a1213db521');
 cp.execFileSync('git',['merge-base','--is-ancestor',BASE,'HEAD'],{cwd:root});
}
async function baseline(){const f=(await list()).find(f=>f.slug==='generate-outfits');assert.equal(f.version,200,'live version drift');assert.equal(f.status,'ACTIVE');assert.equal(f.ezbr_sha256,FINGERPRINT,'live fingerprint drift');assert.equal(hash(await source()),SOURCE,'live v200 source drift');return f;}
async function health(){const results=[];for(const auth of [null,'invalid-stylist5-release-probe']){const started=performance.now(),r=await fetch('https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/generate-outfits',{method:'POST',headers:{'Content-Type':'application/json',...(auth?{Authorization:'Bearer '+auth}:{})},body:'{}',signal:AbortSignal.timeout(20000)});await r.text();assert.equal(r.status,401);results.push({auth:auth?'invalid':'absent',status:r.status,wallMs:Math.round(performance.now()-started)});}return results;}
async function deploy(bytes,verifyJwt){const form=new FormData();form.append('metadata',JSON.stringify({name:'generate-outfits',entrypoint_path:'index.ts',verify_jwt:verifyJwt}));form.append('file',new Blob([bytes],{type:'application/typescript'}),'index.ts');return(await management('/functions/deploy?slug=generate-outfits',{method:'POST',body:form})).json();}
const mode=process.argv[2];assert(['prepare','release','verify','rollback'].includes(mode));
fs.mkdirSync(dir,{recursive:true,mode:0o700});criticalSource();
if(mode==='prepare'){
 assert(!fs.existsSync(path.join(dir,'release-manifest.json')),'manifest already frozen; do not silently overwrite');
 const live=await baseline(),b=await build(),visual=await cache(),sql=await definition();await security();
 save('rollback-visual.json',visual);fs.writeFileSync(path.join(dir,'rollback-retrieval.sql'),sql,{mode:0o600});save('functions-release-before.json',(await list()).map(summary));
 const manifest={baseline:summary(live),baselineCommit:BASE,baselineSourceSha256:SOURCE,bundleSha256:hash(b.outputFiles[0].contents),migrationSha256:hash(fs.readFileSync(path.join(root,migration))),cacheImportSha256:hash(fs.readFileSync(path.join(dir,'cache-import.sql'))),rollbackSqlSha256:hash(sql),rollbackVisualSha256:hash(JSON.stringify(visual)),inputs:Object.keys(b.metafile.inputs).map(p=>({path:p,sha256:hash(fs.readFileSync(path.join(root,p)))}))};
 save('release-manifest.json',manifest);console.log({prepared:true,baseline:200,bundleSha256:manifest.bundleSha256,rollbackProfiles:visual.length});
}else{
 const manifest=read('release-manifest.json');
 assert.equal(hash(fs.readFileSync(path.join(dir,'rollback-retrieval.sql'))),manifest.rollbackSqlSha256);assert.equal(hash(JSON.stringify(read('rollback-visual.json'))),manifest.rollbackVisualSha256);
 if(mode==='rollback'){
  const release=read('deployment.json'),live=(await list()).find(f=>f.slug==='generate-outfits');assert.equal(live.version,release.version);assert.equal(hash(await source()),manifest.bundleSha256);
  assert.equal(hash(JSON.stringify(await cache())),read('cache-after.json').sha256,'derived cache drift; refuse to overwrite newer evidence');
  const prior=read('rollback-visual.json'),sql=fs.readFileSync(path.join(dir,'rollback-retrieval.sql'),'utf8');
  await query(`begin;${sql}\ndelete from public.catalog_visual_fashion;insert into public.catalog_visual_fashion select * from jsonb_populate_recordset(null::public.catalog_visual_fashion,${quote(JSON.stringify(prior))}::jsonb);commit;`);
  const restored=await deploy(fs.readFileSync(path.join(root,'.v199-balanced-build/v200.ts')),manifest.baseline.verify_jwt);assert.equal(hash(await source()),SOURCE);assert.equal(hash(await definition()),manifest.rollbackSqlSha256);
  save('rollback-verification.json',{...summary(restored),sourceSha256:SOURCE,priorVisualCacheRestored:true,feedbackPreserved:true,health:await health()});console.log({restoredExactV200:true});
 }else{
  const b=await build();assert.equal(hash(b.outputFiles[0].contents),manifest.bundleSha256,'bundle drift');
  for(const input of manifest.inputs){assert.equal(hash(fs.readFileSync(path.join(root,input.path))),input.sha256);assert.equal(hash(cp.execFileSync('git',['show','HEAD:'+input.path],{cwd:root})),input.sha256,'uncommitted source '+input.path);}
  assert.equal(hash(fs.readFileSync(path.join(root,migration))),manifest.migrationSha256);
  if(mode==='release'){
   assert(!fs.existsSync(path.join(dir,'release-attempt.json')),'single guarded release already attempted');
   const checks=JSON.parse(fs.readFileSync(path.join(report,'gates.json')));assert(checks.passed===true&&checks.bundleSha256===manifest.bundleSha256,'critical gates missing/stale');
   const benchmark=JSON.parse(fs.readFileSync(path.join(report,'benchmark.json')));assert.equal(benchmark.bundleSha256,manifest.bundleSha256,'benchmark stale');assert(benchmark.scenarios.every(s=>s.after.looks>=2&&s.after.photoProducts/s.after.poolSize>=0.7));
   const visual=JSON.parse(fs.readFileSync(path.join(report,'visual-review.json')));assert(visual.passedForGuardedRelease===true&&visual.bundleSha256===manifest.bundleSha256,'visual review missing/stale');
   const branch=git('branch','--show-current'),head=git('rev-parse','HEAD');assert.equal(git('ls-remote','--heads','origin',branch).split(/\s/)[0],head,'push exact source first');
   await baseline();await security();assert.equal(hash(await definition()),manifest.rollbackSqlSha256,'retrieval drift');assert.equal(hash(JSON.stringify(await cache())),manifest.rollbackVisualSha256,'visual cache drift');
   const cacheSql=fs.readFileSync(path.join(dir,'cache-import.sql'),'utf8');assert.equal(hash(cacheSql),manifest.cacheImportSha256);assert.equal(hash(cp.execFileSync('git',['show','HEAD:'+migration],{cwd:root})),manifest.migrationSha256,'uncommitted migration');
   save('release-attempt.json',{startedAt:new Date().toISOString(),head,bundleSha256:manifest.bundleSha256});
   const integrity="select md5(string_agg(md5(to_jsonb(s)::text),'' order by id)) fingerprint from public.shop_catalog s";
   const sql=fs.readFileSync(path.join(root,migration),'utf8').replace('begin;',`begin isolation level repeatable read;create temporary table source_integrity_before on commit drop as ${integrity};`).replace("notify pgrst,'reload schema';",`${cacheSql}\ndo $$begin if(select fingerprint from source_integrity_before) is distinct from (${integrity}) then raise exception 'source_integrity_changed';end if;end $$;\ninsert into supabase_migrations.schema_migrations(version,name,statements) values('20261108040000','stylist5_editorial',array['Editorial role retrieval; derived evidence capsule; optional composition protocol']);\nnotify pgrst,'reload schema';`);
   await query(sql);save('cache-after.json',{sha256:hash(JSON.stringify(await cache()))});
   const deployed=await deploy(b.outputFiles[0].contents,manifest.baseline.verify_jwt);save('deployment.json',{...summary(deployed),sourceCommit:head,branch,deployedAt:new Date().toISOString()});
  }
  const deployment=read('deployment.json'),all=await list(),live=all.find(f=>f.slug==='generate-outfits');assert.equal(live.version,deployment.version);assert.equal(live.status,'ACTIVE');assert.equal(live.verify_jwt,manifest.baseline.verify_jwt);assert.equal(hash(await source()),manifest.bundleSha256);
  const others=all=>all.filter(f=>f.slug!=='generate-outfits').map(summary).sort((a,b)=>a.slug.localeCompare(b.slug));assert.deepEqual(others(all),others(read('functions-release-before.json')),'other functions changed');
  const verified={...summary(live),checkedAt:new Date().toISOString(),sourceCommit:deployment.sourceCommit,sourceSha256:manifest.bundleSha256,security:await security(),exactV200Rollback:true,health:await health(),ownerVisualQAPending:true,noSyntheticProviderCalls:true,sourceCatalogUnchanged:true};
  save('release-verification.json',verified);fs.writeFileSync(path.join(report,'release.json'),JSON.stringify(verified,null,2)+'\n');console.log(verified);
 }
}
