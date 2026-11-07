/* Guarded Stylist 4 rollout. Credentials stay in env/a local token file.
 * prepare -> migrate -> deploy -> verify. Rollback restores exact v197 code;
 * additive metadata/feedback remain compatible and preserve user feedback. */
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),report=path.join(root,'docs/reports/stylist-4-20261107'),record=path.join(root,'backend/supabase/rollback/generate-outfits-stylist-4-20261107');
const baseline='8282f3c7', fingerprint='354eedb16a4948cc3c7a68c0cef2a46c668fe89284f74c6a82709413277dfd71',sourceHash='06fb242adc0430b48eab97c6905cdef91f3eb087ca18d79811b54525e9cf4cf4';
const base='https://api.supabase.com/v1/projects/fyqpifmrsftsfqibhwhy',entry='backend/supabase/functions/generate-outfits/index.ts',migration='backend/supabase/migrations/20261107230000_visual_fashion_concepts.sql';
const token=process.env.SUPABASE_ACCESS_TOKEN||fs.readFileSync(process.env.SUPABASE_ACCESS_TOKEN_FILE,'utf8').trim(),headers={Authorization:'Bearer '+token};
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),git=(...args)=>cp.execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
const save=(name,x)=>fs.writeFileSync(path.join(report,name),JSON.stringify(x,null,2)+'\n'),read=name=>JSON.parse(fs.readFileSync(path.join(report,name)));
const summary=f=>({slug:f.slug,version:f.version,status:f.status,verify_jwt:f.verify_jwt,ezbr_sha256:f.ezbr_sha256});
async function api(url,init={}){const r=await fetch(base+url,{...init,headers:{...headers,...init.headers},signal:AbortSignal.timeout(60000)});assert(r.ok,'management '+r.status);return r;}
const list=async()=> (await api('/functions')).json();
const query=async query=>(await api('/database/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})})).json();
async function source(){const b=Buffer.from(await(await api('/functions/generate-outfits/body')).arrayBuffer()),key=Buffer.from('"sourcesContent":[');const p=b.indexOf(key);assert(p>=0);const a=p+key.length;let e=a+1;for(;e<b.length;e++){if(b[e]===92){e++;continue;}if(b[e]===34)break;}return JSON.parse(b.subarray(a,e+1).toString());}
const build=()=>require('esbuild').build({absWorkingDir:root,entryPoints:[entry],bundle:true,format:'esm',platform:'neutral',target:'es2022',external:['https://*'],write:false,metafile:true,logLevel:'silent'});
async function baselineLive(){const f=(await list()).find(f=>f.slug==='generate-outfits');assert.equal(f.version,197,'production version drift');assert.equal(f.status,'ACTIVE');assert.equal(f.ezbr_sha256,fingerprint,'production fingerprint drift');assert.equal(hash(await source()),sourceHash,'v197 source drift');return f;}
async function security(){const r=await query(`select jsonb_build_object(
 'rls',not exists(select 1 from pg_class where oid in ('public.catalog_visual_fashion'::regclass,'public.catalog_image_embeddings'::regclass,'public.stylist_look_feedback'::regclass) and not relrowsecurity),
 'invoker',not (select prosecdef from pg_proc where oid='public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)'::regprocedure),
 'readOnlyVisual',not has_table_privilege('authenticated','public.catalog_visual_fashion','INSERT'),
 'privateEmbeddings',not has_table_privilege('authenticated','public.catalog_image_embeddings','SELECT'),
 'serviceOnly',not has_function_privilege('authenticated','public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)','EXECUTE'),
 'feedbackPolicies',(select count(*)=4 from pg_policies where schemaname='public' and tablename='stylist_look_feedback'),
 'footwearGuard',position('n.garment_type = ''shoes''' in pg_get_functiondef('public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)'::regprocedure))>0
 ) state`);assert(Object.values(r[0].state).every(v=>v===true),'SQL security preflight failed');return r[0].state;}
async function health(){const url='https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/generate-outfits';const result=[];for(const request of [{method:'OPTIONS'},{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'},{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer invalid-release-probe'},body:'{}'}]){const r=await fetch(url,{...request,signal:AbortSignal.timeout(20000)});assert.equal(r.status,request.method==='OPTIONS'?200:401);await r.text();result.push({method:request.method,auth:request.headers?.Authorization?'invalid':'absent',status:r.status});}return result;}
function criticalSource(){
 const ts=require('typescript'),functions=source=>{const tree=ts.createSourceFile('index.ts',source,ts.ScriptTarget.Latest,true),result=new Map();function visit(n){if((ts.isFunctionDeclaration(n)||ts.isVariableDeclaration(n))&&n.name&&ts.isIdentifier(n.name))result.set(n.name.text,n.getText(tree));ts.forEachChild(n,visit);}visit(tree);return result;};
 const prior=functions(git('show',baseline+':'+entry)),next=functions(fs.readFileSync(path.join(root,entry),'utf8'));
 for(const name of ['shopPriceOf','outfitShopSpend','validateOutfitAgainstStyle','validateOutfitCompatibility','convertShopAmountEdge','budgetInCurrencyEdge'])assert.equal(next.get(name),prior.get(name),'safeguard changed '+name);
 for(const p of ['backend/supabase/functions/_shared/rateLimit.ts','src/domain/budgetFx.ts','src/domain/outfitGenerationRequest.ts'])assert.equal(fs.readFileSync(path.join(root,p),'utf8').trim(),git('show',baseline+':'+p),'safeguard changed '+p);
 const current=fs.readFileSync(path.join(root,entry),'utf8');for(const condition of ["if (!/^Bearer\\s+\\S+$/i.test(authHeader.trim())) return authReply(401)","if (!user?.id) return authReply(401)",".eq('user_id', requestUserId)","status >= 500 ? 503 : 401"])assert(current.includes(condition),'auth ownership guard missing');
 return true;
}
async function main(){
 const mode=process.argv[2];assert(['prepare','prepare-update','migrate','migrate-feedback','probe-retrieval','migrate-retrieval','deploy','verify','rollback','observe'].includes(mode));
 fs.mkdirSync(report,{recursive:true});fs.mkdirSync(record,{recursive:true});cp.execFileSync('git',['merge-base','--is-ancestor',baseline,'HEAD'],{cwd:root});
 if(mode==='observe'){
 const sql="select timestamp,event_message from logs where source = 'function_logs' and (position(event_message, 'stylist_provider')>0 or position(event_message, 'stylist_request')>0) order by timestamp desc limit 100";
 const params=new URLSearchParams({sql,iso_timestamp_start:new Date(Date.now()-24*3600000).toISOString(),iso_timestamp_end:new Date().toISOString()});
 const data=await(await api('/analytics/endpoints/logs?'+params)).json();save('traffic-latest.json',data);console.log(JSON.stringify(data));return;
 }
 if(mode==='prepare'){
 const live=await baselineLive();criticalSource();const b=await build(),bundle=b.outputFiles[0].contents;
 const original=await source(),tmp=fs.mkdtempSync('/tmp/aiwardrobe-stylist4-rollback-');fs.writeFileSync(path.join(tmp,'index.ts'),original);cp.execFileSync('tar',['-czf',path.join(record,'v197-source.tar.gz'),'-C',tmp,'index.ts']);fs.rmSync(tmp,{recursive:true});
 const manifest={baseline:summary(live),baselineSourceSha256:sourceHash,baselineCommit:git('rev-parse',baseline),bundleSha256:hash(bundle),migrationSha256:hash(fs.readFileSync(path.join(root,migration))),rollbackSha256:hash(fs.readFileSync(path.join(record,'v197-source.tar.gz'))),criticalSourceUnchanged:true,inputs:Object.keys(b.metafile.inputs).map(p=>({path:p,sha256:hash(fs.readFileSync(path.join(root,p)))}))};
 fs.writeFileSync(path.join(record,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');save('functions-before.json',(await list()).map(summary));console.log(JSON.stringify({prepared:true,baseline:197,bundleSha256:manifest.bundleSha256}));return;
 }
 const manifest=JSON.parse(fs.readFileSync(path.join(record,'manifest.json')));
 const archive=path.join(record,'v197-source.tar.gz');assert.equal(hash(fs.readFileSync(archive)),manifest.rollbackSha256);assert.equal(hash(cp.execFileSync('tar',['-xOzf',archive,'index.ts'])),sourceHash);
 if(mode==='probe-retrieval'||mode==='migrate-retrieval'){
 const deployment=read('deployment.json'),live=(await list()).find(f=>f.slug==='generate-outfits');
 assert.equal(live.version,deployment.version);assert.equal(live.ezbr_sha256,deployment.ezbr_sha256);assert.equal(hash(await source()),manifest.bundleSha256);
 const file='backend/supabase/migrations/20261108003000_stylist_set_retrieval.sql',sql=fs.readFileSync(path.join(root,file),'utf8');
 assert.equal(hash(sql),hash(cp.execFileSync('git',['show','HEAD:'+file],{cwd:root})),'commit retrieval migration first');
 const signature='public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)',candidate='catalog_stylist_candidates_candidate';
 const prior=(await query('select pg_get_functiondef('+"'"+signature+"'"+'::regprocedure) definition'))[0].definition;
 const backup=path.join(record,'v199-catalog-stylist-candidates.sql');
 if(mode==='probe-retrieval'){
 assert(!(await query("select to_regprocedure('public.catalog_stylist_candidates_candidate(text[],text,jsonb,integer,integer,text,numeric)') existing"))[0].existing,'candidate already exists');
 fs.writeFileSync(backup,prior+'\n');
 await query(sql.replaceAll('public.catalog_stylist_candidates(','public.'+candidate+'('));
 const context={dressCode:'smart',concepts:[{id:'ivy',preferredSubtypes:['dress-shirt','shirt','blazer','cardigan','chino','tailored-trouser','loafer','derby']},{id:'texture',preferredSubtypes:['shirt','fine-knit','dress-shirt','blazer','coat','tailored-trouser','trouser','derby','loafer','monk']},{id:'knit',preferredSubtypes:['shirt','dress-shirt','polo','cardigan','overshirt','tailored-trouser','chino','loafer','chelsea']}]};
 const call="public."+candidate+"(array['top','outerwear','bottom','shoes'],'stylist4-query-cost','"+JSON.stringify(context)+"',100,0,null,null)";
 const explain=await query('explain (analyze,buffers,format json) select * from '+call);
 const validation=(await query(`with results as(select row from ${call} row)
 select jsonb_build_object('bounded',(select count(*)<=400 from results),
 'realProducts',not exists(select 1 from results r left join public.shop_catalog s on s.id=r.row->>'id' where s.id is null),
 'sourcePrices',not exists(select 1 from results r join public.shop_catalog s on s.id=r.row->>'id' where (r.row->>'price')::numeric<>s.price or r.row->>'currency'<>s.currency or r.row->>'image_url'<>s.image_url or r.row->>'source_url' is distinct from s.source_url),
 'footwear',not exists(select 1 from results where row->>'fashion_slot'='shoes' and row->>'garment_type' is distinct from 'shoes'),
 'private',not has_function_privilege('authenticated','public.catalog_stylist_candidates_candidate(text[],text,jsonb,integer,integer,text,numeric)','EXECUTE'),
 'invoker',not (select prosecdef from pg_proc where oid='public.catalog_stylist_candidates_candidate(text[],text,jsonb,integer,integer,text,numeric)'::regprocedure)
 ) state`))[0].state;
 assert(Object.values(validation).every(v=>v===true),'live candidate integrity failed');
 const ms=explain[0]['QUERY PLAN'][0]['Execution Time'];assert(ms<1500,'candidate query still too slow');
 save('retrieval-query-probe.json',{checkedAt:new Date().toISOString(),migrationSha256:hash(sql),backupSha256:hash(fs.readFileSync(backup)),executionMs:ms,plan:explain,validation});console.log(JSON.stringify({candidateExecutionMs:ms,validation}));return;
 }
 const probe=read('retrieval-query-probe.json');assert.equal(probe.migrationSha256,hash(sql));assert(Object.values(probe.validation).every(v=>v===true));
 assert.equal(hash(fs.readFileSync(backup)),probe.backupSha256);assert.equal(prior.trim(),fs.readFileSync(backup,'utf8').trim(),'retrieval SQL drift');
 await query(sql.replace('commit;',"insert into supabase_migrations.schema_migrations(version,name,statements) values('20261108003000','stylist_set_retrieval',array['Single bounded scan and narrow concept ranking; source checks retained']);\ncommit;"));
 await query('drop function public.catalog_stylist_candidates_candidate(text[],text,jsonb,integer,integer,text,numeric)');
 save('retrieval-migration.json',{version:'20261108003000',sha256:hash(sql),candidateExecutionMs:probe.executionMs,security:await security(),v197DiscoveryUnchanged:(await query("select pg_get_functiondef('public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)'::regprocedure) definition"))[0].definition===read('database-before.json').find(r=>r.kind==='candidate_definition').evidence});
 assert(read('retrieval-migration.json').v197DiscoveryUnchanged);console.log('set-based retrieval activated with preserved v197 discovery');return;
 }
 if(mode==='migrate-feedback'){
 const deployment=read('deployment.json'),live=(await list()).find(f=>f.slug==='generate-outfits');
 assert.equal(live.version,deployment.version);assert.equal(live.ezbr_sha256,deployment.ezbr_sha256);assert.equal(hash(await source()),manifest.bundleSha256);
 const file='backend/supabase/migrations/20261107233000_stylist_feedback_request_keys.sql',sql=fs.readFileSync(path.join(root,file),'utf8');
 assert.equal(hash(sql),hash(cp.execFileSync('git',['show','HEAD:'+file],{cwd:root})),'commit compatibility migration first');
 const before=(await query("select count(*)::int records from public.stylist_look_feedback"))[0].records;
 await query(sql.replace('commit;',"insert into supabase_migrations.schema_migrations(version,name,statements) values('20261107233000','stylist_feedback_request_keys',array['Accept existing bounded mobile request keys without changing ownership']);\ncommit;"));
 await query("notify pgrst,'reload schema'");
 const after=(await query("select count(*)::int records from public.stylist_look_feedback"))[0].records;assert.equal(after,before);
 save('feedback-migration.json',{version:'20261107233000',sha256:hash(sql),recordsPreserved:true,security:await security()});console.log('feedback request-key compatibility passed');return;
 }
 if(mode==='prepare-update'){
 const previous=read('deployment.json'),live=(await list()).find(f=>f.slug==='generate-outfits');
 assert.equal(live.version,previous.version,'production version drift');assert.equal(live.ezbr_sha256,previous.ezbr_sha256,'production fingerprint drift');
 assert.equal(live.status,'ACTIVE');assert.equal(live.verify_jwt,manifest.baseline.verify_jwt);assert.equal(hash(await source()),manifest.bundleSha256,'deployed candidate source drift');
 criticalSource();await security();const b=await build();
 const next={...manifest,releaseFrom:{...summary(live),sourceSha256:manifest.bundleSha256},bundleSha256:hash(b.outputFiles[0].contents),
 inputs:Object.keys(b.metafile.inputs).map(p=>({path:p,sha256:hash(fs.readFileSync(path.join(root,p)))}))};
 fs.writeFileSync(path.join(record,'manifest.json'),JSON.stringify(next,null,2)+'\n');console.log(JSON.stringify({preparedUpdate:true,from:live.version,bundleSha256:next.bundleSha256,rollback:197}));return;
 }
 if(mode==='migrate'){
 await baselineLive();assert.equal(hash(fs.readFileSync(path.join(root,migration))),manifest.migrationSha256);
 assert.equal(hash(cp.execFileSync('git',['show','HEAD:'+migration],{cwd:root})),manifest.migrationSha256,'commit migration first');
 const original=(await query("select pg_get_functiondef('public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)'::regprocedure) definition"))[0].definition;
 const expected=read('database-before.json').find(r=>r.kind==='candidate_definition').evidence;assert.equal(original,expected,'v197 SQL drift');
 const sql=fs.readFileSync(path.join(root,migration),'utf8').replace("notify pgrst,'reload schema';","insert into supabase_migrations.schema_migrations(version,name,statements) values('20261107230000','visual_fashion_concepts',array['Additive Stylist 4 offline visual evidence and private feedback']);\nnotify pgrst,'reload schema';");
 await query(sql);save('migration.json',{appliedAt:new Date().toISOString(),version:'20261107230000',sha256:manifest.migrationSha256,security:await security(),v197SqlUnchanged:true});console.log('additive migration passed');return;
 }
 if(mode==='rollback'){
 const deployment=read('deployment.json'),live=(await list()).find(f=>f.slug==='generate-outfits');assert.equal(live.version,deployment.version);assert.equal(live.ezbr_sha256,deployment.ezbr_sha256);assert.equal(hash(await source()),manifest.releaseFrom?.version===deployment.version?manifest.releaseFrom.sourceSha256:manifest.bundleSha256);
 const original=cp.execFileSync('tar',['-xOzf',archive,'index.ts']);const form=new FormData();form.append('metadata',JSON.stringify({name:'generate-outfits',entrypoint_path:'index.ts',verify_jwt:manifest.baseline.verify_jwt}));form.append('file',new Blob([original],{type:'application/typescript'}),'index.ts');const deployed=await(await api('/functions/deploy?slug=generate-outfits',{method:'POST',body:form})).json();assert.equal(hash(await source()),sourceHash);save('rollback-verification.json',{...summary(deployed),restoresVersion:197,sourceSha256:sourceHash,health:await health()});console.log('restored exact v197 code');return;
 }
 const b=await build();assert.equal(hash(b.outputFiles[0].contents),manifest.bundleSha256,'bundle drift');
 for(const input of manifest.inputs){assert.equal(hash(fs.readFileSync(path.join(root,input.path))),input.sha256,'input drift '+input.path);assert.equal(hash(cp.execFileSync('git',['show','HEAD:'+input.path],{cwd:root})),input.sha256,'uncommitted input '+input.path);}
 const securityState=await security();criticalSource();
 if(mode==='deploy'){
 if(manifest.releaseFrom){const live=(await list()).find(f=>f.slug==='generate-outfits');assert.equal(live.version,manifest.releaseFrom.version);assert.equal(live.ezbr_sha256,manifest.releaseFrom.ezbr_sha256);assert.equal(live.status,'ACTIVE');assert.equal(hash(await source()),manifest.releaseFrom.sourceSha256);}else await baselineLive();
 const branch=git('branch','--show-current'),head=git('rev-parse','HEAD');assert.equal(git('ls-remote','--heads','origin',branch).split(/\s/)[0],head,'push source first');
 const checks=read('essential-checks.json');assert(checks.passed===true && checks.bundleSha256===manifest.bundleSha256,'essential checks missing or stale');
 const form=new FormData();form.append('metadata',JSON.stringify({name:'generate-outfits',entrypoint_path:'index.ts',verify_jwt:manifest.baseline.verify_jwt}));form.append('file',new Blob([b.outputFiles[0].contents],{type:'application/typescript'}),'index.ts');const deployed=await(await api('/functions/deploy?slug=generate-outfits',{method:'POST',body:form})).json();save('deployment.json',{...summary(deployed),sourceCommit:head,branch});
 }
 const deployment=read('deployment.json'),all=await list(),live=all.find(f=>f.slug==='generate-outfits');assert.equal(live.version,deployment.version);assert.equal(live.status,'ACTIVE');assert.equal(live.ezbr_sha256,deployment.ezbr_sha256);assert.equal(live.verify_jwt,manifest.baseline.verify_jwt);assert.equal(hash(await source()),manifest.bundleSha256);
 const others=r=>r.filter(f=>f.slug!=='generate-outfits').map(summary).sort((a,b)=>a.slug.localeCompare(b.slug));assert.deepEqual(others(all),others(read('functions-before.json')),'other functions changed');
 save('verification.json',{checkedAt:new Date().toISOString(),...summary(live),sourceCommit:deployment.sourceCommit,sourceSha256:manifest.bundleSha256,security:securityState,exactV197RollbackVerified:true,health:await health(),noSyntheticInference:true});console.log(JSON.stringify(read('verification.json')));
}
main().catch(e=>{console.error(e.message);process.exitCode=1});
