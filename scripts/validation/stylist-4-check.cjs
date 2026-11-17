/* Focused release gates. No inference calls or broad test suites. */
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),report=path.join(root,'docs/reports/stylist-4-20261107'),results=[];
function run(name,command,args,options={}){
 const r=cp.spawnSync(command,args,{cwd:root,encoding:'utf8',maxBuffer:12*1024*1024,...options});
 const output=(r.stdout||'')+(r.stderr||'');fs.writeFileSync(path.join(report,name+'.txt'),output);
 results.push({name,passed:r.status===0,status:r.status});
 console.log(name+': '+(r.status===0?'passed':'FAILED'));
 if(r.status!==0){console.error(output.slice(-8000));throw Error(name+' failed');}
}
(async()=>{
 try{
 run('edge-compilation',process.env.DENO_BIN||path.join(os.homedir(),'.deno/bin/deno'),['check','--no-config','--node-modules-dir=none','--lock=backend/supabase/functions/deno.lock','--frozen','backend/supabase/functions/generate-outfits/index.ts']);
 const config=path.join(os.tmpdir(),'aiwardrobe-stylist4-app-check.json');
 fs.writeFileSync(config,JSON.stringify({extends:path.join(root,'tsconfig.json'),compilerOptions:{typeRoots:[path.join(root,'node_modules/@types')],types:['jest','node']},include:['src','screens','features','components','hooks','store','navigation'].map(p=>path.join(root,p,'**/*')).concat([path.join(root,'global.d.ts'),path.join(root,'nativewind-env.d.ts')]),exclude:[path.join(root,'node_modules'),path.join(root,'docs'),path.join(root,'__tests__')]}));
 run('app-compilation',path.join(root,'node_modules/.bin/tsc'),['--noEmit','--pretty','false','-p',config]);
 run('critical-tests',path.join(root,'node_modules/.bin/jest'),['--runInBand','__tests__/services/stylist4.test.ts','__tests__/services/validatedQuartets.test.ts','__tests__/services/outfitCoherence.test.ts','__tests__/backend/budgetFx.test.js','__tests__/backend/outfitBudgetSelection.test.js','__tests__/backend/outfitGenerationRequest.test.js','__tests__/backend/nvidiaOutfitRequest.test.js']);
 run('auth-identity-tests',path.join(root,'node_modules/.bin/jest'),['--runInBand','__tests__/backend/outfitBudgetWire.test.js','--testNamePattern=provided clothing cannot bypass|returned product attributes come']);
 run('sql-security',process.env.PSQL_PATH||(fs.existsSync('/opt/homebrew/opt/postgresql@16/bin/psql')?'/opt/homebrew/opt/postgresql@16/bin/psql':'psql'),['-h',process.env.STYLIST_TEST_PGHOST||'/tmp','-p',process.env.STYLIST_TEST_PGPORT||'55439','-d',process.env.STYLIST_TEST_PGDATABASE||'postgres','-v','ON_ERROR_STOP=1','-f','__tests__/fixtures/stylist4-security.sql']);
 const b=await require('esbuild').build({absWorkingDir:root,entryPoints:['backend/supabase/functions/generate-outfits/index.ts'],bundle:true,format:'esm',platform:'neutral',target:'es2022',external:['https://*'],write:false});
 const bundleSha256=crypto.createHash('sha256').update(b.outputFiles[0].contents).digest('hex');
 fs.writeFileSync(path.join(report,'essential-checks.json'),JSON.stringify({checkedAt:new Date().toISOString(),passed:true,bundleSha256,scope:'Focused critical gates, no inference or production performance claim',checks:results},null,2)+'\n');
 }catch(e){fs.writeFileSync(path.join(report,'essential-checks.json'),JSON.stringify({passed:false,checks:results},null,2)+'\n');throw e;}
})().catch(e=>{console.error(e.message);process.exitCode=1});
