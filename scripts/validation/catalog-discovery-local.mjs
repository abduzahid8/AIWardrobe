/** Disposable PostgreSQL verification. Never reads .env or a hosted DB URL.
 * Usage: AIWARDROBE_PG_BIN=/path/to/postgres/bin node scripts/validation/catalog-discovery-local.mjs
 */
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
const bin=process.env.AIWARDROBE_PG_BIN || spawnSync('pg_config',['--bindir'],{encoding:'utf8'}).stdout?.trim();
if(!bin) throw new Error('Set AIWARDROBE_PG_BIN to a local PostgreSQL installation');
const root=mkdtempSync(join(tmpdir(),'aiwardrobe-discovery-')),data=join(root,'data');
const run=(command,args,input)=>{
 const result=spawnSync(join(bin,command),args,{encoding:'utf8',input,maxBuffer:8*1024*1024});
 if(result.status!==0) throw new Error(`${command} failed\n${result.stderr}\n${result.stdout}`);
 return result.stdout;
};
const sql=text=>run('psql',['-X','-h',root,'-p','55440','-U','catalog_test_owner','-d','postgres','-v','ON_ERROR_STOP=1'],text);
let started=false;
try {
 run('initdb',['-D',data,'-U','catalog_test_owner','-A','trust','--no-locale','--encoding=UTF8']);
 // Empty listen_addresses: Unix socket only, isolated from all hosted services.
 run('pg_ctl',['-D',data,'-l',join(root,'postgres.log'),'-o',`-h '' -k ${root} -p 55440`,'start']);started=true;
 sql(readFileSync(resolve('__tests__/fixtures/catalogDiscovery.sql'),'utf8'));
 sql(readFileSync(resolve('backend/supabase/migrations/20261102163044_shop_catalog_discovery.sql'),'utf8'));
 process.stdout.write(sql(readFileSync(resolve('__tests__/fixtures/catalogDiscoveryAssertions.sql'),'utf8')));
 if(process.argv.includes('--purchase') || process.argv.includes('--release')) {
  sql(readFileSync(resolve('__tests__/fixtures/purchase/schema-setup.sql'),'utf8'));
  sql(readFileSync(resolve('backend/supabase/migrations/20261102173109_purchase_intelligence.sql'),'utf8'));
  sql(readFileSync(resolve('__tests__/fixtures/purchase/schema-assertions.sql'),'utf8'));
  if(process.argv.includes('--release')) {
  const {readdirSync}=await import('node:fs');
  for(const name of readdirSync('backend/supabase/migrations').filter(n=>/^2026110[3-6].*\.sql$/.test(n)).sort()) {
   sql(readFileSync(resolve('backend/supabase/migrations',name),'utf8'));
   process.stdout.write(`PASS replay ${name}\n`);
  }
  sql(readFileSync(resolve('__tests__/fixtures/releaseIsolation.sql'),'utf8'));
  for(const phase of [6,5,4,3]) sql(readFileSync(resolve(`backend/supabase/rollback/phase${phase}_disable.sql`),'utf8'));
  sql(`do $$ begin
    if (select count(*) from public.style_evidence) <> 1 then raise exception 'rollback lost style evidence'; end if;
    if has_table_privilege('authenticated','public.style_evidence','SELECT') then raise exception 'rollback left client grants'; end if;
  end $$;`);
  process.stdout.write('PASS phase3-6 isolation, explicit grants, reward write denial, indexes, reverse disable/data retention\n');

 }
 sql(readFileSync(resolve('backend/supabase/rollback/purchase_intelligence_disable.sql'),'utf8'));
  const retained=sql("select count(*) from public.purchase_fit_evidence;");
  if(!/\b1\b/.test(retained))throw new Error('rollback lost private fit evidence');
  process.stdout.write('PASS purchase schema: private owner RLS, public offer/history RLS, service writes, changed-price coalescing, immutable identity, non-destructive rollback\n');
 }
 sql(readFileSync(resolve('backend/supabase/rollback/shop_catalog_discovery.sql'),'utf8'));
 const remaining=sql("select count(*) from pg_proc where proname='catalog_discovery_candidates';");
 if(!/\b0\b/.test(remaining))throw new Error('rollback did not remove RPC');
 process.stdout.write('PASS local discovery: coverage, filters, RLS, bounds, pagination, rotation, native floors, invalid inputs, rollback\n');
} finally {
 if(started)run('pg_ctl',['-D',data,'-m','immediate','stop']);
 rmSync(root,{recursive:true,force:true});
}
