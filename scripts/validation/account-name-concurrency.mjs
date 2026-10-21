/** Local Postgres race against the actual Auth trigger and unique constraint. */
import { spawn, execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
const ids=[randomUUID(),randomUUID()],handle='race_'+randomUUID().replaceAll('-','').slice(0,20);
const args=['-h','/tmp','-p','55439','-d','postgres','-v','ON_ERROR_STOP=1','-v','VERBOSITY=verbose'];
const run=sql=>new Promise(resolve=>{const child=spawn('psql',args,{stdio:['pipe','pipe','pipe']});let err='';child.stderr.on('data',x=>err+=x);child.stdout.resume();child.on('close',code=>resolve({code,uniqueViolation:/23505/.test(err)}));child.stdin.end(sql);});
let report;
try {
 const results=await Promise.all(ids.map((id,index)=>run(`begin; insert into auth.users(id,email,raw_user_meta_data) values('${id}','${id}@example.invalid',jsonb_build_object('handle','${index?handle.toUpperCase():handle}','display_name','Shared Name')); select pg_sleep(0.2); commit;`)));
 report={date:'2026-10-20',scope:'Two concurrent local PostgreSQL Auth-trigger transactions request the same case-insensitive handle',results,passed:results.filter(x=>x.code===0).length===1&&results.filter(x=>x.uniqueViolation).length===1};
} finally {
 execFileSync('psql',[...args,'-c',`delete from auth.users where id in('${ids[0]}','${ids[1]}');`],{stdio:'ignore'});
}
await writeFile('docs/reports/readiness-evidence/account-name-concurrency.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;
