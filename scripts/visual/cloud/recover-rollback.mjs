/** Recover guarded rollback inputs from the private cloud record; no deployment. */
import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';import {query} from '../../stylist5/private-api.mjs';
const dir=process.env.STYLIST5_PRIVATE_DIR;if(!dir)throw Error('private recovery directory required');
const rows=await query("select * from public.stylist5_release_rollback where release_id='stylist5-v200-20261108'");assert.equal(rows.length,1);const r=rows[0];
assert.equal(crypto.createHash('sha256').update(r.baseline_source).digest('hex'),r.baseline_source_sha256);assert.equal(r.baseline_source_sha256,'6dc1c4c40e96e954685b4e27ec5803462c3308060a1835be18ac78a9884d425d');
fs.mkdirSync(dir,{recursive:true,mode:0o700});
for(const [name,value] of [['v200.ts',r.baseline_source],['retrieval-v200.sql',r.baseline_retrieval],['visual-v200.json',JSON.stringify(r.baseline_visual)],['manifest.json',JSON.stringify({generateSha:r.candidate_source_sha256,head:r.deployed_commit})],['deployment.json',JSON.stringify({version:r.deployed_version,sha:r.candidate_source_sha256})]])fs.writeFileSync(dir+'/'+name,value,{mode:0o600});
console.log({cloudRollbackRecovered:true,baselineVersion:r.baseline_version});
