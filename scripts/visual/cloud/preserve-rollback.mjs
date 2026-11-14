/** Store exact UTF-8 rollback bytes without SQL string escape transformations. */
import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';import {query} from '../../stylist5/private-api.mjs';
const dir=process.env.STYLIST5_PRIVATE_DIR;if(!dir)throw Error('private release directory required');
const source=fs.readFileSync(dir+'/v200.ts'),retrieval=fs.readFileSync(dir+'/retrieval-v200.sql'),visual=fs.readFileSync(dir+'/visual-v200.json');
const original='6dc1c4c40e96e954685b4e27ec5803462c3308060a1835be18ac78a9884d425d';assert.equal(crypto.createHash('sha256').update(source).digest('hex'),original);
const deployment=JSON.parse(fs.readFileSync(dir+'/deployment.json'));
const current=await query("select baseline_source_sha256,candidate_source_sha256,deployed_version from public.stylist5_release_rollback where release_id='stylist5-v200-20261108'");assert.equal(current.length,1);assert.equal(current[0].baseline_source_sha256,original);assert.equal(current[0].candidate_source_sha256,deployment.sha);assert.equal(current[0].deployed_version,deployment.version);
const literal=bytes=>"convert_from(decode('"+bytes.toString('base64')+"','base64'),'UTF8')";
await query(`update public.stylist5_release_rollback set baseline_source=${literal(source)},baseline_retrieval=${literal(retrieval)},baseline_visual=${literal(visual)}::jsonb where release_id='stylist5-v200-20261108'`);
const stored=(await query("select baseline_source,baseline_retrieval,baseline_visual from public.stylist5_release_rollback where release_id='stylist5-v200-20261108'"))[0];
assert.equal(crypto.createHash('sha256').update(stored.baseline_source).digest('hex'),original,'cloud rollback source integrity failed');assert.equal(stored.baseline_retrieval,retrieval.toString());assert.deepEqual(stored.baseline_visual,JSON.parse(visual));
console.log({exactCloudRollbackReadbackConfirmed:true,baselineVersion:200,originalVisualProfiles:stored.baseline_visual.length});
