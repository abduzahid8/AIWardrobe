import test from 'node:test';
import assert from 'node:assert/strict';
import {REGIONAL_MEDIA_CASES,verifyRegionalMediaExecution} from './regional-media.mjs';
test('regional comparison uses fixed separate cohorts and verified runtime plus response header',()=>{
 assert.equal(new Set(REGIONAL_MEDIA_CASES.map(value=>value.cohort)).size,2);
 for(const value of REGIONAL_MEDIA_CASES)assert.doesNotThrow(()=>verifyRegionalMediaExecution({region:value.region},value.region,value.region));
});
test('wrong runtime, missing response header or unsupported requested region cannot count as regional evidence',()=>{
 for(const [result,requested,header] of [[{region:'eu-central-1'},'ap-northeast-1','ap-northeast-1'],[{region:'ap-northeast-1'},'ap-northeast-1',null],[{region:'ap-northeast-1'},'ap-northeast-1','eu-central-1'],[{region:'unknown'},'unknown','unknown']])assert.throws(()=>verifyRegionalMediaExecution(result,requested,header),/ProbeRegionMismatch/);
});
