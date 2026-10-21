import test from 'node:test';
import assert from 'node:assert/strict';
import {responseDiagnostics,timingSummary} from './http-diagnostics.mjs';
test('keeps only validated correlation IDs and bounded upstream timing',()=>{
 const result=responseDiagnostics(new Response('',{headers:{'sb-request-id':'12345678-1234-1234-1234-123456789abc','cf-ray':'1234567890abcdef-FRA','x-envoy-upstream-service-time':'4','authorization':'Bearer private-token','set-cookie':'private-cookie'}}));
 assert.deepEqual(result,{supabaseRequestId:'12345678-1234-1234-1234-123456789abc',cloudflareRay:'1234567890abcdef-FRA',upstreamMs:4});
 assert.equal(JSON.stringify(result).includes('private'),false);
});
test('drops arbitrary header content and impossible timing',()=>{
 assert.deepEqual(responseDiagnostics(new Response('',{headers:{'sb-request-id':'Bearer private-token','cf-ray':'https://private.invalid','x-envoy-upstream-service-time':'9999999'}})),{});
 assert.deepEqual(responseDiagnostics(new Response('')),{});
});
test('stage percentiles preserve original samples',()=>{
 const values=[50,1,5,2];assert.deepEqual(timingSummary(values),{samples:4,p50:2,p95:50,p99:50});assert.deepEqual(values,[50,1,5,2]);
 assert.deepEqual(timingSummary([]),{samples:0,p50:null,p95:null,p99:null});
});
