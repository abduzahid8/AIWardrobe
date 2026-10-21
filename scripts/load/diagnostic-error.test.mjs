import test from 'node:test';
import assert from 'node:assert/strict';
import {diagnosticError} from './diagnostic-error.mjs';
test('upload diagnostics retain bounded transport/status metadata',()=>{
 assert.deepEqual(diagnosticError({name:'StorageUnknownError',originalError:{name:'TypeError',cause:{code:'UND_ERR_CONNECT_TIMEOUT'}}},20001.2),{name:'StorageUnknownError',originalName:'TypeError',status:null,networkCode:'UND_ERR_CONNECT_TIMEOUT',elapsedMs:20001});
 assert.equal(diagnosticError({name:'StorageApiError',statusCode:'504'},500).status,504);
});
test('arbitrary names, codes, messages and response bodies cannot leak through diagnostics',()=>{
 const secret='private-token-do-not-persist';const result=diagnosticError({name:secret,message:secret,statusCode:secret,code:secret,body:secret,originalError:{name:secret,message:secret}},NaN);
 assert.ok(!JSON.stringify(result).includes(secret));assert.equal(result.name,'UnclassifiedError');assert.equal(result.elapsedMs,null);
});
