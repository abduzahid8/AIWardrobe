/** Exercise advisory attack shapes against every installed copy in the lock graph. */
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdtempSync,mkdirSync,existsSync,rmSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const lock=JSON.parse(readFileSync(new URL('../../package-lock.json',import.meta.url),'utf8'));
const root=fileURLToPath(new URL('../..',import.meta.url));
const copies=name=>Object.keys(lock.packages).filter(path=>path===`node_modules/${name}`||path.endsWith(`/node_modules/${name}`));
for(const path of copies('proxy-addr')){
 test(`${path}: malformed mapped subnet cannot trust a public spoofing client`,()=>{
  const proxy=require(resolve(root,path));
  const request={connection:{remoteAddress:'203.0.113.8'},headers:{'x-forwarded-for':'10.9.9.9'}};
  assert.equal(proxy(request,proxy.compile('::ffff:10.0.0.0/8')),'203.0.113.8');
  assert.equal(proxy.compile('::ffff:10.0.0.0/104')('10.1.2.3'),true);
  assert.equal(proxy.compile('10.0.0.0/8')('203.0.113.8'),false);
 });
}
for(const path of copies('shell-quote')){
 test(`${path}: newline operators and post-comment tokens reject without execution`,()=>{
  const shell=require(resolve(root,path));
  assert.throws(()=>shell.quote([{op:'\nprintf injected'}]),/invalid.*op/i);
  assert.throws(()=>shell.quote([{comment:'comment'},'\nprintf injected']),/line terminator/i);
  const words=['ordinary word','two\nlines',"single'quote",'$HOME'];
  assert.deepEqual(shell.parse(shell.quote(words)),words);
 });
}
const fixturePython=`import zipfile,stat,sys
with zipfile.ZipFile(sys.argv[1],'w') as z:
 if sys.argv[2]=='attack':
  entry=zipfile.ZipInfo('link');entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16
  z.writestr(entry,'../outside');z.writestr('link/escaped.txt','controlled test content')
 else:z.writestr('folder/safe.txt','safe fixture')
`;
for(const path of copies('@appium/support')){
 test(`${path}: ZIP symlinks cannot escape extraction root`,async()=>{
  const {zip}=require(resolve(root,path));const work=mkdtempSync(join(tmpdir(),'aiwardrobe-zip-security-'));
  try{
   const archive=join(work,'attack.zip'),output=join(work,'output');mkdirSync(join(work,'outside'));
   execFileSync('python3',['-c',fixturePython,archive,'attack']);
   await assert.rejects(zip.extractAllTo(archive,output),/Out of bound/);
   assert.equal(existsSync(join(work,'outside','escaped.txt')),false);
  }finally{rmSync(work,{recursive:true,force:true});}
 });
 test(`${path}: ordinary ZIP extraction still works`,async()=>{
  const {zip}=require(resolve(root,path));const work=mkdtempSync(join(tmpdir(),'aiwardrobe-zip-valid-'));
  try{
   const archive=join(work,'safe.zip'),output=join(work,'output');execFileSync('python3',['-c',fixturePython,archive,'safe']);
   await zip.extractAllTo(archive,output);assert.equal(readFileSync(join(output,'folder','safe.txt'),'utf8'),'safe fixture');
  }finally{rmSync(work,{recursive:true,force:true});}
 });
}
test('advisory checks found the expected dependency families',()=>{
 for(const name of ['proxy-addr','shell-quote','@appium/support'])assert.ok(copies(name).length>0,name);
});
for (const path of copies('fast-uri')) {
 test(`${path}: percent-encoded hostname case cannot evade canonical comparison`, () => {
  const uri=require(resolve(root,path));
  assert.equal(uri.parse('//%41.com').host,'a.com');
  assert.equal(uri.equal('//%41.com','//a.com'),true);
  assert.equal(uri.equal('//A.com','//a.com'),true);
  assert.equal(uri.equal('//other.com','//a.com'),false);
  assert.equal(uri.normalize('https://EXAMPLE.com/a'),'https://example.com/a');
 });
}
for (const path of copies('ip-address')) {
 test(`${path}: address families cannot cross subnet allowlists`, () => {
  const {Address4,Address6}=require(resolve(root,path));
  assert.equal(new Address6('a00::1').isInSubnet(new Address4('10.0.0.0/8')),false);
  assert.equal(new Address4('32.1.13.184').isInSubnet(new Address6('2001:db8::/32')),false);
  assert.equal(new Address4('10.1.2.3').isInSubnet(new Address4('10.0.0.0/8')),true);
  assert.equal(new Address6('2001:db8::1').isInSubnet(new Address6('2001:db8::/32')),true);
 });
 test(`${path}: oversized invalid IPv6 input is rejected within a bounded process`, () => {
  const code=`const {Address6}=require(${JSON.stringify(resolve(root,path))});try{new Address6(':'.repeat(1000000));process.exit(2);}catch{process.exit(0);}`;
  execFileSync(process.execPath,['-e',code],{timeout:3000,stdio:'pipe'});
 });
}

for (const path of copies('serialize-javascript')) {
 for (const shape of ['regexp','date']) {
  test(`${path}: ${shape} metadata cannot inject evaluated code`, () => {
   const serialize=require(resolve(root,path));let input;
   if(shape==='regexp'){
    input=Object.create(RegExp.prototype);
    Object.defineProperty(input,'source',{get:()=> 'x'});
    Object.defineProperty(input,'flags',{get:()=> '\"+(globalThis.INJECTED=true)+\"'});
    input.toJSON=()=> '@placeholder';
   }else{
    input=Object.create(Date.prototype);input.toISOString=()=> '\"+(globalThis.INJECTED=true)+\"';input.toJSON=()=> '2024-01-01';
   }
   const context={};let output;
   try{output=serialize({value:input});}
   catch(error){assert.equal(shape,'date');assert.match(error.message,/Invalid Date ISO string/);return;}
   try{vm.runInNewContext(`(${output})`,context,{timeout:1000});}catch{} // Invalid flags may throw, but must never execute the injected expression.
   assert.equal(context.INJECTED,undefined);
  });
 }
 test(`${path}: ordinary dates, regexps and functions retain compatibility`,()=>{
  const serialize=require(resolve(root,path));
  const output=serialize({date:new Date('2024-01-01T00:00:00Z'),pattern:/a+/gi,fn:function sum(a,b){return a+b;}});
  const value=vm.runInNewContext(`(${output})`,{}, {timeout:1000});
  assert.equal(value.date.toISOString(),'2024-01-01T00:00:00.000Z');assert.equal(value.pattern.test('AAA'),true);assert.equal(value.fn(2,3),5);
 });
}
test('Mocha serial and parallel modes retain test-runner compatibility',()=>{
 const directory=mkdtempSync(join(tmpdir(),'aiwardrobe-mocha-compat-'));
 try{
  for(const name of ['first','second'])writeFileSync(join(directory,name+'.cjs'),`const assert=require('node:assert/strict');describe('${name}',()=>{it('executes async test',async()=>{await Promise.resolve();assert.equal(2+2,4);});});`);
  for(const parallel of [false,true]){
   const args=[resolve(root,'node_modules/mocha/bin/mocha.js'),'--no-config','--no-package','--reporter','json',...(parallel?['--parallel','--jobs','2']:[]),join(directory,'first.cjs'),join(directory,'second.cjs')];
   const report=JSON.parse(execFileSync(process.execPath,args,{timeout:15000,encoding:'utf8'}));assert.equal(report.stats.passes,2);assert.equal(report.stats.failures,0);
  }
 }finally{rmSync(directory,{recursive:true,force:true});}
});
