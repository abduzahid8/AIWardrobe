/** Installed assertion/visual-service integration. No browser session or screenshots. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {expect as assertion} from 'expect-webdriverio';
import {_setGlobal,expect as globalExpect} from '@wdio/globals';
import VisualService from '@wdio/visual-service';

test('visual service registers matchers on the active WebDriver assertion instance',async()=>{
 const directory=await mkdtemp(join(tmpdir(),'aiwardrobe-visual-interop-'));
 const previous=process.cwd();process.chdir(directory);
 try{
  _setGlobal('expect',assertion,false);
  const browser=Object.assign(new EventEmitter(),{
   capabilities:{browserName:'Safari',browserVersion:'1',platformName:'macOS'},
   requestedCapabilities:{browserName:'Safari'},isMultiremote:false,isMobile:false,isIOS:false,isAndroid:false,isBidi:false,
   execute:async()=>1,
   addCommand(name,command){this[name]=command;},
  });
  const service=new VisualService({baselineFolder:join(directory,'baseline'),screenshotPath:join(directory,'screens')},undefined,{framework:'mocha'});
  await service.before(browser.capabilities,[],browser);
  assert.equal(typeof browser.checkScreen,'function');
  assert.equal(typeof globalExpect(browser).toMatchScreenSnapshot,'function');
  const folders={baseline:'controlled baseline',actual:'controlled actual',diff:'controlled diff'};
  browser.checkScreen=async()=>({misMatchPercentage:0,folders});
  await globalExpect(browser).toMatchScreenSnapshot('matching');
  browser.checkScreen=async()=>({misMatchPercentage:4,folders});
  await assert.rejects(globalExpect(browser).toMatchScreenSnapshot('different'),/mismatch percentage/);
 }finally{process.chdir(previous);globalThis._wdioGlobals?.delete('expect');await rm(directory,{recursive:true,force:true});}
});
