/** @jest-environment node */
const fs=require('fs'),ts=require('typescript');
const source=fs.readFileSync(require.resolve('../../screens/WardrobeVideoScreen.tsx'),'utf8');
const file=ts.createSourceFile('WardrobeVideoScreen.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);let expression;
function visit(node){if(ts.isCallExpression(node)&&node.expression.getText(file)==='useEffect'&&node.arguments[0]?.getText(file).includes('subscribeScanReviews'))expression=node.arguments[0].getText(file);ts.forEachChild(node,visit);}visit(file);
if(!expression)throw Error('Missing production restore effect');
const compiled=ts.transpile(`const effect=${expression};`,{target:ts.ScriptTarget.ES2020});
const tick=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
function setup({reviews=[{id:'review',count:2}],pendingMedia=false,existing=null}={}){
 let owner='owner',notify;
 const loaded={reviewId:'review',detectedItems:[{itemType:'jacket'}]};
 const scope={reviewOwner:'owner',Platform:{OS:'ios'},reviewRestorePending:{current:false},queueGate:{current:{analyzing:false,results:existing}},initialMediaPending:{current:pendingMedia},
 listScanReviews:jest.fn().mockResolvedValue(reviews),loadScanReview:jest.fn().mockResolvedValue(loaded),setSavedReviews:jest.fn(),setUnavailableReviews:jest.fn(),reviewListOwner:{current:undefined},restoreReview:jest.fn(),reset:jest.fn(),authEvents:{getUserId:()=>owner},Alert:{alert:jest.fn()},t:key=>key,uploadQueueProcessor:{processQueue:jest.fn()},subscribeScanReviews:listener=>{notify=listener;return jest.fn();}};
 const effect=Function(...Object.keys(scope),`${compiled} return effect;`)(...Object.values(scope));const stop=effect();
 return{scope,loaded,stop,notify:()=>notify(),switchOwner:()=>{owner='other';}};
}
it('startup restores a committed review and sets the immediate queue gate before processing',async()=>{
 const app=setup();await tick();expect(app.scope.restoreReview).toHaveBeenCalledWith(app.loaded);expect(app.scope.queueGate.current.results).toBe(app.loaded);expect(app.scope.reviewRestorePending.current).toBe(false);expect(app.scope.uploadQueueProcessor.processQueue).toHaveBeenCalled();app.stop();
});
it('a new selected capture is not replaced by an older saved review',async()=>{
 const app=setup({pendingMedia:true});await tick();expect(app.scope.restoreReview).not.toHaveBeenCalled();expect(app.scope.loadScanReview).not.toHaveBeenCalled();app.stop();
});
it('acknowledging the final item clears the active review rather than leaving saved candidates visible',async()=>{
 const app=setup({reviews:[],existing:{reviewId:'review',detectedItems:[{}]}});await tick();expect(app.scope.reset).toHaveBeenCalled();expect(app.scope.queueGate.current.results).toBeNull();app.stop();
});
it('a late load after an account change cannot publish private review content',async()=>{
 const app=setup();let finish;app.scope.loadScanReview.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));await tick();app.switchOwner();finish(app.loaded);await tick();expect(app.scope.restoreReview).not.toHaveBeenCalled();app.stop();
});
it('unmount prevents an in-flight restore from publishing data',async()=>{
 const app=setup();let finish;app.scope.loadScanReview.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));await tick();app.stop();finish(app.loaded);await tick();expect(app.scope.restoreReview).not.toHaveBeenCalled();
});
it('reports unavailable records while restoring the valid review',async()=>{
 const app=setup();await tick();
 app.scope.listScanReviews.mockImplementationOnce(async(owner,report)=>{report(1);return[{id:'review',count:2}];});
 app.notify();await tick();expect(app.scope.setUnavailableReviews).toHaveBeenLastCalledWith(1);expect(app.scope.restoreReview).toHaveBeenLastCalledWith(app.loaded);app.stop();
});
