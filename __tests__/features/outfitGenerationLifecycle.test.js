/** @jest-environment node */
// Execute the real screen handlers with controllable admission/provider promises.
const fs=require('fs'),ts=require('typescript');
const source=fs.readFileSync('screens/AIOutfitmaker.tsx','utf8');
const ast=ts.createSourceFile('screen.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
const names=['handleGeneratePress','cancelOutfitGeneration'];const declarations=[];
function visit(node){if(ts.isVariableDeclaration(node)&&names.includes(node.name.text))declarations.push(`const ${node.getText(ast)};`);ts.forEachChild(node,visit);}visit(ast);
const code=ts.transpileModule(declarations.join('\n'),{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{promise,resolve,reject};};
function setup(authenticated=true){
 let owner=authenticated ? 'owner-a' : null,revision=1;
 // v209: the press handler checks a synchronous feature gate (requireFeature); quota consumption
 // moved post-success into generateOutfits' publication gate (consumeOrPaywall with cancelSignal).
 const deps={authEvents:{getUserId:()=>owner,getRevision:()=>revision},generationRequest:{current:null},generationNonce:{current:0},isMounted:{current:true},setLoading:jest.fn(),setError:jest.fn(),setOutfits:jest.fn(),setSelectedStyle:jest.fn(),requireFeature:jest.fn(()=>true),consumeOrPaywall:jest.fn(async()=>true),generateOutfits:jest.fn(async()=>{}),Haptics:{notificationAsync:jest.fn(async()=>{}),NotificationFeedbackType:{Success:'success'}},OutfitGenerationFailure:class extends Error{},t:key=>key};
 const api=Function(...Object.keys(deps),code+';return {handleGeneratePress,cancelOutfitGeneration};')(...Object.values(deps));
 return{...deps,...api,switchOwner:()=>{owner='owner-b';revision++;}};
}
test('double taps start one provider request and consume no quota at press time',async()=>{
 const s=setup(),first=s.handleGeneratePress();await s.handleGeneratePress();await first;
 expect(s.generateOutfits).toHaveBeenCalledTimes(1);expect(s.generationRequest.current).toBeNull();
 // Redesign v209 quota-after-success: dispatch itself must not consume quota; consumption
 // happens only inside generateOutfits after a validated deck is ready to publish.
 expect(s.consumeOrPaywall).not.toHaveBeenCalled();
});
test('cancelling an in-flight generation frees the lock and a late completion cannot surface errors',async()=>{
 const s=setup(),first=s.handleGeneratePress(); // v209 dispatches immediately — no admission gate before the provider call
 expect(s.generateOutfits).toHaveBeenCalledTimes(1);
 s.cancelOutfitGeneration();expect(s.generationRequest.current).toBeNull();
 s.setError.mockClear();s.setLoading.mockClear();
 await first;expect(s.setError).not.toHaveBeenCalled();expect(s.setLoading).not.toHaveBeenCalled();
 await s.handleGeneratePress();expect(s.generateOutfits).toHaveBeenCalledTimes(2); // lock was freed by cancel
});
test('cancel aborts the provider signal and an old completion cannot stop a retry',async()=>{
 const s=setup(),old=deferred(),retry=deferred();s.generateOutfits.mockReturnValueOnce(old.promise).mockReturnValueOnce(retry.promise);
 const first=s.handleGeneratePress();await Promise.resolve();const oldSignal=s.generateOutfits.mock.calls[0][1];
 s.cancelOutfitGeneration();expect(oldSignal.aborted).toBe(true);
 const second=s.handleGeneratePress();await Promise.resolve();const newController=s.generationRequest.current;s.setLoading.mockClear();
 old.resolve();await first;expect(s.generationRequest.current).toBe(newController);expect(s.setLoading).not.toHaveBeenCalled();
 retry.resolve();await second;expect(s.setLoading).toHaveBeenLastCalledWith(false);expect(s.generationRequest.current).toBeNull();
});
test('a denied dispatch releases the single request lock so another attempt can run',async()=>{
 // Redesign v209: press-level denial is now the feature gate (quota denial moved post-success).
 const s=setup();s.requireFeature.mockReturnValueOnce(false);
 await s.handleGeneratePress();expect(s.generateOutfits).not.toHaveBeenCalled();expect(s.generationRequest.current).toBeNull();
 await s.handleGeneratePress();expect(s.generateOutfits).toHaveBeenCalledTimes(1);
});
test('an account switch during an in-flight generation cannot surface stale errors for the new owner',async()=>{
 // v209 dispatches immediately, so the switch happens while the provider request is in flight;
 // the current() guard keeps the late failure invisible to the new owner and the lock is freed.
 const s=setup(),pendingGen=deferred();s.generateOutfits.mockReturnValueOnce(pendingGen.promise);
 const first=s.handleGeneratePress();await Promise.resolve();
 s.switchOwner();s.setError.mockClear();
 pendingGen.reject(new Error('boom'));await first;
 expect(s.setError).not.toHaveBeenCalled();expect(s.generationRequest.current).toBeNull();
 await s.handleGeneratePress();expect(s.generateOutfits).toHaveBeenCalledTimes(2); // the new owner can start fresh
});
test('unexpected generation failure unlocks retry and clears loading',async()=>{
 const s=setup();s.generateOutfits.mockRejectedValueOnce(new Error('failure'));await s.handleGeneratePress();expect(s.setError).toHaveBeenCalledWith('outfitMaker.generateFailed');expect(s.setLoading).toHaveBeenLastCalledWith(false);await s.handleGeneratePress();expect(s.generateOutfits).toHaveBeenCalledTimes(2);
});

test('signed-out requests cannot consume quota or start generation',async()=>{
 const s=setup(false);await s.handleGeneratePress();expect(s.consumeOrPaywall).not.toHaveBeenCalled();expect(s.generateOutfits).not.toHaveBeenCalled();
});

test('a cancelled mapped provider failure cannot overwrite a retry',async()=>{
 const s=setup(),pending=deferred();s.generateOutfits.mockReturnValueOnce(pending.promise);
 const first=s.handleGeneratePress();await Promise.resolve();s.cancelOutfitGeneration();s.setError.mockClear();
 const error=new s.OutfitGenerationFailure();error.translationKey='outfitMaker.noMatchingClothing';
 pending.reject(error);await first;expect(s.setError).not.toHaveBeenCalled();
});
test('an active mapped provider failure shows localized error and clears loading',async()=>{
 const s=setup(),error=new s.OutfitGenerationFailure();error.translationKey='outfitMaker.noMatchingClothing';
 s.generateOutfits.mockRejectedValueOnce(error);await s.handleGeneratePress();
 expect(s.setError).toHaveBeenCalledWith(error.translationKey);expect(s.setLoading).toHaveBeenLastCalledWith(false);
});
