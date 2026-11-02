/** @jest-environment node */
// Execute the real screen handlers with controllable admission/provider promises.
const fs=require('fs'),ts=require('typescript');
const source=fs.readFileSync('screens/AIOutfitmaker.tsx','utf8');
const ast=ts.createSourceFile('screen.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
const names=['handleGeneratePress','cancelOutfitGeneration'];const declarations=[];
function visit(node){if(ts.isVariableDeclaration(node)&&names.includes(node.name.text))declarations.push(`const ${node.getText(ast)};`);ts.forEachChild(node,visit);}visit(ast);
const code=ts.transpileModule(declarations.join('\n'),{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{promise,resolve,reject};};
function setup(generationAvailable=true){
 let owner='owner-a',revision=1;
 const deps={generationAvailable,authEvents:{getUserId:()=>owner,getRevision:()=>revision},generationRequest:{current:null},generationNonce:{current:0},isMounted:{current:true},setLoading:jest.fn(),setError:jest.fn(),setOutfits:jest.fn(),setSelectedStyle:jest.fn(),consumeOrPaywall:jest.fn(async()=>true),generateOutfits:jest.fn(async()=>{}),Haptics:{notificationAsync:jest.fn(async()=>{}),NotificationFeedbackType:{Success:'success'}},OutfitGenerationFailure:class extends Error{},t:key=>key};
 const api=Function(...Object.keys(deps),code+';return {handleGeneratePress,cancelOutfitGeneration};')(...Object.values(deps));
 return{...deps,...api,switchOwner:()=>{owner='owner-b';revision++;}};
}
test('double taps during admission consume once and start one provider request',async()=>{
 const s=setup(),admission=deferred();s.consumeOrPaywall.mockReturnValue(admission.promise);
 const first=s.handleGeneratePress();await s.handleGeneratePress();expect(s.consumeOrPaywall).toHaveBeenCalledTimes(1);
 admission.resolve(true);await first;expect(s.generateOutfits).toHaveBeenCalledTimes(1);expect(s.generationRequest.current).toBeNull();
});
test('cancelling admission prevents a late approval from starting generation',async()=>{
 const s=setup(),admission=deferred();s.consumeOrPaywall.mockReturnValue(admission.promise);
 const first=s.handleGeneratePress();s.cancelOutfitGeneration();admission.resolve(true);await first;
 expect(s.generateOutfits).not.toHaveBeenCalled();expect(s.setLoading).toHaveBeenLastCalledWith(false);
});
test('cancel aborts the provider signal and an old completion cannot stop a retry',async()=>{
 const s=setup(),old=deferred(),retry=deferred();s.generateOutfits.mockReturnValueOnce(old.promise).mockReturnValueOnce(retry.promise);
 const first=s.handleGeneratePress();await Promise.resolve();const oldSignal=s.generateOutfits.mock.calls[0][1];
 s.cancelOutfitGeneration();expect(oldSignal.aborted).toBe(true);
 const second=s.handleGeneratePress();await Promise.resolve();const newController=s.generationRequest.current;s.setLoading.mockClear();
 old.resolve();await first;expect(s.generationRequest.current).toBe(newController);expect(s.setLoading).not.toHaveBeenCalled();
 retry.resolve();await second;expect(s.setLoading).toHaveBeenLastCalledWith(false);expect(s.generationRequest.current).toBeNull();
});
test('denial releases the single request lock so another attempt can run',async()=>{
 const s=setup();s.consumeOrPaywall.mockResolvedValueOnce(false);await s.handleGeneratePress();await s.handleGeneratePress();expect(s.generateOutfits).toHaveBeenCalledTimes(1);
});
test('an account switch during admission cannot start work for the new owner',async()=>{
 const s=setup(),admission=deferred();s.consumeOrPaywall.mockReturnValue(admission.promise);const pending=s.handleGeneratePress();s.switchOwner();admission.resolve(true);await pending;expect(s.generateOutfits).not.toHaveBeenCalled();expect(s.generationRequest.current).toBeNull();
});
test('unexpected generation failure unlocks retry and clears loading',async()=>{
 const s=setup();s.generateOutfits.mockRejectedValueOnce(new Error('failure'));await s.handleGeneratePress();expect(s.setError).toHaveBeenCalledWith('outfitMaker.generateFailed');expect(s.setLoading).toHaveBeenLastCalledWith(false);await s.handleGeneratePress();expect(s.generateOutfits).toHaveBeenCalledTimes(2);
});

test('unavailable choices cannot consume quota or start generation',async()=>{
 const s=setup(false);await s.handleGeneratePress();expect(s.consumeOrPaywall).not.toHaveBeenCalled();expect(s.generateOutfits).not.toHaveBeenCalled();
});

function generationFailureHandler(current) {
 let block;
 function find(node){if(ts.isCatchClause(node)&&node.block.getText(ast).includes("console.warn('[AIOutfitmaker] Edge function failed:'"))block=node.block.getText(ast);ts.forEachChild(node,find);}find(ast);
 const failureCode=ts.transpileModule(block,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
 const deps={fallbackTimer:{current:null},isCurrentGeneration:()=>current,OutfitGenerationFailure:class extends Error{},isMounted:{current:true},setOutfits:jest.fn(),setLoading:jest.fn(),setError:jest.fn(),t:key=>key};
 return {...deps,run:Function(...Object.keys(deps),'err',failureCode).bind(null,...Object.values(deps))};
}
test('a cancelled or stale edge request exits without a warning or failure UI',()=>{
 const warn=jest.spyOn(console,'warn').mockImplementation(()=>{});
 try{const s=generationFailureHandler(false);s.run(Object.assign(new Error('cancelled'),{name:'AbortError'}));expect(warn).not.toHaveBeenCalled();expect(s.setError).not.toHaveBeenCalled();expect(s.setLoading).not.toHaveBeenCalled();}finally{warn.mockRestore();}
});
test('an active mapped failure still shows its message without an Expo warning overlay',()=>{
 const warn=jest.spyOn(console,'warn').mockImplementation(()=>{});
 try{const s=generationFailureHandler(true);const error=new s.OutfitGenerationFailure();error.translationKey='outfitMaker.noMatchingClothing';s.run(error);expect(s.setError).toHaveBeenCalledWith(error.translationKey);expect(s.setLoading).toHaveBeenCalledWith(false);expect(warn).not.toHaveBeenCalled();}finally{warn.mockRestore();}
});
