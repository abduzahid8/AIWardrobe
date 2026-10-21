/** @jest-environment node */
const fs=require('fs'),ts=require('typescript');const ast=ts.createSourceFile('screen.tsx',fs.readFileSync('screens/AIOutfitmaker.tsx','utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
const expressions={};function walk(node){if(ts.isVariableDeclaration(node)&&['handleGeneratePress','isCurrentGeneration'].includes(node.name.getText(ast)))expressions[node.name.getText(ast)]=node.initializer.getText(ast);ts.forEachChild(node,walk);}walk(ast);
const make=(text,state)=>Function(...Object.keys(state),ts.transpileModule(`return (${text});`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText)(...Object.values(state));
function guardSetup(){let owner='owner',revision=1;const state={cancelSignal:undefined,isMounted:{current:true},generationNonce:{current:1},tapNonce:1,generationOwner:'owner',generationRevision:1,authEvents:{getUserId:()=>owner,getRevision:()=>revision}};return{state,guard:make(expressions.isCurrentGeneration,state),changeOwner:()=>{owner='other';revision++;},sameOwnerNewSession:()=>revision++};}
test('the actual generation guard requires the original owner, session, nonce and mount',()=>{
 const a=guardSetup();expect(a.guard()).toBe(true);a.changeOwner();expect(a.guard()).toBe(false);
 const b=guardSetup();b.sameOwnerNewSession();expect(b.guard()).toBe(false);
 const c=guardSetup();c.state.generationNonce.current++;expect(c.guard()).toBe(false);
 const d=guardSetup();d.state.isMounted.current=false;expect(d.guard()).toBe(false);
});
test('an account change while the quota gate is pending cannot dispatch generation',async()=>{
 let owner='owner',revision=1,finish;const state={generationRequest:{current:null},OutfitGenerationFailure:class extends Error{},t:key=>key,authEvents:{getUserId:()=>owner,getRevision:()=>revision},isMounted:{current:true},setLoading:jest.fn(),setError:jest.fn(),consumeOrPaywall:()=>new Promise(resolve=>{finish=resolve;}),Haptics:{notificationAsync:jest.fn().mockResolvedValue(undefined),NotificationFeedbackType:{Success:'success'}},setSelectedStyle:jest.fn(),generateOutfits:jest.fn()};
 const run=make(expressions.handleGeneratePress,state),pending=run();owner='other';revision++;finish(true);await pending;expect(state.generateOutfits).not.toHaveBeenCalled();expect(state.Haptics.notificationAsync).not.toHaveBeenCalled();
});
test('same-owner approved generation still proceeds',async()=>{
 const state={generationRequest:{current:null},OutfitGenerationFailure:class extends Error{},t:key=>key,authEvents:{getUserId:()=> 'owner',getRevision:()=>1},isMounted:{current:true},setLoading:jest.fn(),setError:jest.fn(),consumeOrPaywall:jest.fn().mockResolvedValue(true),Haptics:{notificationAsync:jest.fn().mockResolvedValue(undefined),NotificationFeedbackType:{Success:'success'}},setSelectedStyle:jest.fn(),generateOutfits:jest.fn()};await make(expressions.handleGeneratePress,state)('casual');expect(state.generateOutfits).toHaveBeenCalledWith('casual',expect.any(AbortSignal));
});
test('an aborted generation can never apply a late result',()=>{
 const a=guardSetup();const controller=new AbortController();a.state.cancelSignal=controller.signal;
 const guard=make(expressions.isCurrentGeneration,a.state);expect(guard()).toBe(true);controller.abort();expect(guard()).toBe(false);
});
