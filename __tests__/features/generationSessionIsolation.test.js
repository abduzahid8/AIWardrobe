/** @jest-environment node */
const fs=require('fs'),ts=require('typescript');const ast=ts.createSourceFile('screen.tsx',fs.readFileSync('screens/AIOutfitmaker.tsx','utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
const expressions={};let gateBlock='';
function walk(node){
 if(ts.isVariableDeclaration(node)&&['handleGeneratePress','isCurrentGeneration'].includes(node.name.getText(ast)))expressions[node.name.getText(ast)]=node.initializer.getText(ast);
 if(ts.isIfStatement(node)&&node.getText(ast).includes('consumeOrPaywall')&&node.getText(ast).includes('isCurrentGeneration')){
  const stmts=node.parent.statements,idx=stmts.indexOf(node);
  const end=stmts.findIndex((st,i)=>i>idx&&st.getText(ast).startsWith('setLoading(false)'));
  gateBlock=stmts.slice(idx,end+1).map(st=>st.getText(ast)).join('\n');
 }
 ts.forEachChild(node,walk);
}
walk(ast);
const make=(text,state)=>Function(...Object.keys(state),ts.transpileModule(`return (${text});`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText)(...Object.values(state));
const makeBlock=(text,state)=>Function(...Object.keys(state),ts.transpileModule(`return async()=>{${text}}`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText)(...Object.values(state));
function guardSetup(){let owner='owner',revision=1;const state={cancelSignal:undefined,isMounted:{current:true},generationNonce:{current:1},tapNonce:1,generationOwner:'owner',generationRevision:1,authEvents:{getUserId:()=>owner,getRevision:()=>revision}};return{state,guard:make(expressions.isCurrentGeneration,state),changeOwner:()=>{owner='other';revision++;},sameOwnerNewSession:()=>revision++};}
test('the actual generation guard requires the original owner, session, nonce and mount',()=>{
 const a=guardSetup();expect(a.guard()).toBe(true);a.changeOwner();expect(a.guard()).toBe(false);
 const b=guardSetup();b.sameOwnerNewSession();expect(b.guard()).toBe(false);
 const c=guardSetup();c.state.generationNonce.current++;expect(c.guard()).toBe(false);
 const d=guardSetup();d.state.isMounted.current=false;expect(d.guard()).toBe(false);
});
test('an account change while the quota gate is pending cannot publish the deck',async()=>{
 // Redesign v209 moved the quota gate post-success into generateOutfits: approval now sits on the
 // same line as an isCurrentGeneration re-check, so a switched account can never publish.
 if(!gateBlock)throw new Error('quota gate statement not found in screens/AIOutfitmaker.tsx');
 const startGate=(ownerRef,consume)=>{
  const cancelSignal=undefined;
  const guardState={cancelSignal,isMounted:{current:true},generationNonce:{current:1},tapNonce:1,generationOwner:ownerRef.name,generationRevision:ownerRef.revision,authEvents:{getUserId:()=>ownerRef.name,getRevision:()=>ownerRef.revision}};
  const isCurrentGeneration=make(expressions.isCurrentGeneration,guardState);
  const pub={setResultBudget:jest.fn(),shopBudget:{amount:120,currency:'USD'},setInventoryNote:jest.fn(),data:{generationVersion:'stylist-5-test'},t:key=>key,setLastEngineVersion:jest.fn(),setLastEngineDetail:jest.fn(),s6Active:false,s6CloudActive:false,shownOutfits:[{items:['look']}],setOutfits:jest.fn(),recordDisplayedDeck:jest.fn(async()=>{}),scope:'anonymous',requestId:'req-1',console,recordShownOutfits:jest.fn(),recencyScope:'anonymous',__DEV__:false,setLoading:jest.fn()};
  const pending=makeBlock(gateBlock,{consumeOrPaywall:consume,cancelSignal,isCurrentGeneration,...pub})();
  return{pub,pending};
 };
 // denial never publishes
 {const{pub,pending}=startGate({name:'owner',revision:1},async()=>false);await pending;
  expect(pub.setOutfits).not.toHaveBeenCalled();expect(pub.setLoading).not.toHaveBeenCalled();expect(pub.recordDisplayedDeck).not.toHaveBeenCalled();}
 // an account change while the gate is pending never publishes, even when approval lands late
 {const ownerRef={name:'owner',revision:1};let finish;const{pub,pending}=startGate(ownerRef,()=>new Promise(resolve=>{finish=resolve;}));
  ownerRef.name='other';ownerRef.revision++;finish(true);await pending;
  expect(pub.setOutfits).not.toHaveBeenCalled();expect(pub.setLoading).not.toHaveBeenCalled();expect(pub.recordDisplayedDeck).not.toHaveBeenCalled();expect(pub.setResultBudget).not.toHaveBeenCalled();}
 // same-owner approval still publishes the validated deck
 {const{pub,pending}=startGate({name:'owner',revision:1},async()=>true);await pending;
  expect(pub.setResultBudget).toHaveBeenCalled();expect(pub.setOutfits).toHaveBeenCalledWith([{items:['look']}]);expect(pub.setLoading).toHaveBeenLastCalledWith(false);}
 // cloud already debits the authoritative server quota; its successful deck
 // must publish without a second client usage RPC.
 {const count=jest.fn(async()=>true);const{pub,pending}=startGate({name:'owner',revision:1},count);pub.s6CloudActive=true;await pending;
  expect(count).not.toHaveBeenCalled();expect(pub.setOutfits).toHaveBeenCalledWith([{items:['look']}]);}
});
test('same-owner generation still proceeds when the feature gate passes',async()=>{
 // Redesign v209 replaced the pre-dispatch quota admission with a synchronous feature gate;
 // quota is consumed post-success inside generateOutfits (consumeOrPaywall no longer runs here).
 const state={generationRequest:{current:null},OutfitGenerationFailure:class extends Error{},t:key=>key,authEvents:{getUserId:()=>'owner',getRevision:()=>1},isMounted:{current:true},setLoading:jest.fn(),setError:jest.fn(),requireFeature:jest.fn(()=>true),Haptics:{notificationAsync:jest.fn().mockResolvedValue(undefined),NotificationFeedbackType:{Success:'success'}},setSelectedStyle:jest.fn(),generateOutfits:jest.fn()};
 await make(expressions.handleGeneratePress,state)('casual');
 expect(state.generateOutfits).toHaveBeenCalledWith('casual',expect.any(AbortSignal));
 expect(state.setSelectedStyle).toHaveBeenCalledWith('casual');
 expect(state.Haptics.notificationAsync).toHaveBeenCalledWith('success');
 expect(state.generationRequest.current).toBeNull();
});
test('an aborted generation can never apply a late result',()=>{
 const a=guardSetup();const controller=new AbortController();a.state.cancelSignal=controller.signal;
 const guard=make(expressions.isCurrentGeneration,a.state);expect(guard()).toBe(true);controller.abort();expect(guard()).toBe(false);
});
