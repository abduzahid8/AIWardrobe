/** @jest-environment node */
const fs=require('node:fs'),path=require('node:path'),ts=require('typescript');
const source=ts.createSourceFile('AIOutfitmaker.tsx',fs.readFileSync(path.resolve(__dirname,'../../screens/AIOutfitmaker.tsx'),'utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
let handler;
function visit(node){if(ts.isVariableDeclaration(node)&&node.name.getText(source)==='handleGeneratePress')handler=node.initializer;else ts.forEachChild(node,visit);}
visit(source);
function mount(allowed=true){
 const bindings={authEvents:{getUserId:()=> 'owner',getRevision:()=>1},generationRequest:{current:null},isMounted:{current:true},
  requireFeature:jest.fn(()=>allowed),consumeOrPaywall:jest.fn(),generateOutfits:jest.fn(async()=>{}),
  setLoading:jest.fn(),setError:jest.fn(),setSelectedStyle:jest.fn(),t:key=>key,
  Haptics:{notificationAsync:async()=>{},NotificationFeedbackType:{Success:'success'}},OutfitGenerationFailure:class extends Error{}};
 const compiled=ts.transpileModule(`const run=${handler.getText(source)};`,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText;
 return {...bindings,run:new Function(...Object.keys(bindings),compiled+'\nreturn run;')(...Object.values(bindings))};
}
test('a failed generation checks access without incrementing usage',async()=>{
 const app=mount();app.generateOutfits.mockRejectedValue(new Error('generation failed'));
 await app.run();expect(app.requireFeature).toHaveBeenCalledWith('aiOutfits');
 expect(app.consumeOrPaywall).not.toHaveBeenCalled();expect(app.setError).toHaveBeenCalledWith('outfitMaker.generateFailed');
 expect(app.generationRequest.current).toBeNull();
});
test('a denied entitlement never starts generation or increments usage',async()=>{
 const app=mount(false);await app.run();expect(app.generateOutfits).not.toHaveBeenCalled();expect(app.consumeOrPaywall).not.toHaveBeenCalled();
});
test('the Generate handler leaves usage recording to successful publication',async()=>{
 const app=mount();await app.run('classic');expect(app.generateOutfits).toHaveBeenCalledWith('classic',expect.any(AbortSignal));
 expect(app.consumeOrPaywall).not.toHaveBeenCalled();expect(app.setLoading).toHaveBeenLastCalledWith(false);
});
