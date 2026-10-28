import fs from 'node:fs';
import ts from 'typescript';
import { needsLayering } from '../../features/outfit-generator/utils/styleInference';
import { isCompleteNormalizedOutfit } from '../../features/outfit-generator/utils/normalizeSlots';
import { needsLayering as needsLayeringShared } from '../../backend/supabase/functions/_shared/outfitComposition';
function loadFunction(path: string, name: string, bindings: Record<string, unknown> = {}) {
 const text=fs.readFileSync(path,'utf8');
 const file=ts.createSourceFile(path,text,ts.ScriptTarget.Latest,true);
 const node=file.statements.find(n=>ts.isFunctionDeclaration(n)&&n.name?.text===name)!;
 const args=Object.keys(bindings);
 const fn=Function(...args,ts.transpileModule(node.getText(file),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText+`;return ${name};`);
 return fn(...args.map(key=>bindings[key]));
}
const serverNeedsLayering=loadFunction('backend/supabase/functions/generate-outfits/index.ts','needsLayering',{needsLayeringShared});
const clientNeedsLayering=loadFunction('src/services/outfitGenerationService.ts','isLayeredWeather',{needsLayeringShared});
const pieces=[{id:'layer',name:'Linen blazer',macroCategory:'outerwear'},{id:'shirt',name:'Cotton shirt',macroCategory:'top'},{id:'pants',name:'Chino trousers',macroCategory:'bottom'},{id:'shoes',name:'Leather loafers',macroCategory:'shoes'}];
describe('required layer + shirt/T-shirt + pants + shoes',()=>{
 test.each([undefined,{temp:35,condition:'sunny'},{temp:5,condition:'snow'}])('every weather retains four slots: %j',weather=>{
  expect(serverNeedsLayering('classic',weather,'summer work')).toBe(true);
  expect(clientNeedsLayering(weather,'summer work')).toBe(true);
  expect(needsLayering('casual',weather,'summer work')).toBe(true);
 });
 test('three items cannot be displayed as complete',()=>expect(isCompleteNormalizedOutfit(pieces.slice(1))).toBe(false));
 test('four distinct categories are complete',()=>expect(isCompleteNormalizedOutfit(pieces)).toBe(true));
 test('a second layer cannot stand in for the base shirt',()=>expect(isCompleteNormalizedOutfit(pieces.map(p=>p.id==='shirt'?{...p,name:'Wool sweater'}:p))).toBe(false));
 test('shorts cannot stand in for pants',()=>expect(isCompleteNormalizedOutfit(pieces.map(p=>p.id==='pants'?{...p,name:'Cotton shorts'}:p))).toBe(false));
});

test('real Aurélien polo with Jeans in its title cannot fill the pants category',()=>{
 const shirtNamedJeans={id:'aurelien-7649888796889',name:'Jeans Terry Towelling Polo Shirt',type:'lower_body',macroCategory:'bottom'};
 expect(isCompleteNormalizedOutfit([pieces[0],pieces[1],shirtNamedJeans,pieces[3]])).toBe(false);
 expect(isCompleteNormalizedOutfit([pieces[0],{...shirtNamedJeans,macroCategory:'top',type:'polo shirt'},pieces[2],pieces[3]])).toBe(true);
});
