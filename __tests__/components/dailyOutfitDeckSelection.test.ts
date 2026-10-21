import fs from 'fs';
import path from 'path';
import ts from 'typescript';
const source=fs.readFileSync(path.resolve(__dirname,'../../screens/HomeScreen.tsx'),'utf8');
const tree=ts.createSourceFile('HomeScreen.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
let declaration='';
function visit(node:ts.Node){
 if(ts.isVariableDeclaration(node)&&node.name.getText(tree)==='data'&&node.initializer?.getText(tree).includes('aiOutfits')) declaration=node.getText(tree);
 ts.forEachChild(node,visit);
}
visit(tree);if(!declaration)throw new Error('Daily deck source selection not found');
const compiled=ts.transpileModule(`function select(aiOutfits:any[],style:string,curatedCombinations:any[]){const ${declaration};return data;}`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
const select=new Function(compiled+'\nreturn select;')();
test('empty AI result cannot fall through to a locally assembled catalog deck',()=>{
 expect(select([],'casual',[{id:'local-fallback',mainTop:{id:'other-product'}}])).toEqual([]);
});
test('accepted AI variants keep their exact outfit reference without replacement variants',()=>{
 const outfit={id:'accepted',items:[{id:'selected-product'}]};
 const result=select([outfit],'casual',[{id:'other'}]);
 expect(result).toHaveLength(1);expect(result[0].outfit).toBe(outfit);expect(result[0].id).toContain('accepted');
});
