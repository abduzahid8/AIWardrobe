import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import ts from 'typescript';
const root = resolve(new URL('../..', import.meta.url).pathname);
function sourceFiles(directory) {
  return readdirSync(directory,{withFileTypes:true}).flatMap(entry => {
    const path=join(directory,entry.name);
    return entry.isDirectory() ? sourceFiles(path) : /\.(ts|tsx)$/.test(entry.name) ? [path] : [];
  });
}
test('application image imports route through the owner-aware native/Expo boundary', () => {
  const files=['components','screens','src','features','navigation','hooks','constants','data','i18n','lib','store']
    .flatMap(dir=>sourceFiles(join(root,dir))).concat(join(root,'App.tsx'),join(root,'index.ts'));
  const bypasses=[];
  for(const file of files){
    if(relative(root,file)==='src/components/MediaImage.tsx')continue;
    const ast=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const node of ast.statements){
      if(!ts.isImportDeclaration(node)||!['react-native','expo-image'].includes(node.moduleSpecifier.text))continue;
      const clause=node.importClause,bindings=clause?.namedBindings;
      if(clause?.isTypeOnly||!bindings||!ts.isNamedImports(bindings))continue;
      if(bindings.elements.some(specifier=>!specifier.isTypeOnly&&(specifier.propertyName?.text??specifier.name.text)==='Image'))bypasses.push(relative(root,file));
    }
  }
  assert.deepEqual(bypasses,[],'A direct image import bypasses private media authorization/cache isolation');
});
