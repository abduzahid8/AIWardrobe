import React from 'react';
import { ActivityIndicator, Text, TouchableOpacity } from 'react-native';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import fs from 'fs';
import path from 'path';
import ts from 'typescript';
import en from '../../i18n/locales/en.json';

// Render the actual Home section's status expression without unrelated account,
// location and animation providers. The branch under test is not reimplemented.
const source=fs.readFileSync(path.resolve(__dirname,'../../screens/HomeScreen.tsx'),'utf8');
const tree=ts.createSourceFile('HomeScreen.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
let expression='';
function visit(node:ts.Node){
 if(ts.isVariableDeclaration(node)&&node.name.getText(tree)==='emptyStateContent') expression=node.initializer!.getText(tree);
 ts.forEachChild(node,visit);
}
visit(tree);if(!expression)throw new Error('Home status expression not found');
const bindings='React,ActivityIndicator,ScaledText,TouchableOpacity,Ionicons,isSwipeLocked,canGenerateNewLooks,colors,spacing,suggestionTextStyle,styles,t,dailyAI,stackExhausted,emptyIconName,Haptics,navigation,consumeOrPaywall,logger,style';
const compiled=ts.transpileModule(`function status(scope:any){const {${bindings}}=scope;return (${expression});}`,{compilerOptions:{target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.React}}).outputText;
const status=new Function(compiled+'\nreturn status;')();
function setup(loading=false,error:string|null='outfitMaker.budgetTooLow',canGenerateNewLooks=true){
 const regenerate=jest.fn();
 const scope={React,ActivityIndicator,ScaledText:Text,TouchableOpacity,Ionicons:()=>null,isSwipeLocked:false,canGenerateNewLooks,colors:{accent:{primary:'blue'},text:{tertiary:'gray'}},spacing:{md:12,xs:4},suggestionTextStyle:{},styles:{createAvatarButton:{},createAvatarText:{}},t:(key:string)=>key.split('.').reduce((value:any,part)=>value?.[part],en)||key,dailyAI:{loading,error,regenerate},stackExhausted:true,emptyIconName:'alert',Haptics:{impactAsync:jest.fn(),ImpactFeedbackStyle:{Medium:1}},navigation:{navigate:jest.fn()},consumeOrPaywall:async()=>true,logger:{debug:jest.fn()},style:'casual'};
 return {...render(status(scope)),regenerate,navigate:scope.navigation.navigate};
}
test('failed refresh on an exhausted deck shows the localized reason and a working retry, not a perpetual preparing state',async()=>{
 const view=setup();
 expect(view.getByText(en.outfitMaker.budgetTooLow)).toBeTruthy();
 expect(view.queryByText('Preparing new looks…')).toBeNull();
 fireEvent.press(view.getByLabelText(en.home.tryAgain));
 await waitFor(()=>expect(view.regenerate).toHaveBeenCalledTimes(1));
});
test('a running refresh still presents the preparing state',()=>{
 const view=setup(true,null);expect(view.getByText('Preparing new looks…')).toBeTruthy();
 expect(view.queryByText(en.outfitMaker.budgetTooLow)).toBeNull();
});
test('unmapped internal failures use local generic copy',()=>{
 const view=setup(false,'private_internal_detail');
 expect(view.getByText(en.outfitMaker.generateFailed)).toBeTruthy();expect(view.queryByText('private_internal_detail')).toBeNull();
});

test('free empty deck shows subscription access rather than retry or an endless spinner',()=>{
 const view=setup(false,'outfitMaker.subscriptionRequired',false);
 expect(view.getByText(en.outfitMaker.subscriptionRequired)).toBeTruthy();
 expect(view.queryByText('Preparing new looks…')).toBeNull();
 expect(view.queryByLabelText(en.home.tryAgain)).toBeNull();
 fireEvent.press(view.getByRole('button'));
 expect(view.navigate).toHaveBeenCalledWith('Paywall');
 expect(view.regenerate).not.toHaveBeenCalled();
});
