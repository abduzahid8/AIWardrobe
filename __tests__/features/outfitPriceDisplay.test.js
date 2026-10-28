/** @jest-environment node */
const fs=require('fs'),ts=require('typescript'),React=require('react');
const {resolveShopQuote,formatMoney}=require('../../src/utils/outfitBudget');
const {formatUzsPrice}=require('../../src/utils/uzsPrice');
const {makeReferenceSnapshot}=require('../fixtures/budgetFx');
const source=fs.readFileSync(require.resolve('../../screens/AIOutfitmaker.tsx'),'utf8');
const ast=ts.createSourceFile('AIOutfitmaker.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);let gridCode,cellCode;
for(const node of ast.statements){
 if(ts.isFunctionDeclaration(node)&&node.name?.text==='OutfitSlotGrid')gridCode=`const value=${node.getText(ast)};`;
 if(ts.isVariableStatement(node))for(const d of node.declarationList.declarations)if(d.name.getText(ast)==='SlotCell')cellCode=`const value=${d.initializer.arguments[0].getText(ast)};`;
}
function build(code,scope){if(!code)throw Error('Missing production price component');return Function(...Object.keys(scope),ts.transpile(code,{target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.React})+';return value;')(...Object.values(scope));}
const react={...React,useMemo:fn=>fn(),useCallback:fn=>fn,useState:()=>[false,()=>{}],useEffect:()=>{}};
const cell=build(cellCode,{React:react,View:'View',ExpoImage:'Image',ScaledText:'Text',TouchableOpacity:'Button',Ionicons:'Icon',slotStyles:{},formatMoney,useUzsPrice:(price,currency)=>formatUzsPrice(price,currency,makeReferenceSnapshot())});
const grid=build(gridCode,{React:react,View:'View',SlotCell:'Cell',useDesignTokens:()=>({}),useTranslation:()=>({t:(_key,fallback)=>fallback}),useCallback:fn=>fn,canonicalizeMacroCategory:x=>x,getSlotImageSource:()=>({uri:'https://fixture.invalid/coat.jpg'}),getShopDestination:()=>'',slotStyles:{},resolveShopQuote});
function text(node){if(!node)return[];if(Array.isArray(node))return node.flatMap(text);if(typeof node==='string')return[node];return text(node.props?.children);}
it('the actual grid and cell convert the matching EUR quote to UZS instead of relabeling the amount',()=>{
 const result=grid({items:[{id:'shop_coat',name:'Coat',type:'outerwear',isShopItem:true,price:100}],priceCatalog:[{id:'other',price:50,currency:'USD'},{id:'coat',price:90,currency:'EUR'}]});
 const props=result.props.children[0].props;expect(props).toMatchObject({price:90,priceCurrency:'EUR'});expect(text(cell(props))).toContain('1 217 252 UZS');
});
it('the actual cell does not label an unknown currency as USD',()=>{
 const result=grid({items:[{id:'shop_missing',name:'Coat',type:'outerwear',isShopItem:true,price:100}],priceCatalog:[{id:'other',price:50,currency:'USD'}]});
 const props=result.props.children[0].props;expect(props.priceCurrency).toBeUndefined();expect(text(cell(props))).not.toContain('$100');
});
