/** @jest-environment node */
const fs=require('fs'),ts=require('typescript'),React=require('react');
const {resolveShopQuote,outfitTotalQuote,formatMoney}=require('../../src/utils/outfitBudget');
const {formatUzsPrice}=require('../../src/utils/uzsPrice');
const {makeReferenceSnapshot}=require('../fixtures/budgetFx');
const source=fs.readFileSync(require.resolve('../../screens/AIOutfitmaker.tsx'),'utf8');
const ast=ts.createSourceFile('AIOutfitmaker.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);let gridCode,cellCode,totalCode;
for(const node of ast.statements){
 if(ts.isFunctionDeclaration(node)&&node.name?.text==='OutfitSlotGrid')gridCode=`const value=${node.getText(ast)};`;
 if(ts.isFunctionDeclaration(node)&&node.name?.text==='outfitTotalDisplay')totalCode=`const value=${node.getText(ast)};`;
 if(ts.isVariableStatement(node))for(const d of node.declarationList.declarations)if(d.name.getText(ast)==='SlotCell')cellCode=`const value=${d.initializer.arguments[0].getText(ast)};`;
}
function build(code,scope){if(!code)throw Error('Missing production price component');return Function(...Object.keys(scope),ts.transpile(code,{target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.React})+';return value;')(...Object.values(scope));}
const react={...React,useMemo:fn=>fn(),useCallback:fn=>fn,useState:()=>[false,()=>{}],useEffect:()=>{}};
const cell=build(cellCode,{React:react,View:'View',ExpoImage:'Image',ScaledText:'Text',TouchableOpacity:'Button',Ionicons:'Icon',slotStyles:{},formatMoney,useUzsPrice:(price,currency)=>formatUzsPrice(price,currency,makeReferenceSnapshot())});
const grid=build(gridCode,{React:react,View:'View',SlotCell:'Cell',useDesignTokens:()=>({}),useTranslation:()=>({t:(_key,fallback)=>fallback}),useCallback:fn=>fn,canonicalizeMacroCategory:x=>x,getSlotImageSource:()=>({uri:'https://fixture.invalid/coat.jpg'}),getShopDestination:()=>'',slotStyles:{},resolveShopQuote});
function text(node){if(!node)return[];if(Array.isArray(node))return node.flatMap(text);if(typeof node==='string')return[node];return text(node.props?.children);}
it('the actual grid and cell show the matching EUR quote at its original listing price and never relabel the amount',()=>{
 const result=grid({items:[{id:'shop_coat',name:'Coat',type:'outerwear',isShopItem:true,price:100}],priceCatalog:[{id:'other',price:50,currency:'USD'},{id:'coat',price:90,currency:'EUR'}]});
 const props=result.props.children[0].props;expect(props).toMatchObject({price:90,priceCurrency:'EUR'});
 // Stylist 6 redesign ("original-currency cards"): tiles always show the
 // original listing price + currency; a converted guess would mislabel the
 // listing. UZS conversion happens once, on the card total, with loaded rates.
 expect(text(cell(props))).toContain('€90');
 expect(text(cell(props))).not.toContain('UZS');
});
it('the actual cell does not label an unknown currency as USD',()=>{
 const result=grid({items:[{id:'shop_missing',name:'Coat',type:'outerwear',isShopItem:true,price:100}],priceCatalog:[{id:'other',price:50,currency:'USD'}]});
 const props=result.props.children[0].props;expect(props.priceCurrency).toBeUndefined();expect(text(cell(props))).not.toContain('$100');
});

const total=build(totalCode,{outfitTotalQuote,formatUzsPrice,formatMoney,makeReferenceSnapshot});
it('the card total sums same-currency quotes and converts with loaded rates',()=>{
 const items=[{id:'shop_a',name:'Shirt',isShopItem:true,price:100,currency:'EUR'},{id:'shop_b',name:'Coat',isShopItem:true,price:50,currency:'EUR'}];
 expect(outfitTotalQuote(items,[])).toEqual({price:150,currency:'EUR'});
 // The total now also carries `converted`, flagging the fx budget spend-cap
 // path; the plain total-quote path is always false even when display rates
 // converted the label to UZS.
 expect(total(items,[],makeReferenceSnapshot())).toEqual({label:'2 028 753 UZS',converted:false});
 expect(total(items,[],null)).toEqual({label:'€150',converted:false});
});
it('the card total is all-or-none across shop pieces and rejects mixed currencies',()=>{
 expect(total([{id:'shop_a',isShopItem:true,price:100,currency:'USD'},{id:'shop_c',isShopItem:true,price:25}],[])).toBeNull();
 expect(total([{id:'shop_a',isShopItem:true,price:100,currency:'USD'},{id:'shop_b',isShopItem:true,price:10,currency:'EUR'}],[])).toBeNull();
 expect(total([{id:'shop_a',isShopItem:true,price:100,currency:'USD'},{id:'shop_b',isShopItem:true,price:10,currency:'USD'}],[])).toEqual({label:'$110',converted:false});
});
it('owned wardrobe pieces cost zero and never invent a total',()=>{
 expect(outfitTotalQuote([{id:'w1',isWardrobe:true},{id:'shop_a',isShopItem:true,price:100,currency:'USD'}],[])).toEqual({price:100,currency:'USD'});
 expect(outfitTotalQuote([{id:'w1',isWardrobe:true},{id:'w2',isWardrobe:true}],[])).toBeNull();
});
