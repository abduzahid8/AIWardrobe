const fs=require('fs'),path=require('path'),ts=require('/Users/v.zahid/AIWardrobe/node_modules/typescript');
const out={};new Function('exports',ts.transpileModule(fs.readFileSync('/Users/v.zahid/AIWardrobe/backend/supabase/functions/_shared/outfitDiversity.ts','utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText)(out);
const root='/tmp/stylist-shoe-rotation/',history=JSON.parse(fs.readFileSync(root+'history-after.json','utf8')),pool=JSON.parse(fs.readFileSync(root+'pool-after.json','utf8'));
const admitted=pool.rows.filter(r=>r.admitted),byId=new Map(pool.rows.map(r=>[r.id,r]));
const groups=keys=>{const result=[];for(const key of keys){if(key.startsWith('id:'))result.push({id:key.slice(3),keys:[]});if(result.length)result.at(-1).keys.push(key);}return result;};
const results=history.decks.map((deck,g)=>{const products=groups(deck.productKeys),shoes=products.filter(p=>byId.has(p.id)),clothing=products.filter(p=>!byId.has(p.id));const prior=g?history.decks[g-1]:history.olderBeforeRun.at(-1);const priorKeys=new Set(prior?.productKeys||[]);
 const overlap=products.filter(p=>p.keys.some(k=>priorKeys.has(k))),shoeOverlap=shoes.filter(p=>p.keys.some(k=>priorKeys.has(k)));
 const before=[...history.olderBeforeRun,...history.decks.slice(0,g)].slice(-12);
 return {generation:g+1,requestId:deck.requestId,scope:history.scope,cards:products.length/4,itemsPerCard:[4,4,4],shoeIds:shoes.map(p=>p.id),shoes:shoes.map(p=>({...p,name:byId.get(p.id).name})),clothingIds:clothing.map(p=>p.id),clothingOverlap:overlap.filter(p=>!byId.has(p.id)),previousShoeOverlap:shoeOverlap.length,validPhysicalShoesAvailable:pool.admittedPhysicalShoes,theoreticalMinimum:Math.max(0,6-pool.admittedPhysicalShoes),admittedShoes:admitted.map(p=>{let last=-1,count=0;before.forEach((d,i)=>{const ks=new Set(d.productKeys);if(p.physicalKeys.some(k=>ks.has(k))){last=i;count++;}});return {id:p.id,lastUse:last,appearances:count,inPrevious:p.physicalKeys.some(k=>priorKeys.has(k)),selected:shoes.some(s=>s.id===p.id)}})};
});

const compare=(a,b)=>{for(let i=0;i<Math.max(a.length,b.length);i++){const d=(a[i]||0)-(b[i]||0);if(d)return d;}return 0;};
for(const r of results){
 const order=stats=>{const ages=Array(13).fill(0),counts=Array(13).fill(0);for(const p of stats){ages[p.lastUse+1]++;counts[p.appearances]++;}return [-stats.filter(p=>p.inPrevious).length,...ages,...counts];};
 const selected=r.admittedShoes.filter(p=>p.selected);let best=null;
 for(let a=0;a<r.admittedShoes.length;a++)for(let b=a+1;b<r.admittedShoes.length;b++)for(let c=b+1;c<r.admittedShoes.length;c++){
  const value=order([r.admittedShoes[a],r.admittedShoes[b],r.admittedShoes[c]]);if(!best||compare(value,best)>0)best=value;
 }
 r.fairness={matchesUnconstrainedLRUOrder:compare(order(selected),best||[])===0,selectedOrder:order(selected),bestOrder:best};
}
const used=new Set(results.flatMap(r=>r.shoeIds));const checks={fiveDisplayedDecks:results.length===5,threeCards:results.length===5&&results.every(r=>r.cards===3),fourItems:results.every(r=>r.itemsPerCard.every(n=>n===4)),scopeStable:new Set(results.map(r=>r.scope)).size===1,uniqueRequestIds:new Set(results.map(r=>r.requestId)).size===results.length,minimumShoeOverlap:results.every(r=>r.previousShoeOverlap===r.theoreticalMinimum),clothingRotation:results.every(r=>r.clothingOverlap.length===0),allAdmissibleShoesUsed:admitted.every(p=>used.has(p.id)),fairLRU:results.every(r=>r.fairness.matchesUnconstrainedLRUOrder)};
const report={source:'Final accepted displayed decks persisted by recordDisplayedDeck after client validation',capturedAt:new Date().toISOString(),poolSize:pool.admittedPhysicalShoes,checks,generations:results};
fs.writeFileSync(root+'real-app-proof.json',JSON.stringify(report,null,2));console.log(JSON.stringify({checks,generations:results.map(r=>({generation:r.generation,requestId:r.requestId,shoes:r.shoes.map(p=>p.name),overlap:r.previousShoeOverlap,clothingOverlap:r.clothingOverlap.length}))},null,2));if(!Object.values(checks).every(Boolean))process.exitCode=1;
