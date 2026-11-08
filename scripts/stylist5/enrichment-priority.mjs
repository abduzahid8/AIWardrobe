/** Deterministic priority queue from the frozen real inventory. Files stay private. */
import fs from 'node:fs';import path from 'node:path';
const dir=process.argv[2];if(!dir)throw Error('Provide private freeze directory');
const rows=JSON.parse(fs.readFileSync(path.join(dir,'catalog.json'))),frequency=new Map();
for(let n=0;n<6;n++)for(const wrapped of JSON.parse(fs.readFileSync(path.join(dir,`pool-${n}.json`)))){const row=wrapped.catalog_stylist_candidates||wrapped;frequency.set(row.id,(frequency.get(row.id)||0)+1);}
const valid=rows.filter(r=>/^https:\/\//.test(r.image_url||''));
const shoes=valid.filter(r=>r.fashion_slot==='shoes');
const rest=valid.filter(r=>r.fashion_slot!=='shoes').map(r=> {
 const peers=valid.filter(p=>p.fashion_slot===r.fashion_slot&&p.currency===r.currency&&p.price>0).sort((a,b)=>a.price-b.price);
 const median=peers[Math.floor(peers.length/2)]?.price||r.price;
 const d=r.fashion_intelligence?.descriptor||{};
 return {row:r,priority:(frequency.get(r.id)||0)*3+Math.min(2,median/r.price)+(r.visual_fashion?0:1),family:[r.fashion_slot,d.subtype,d.color,r.brand].join(':')};
}).sort((a,b)=>b.priority-a.priority||a.row.id.localeCompare(b.row.id));
const selected=[],counts=new Map();
// Broad subtype/brand/color capsule, then frequent usable-pool members.
for(const e of rest){if((counts.get(e.family)||0)>=2)continue;selected.push(e.row);counts.set(e.family,(counts.get(e.family)||0)+1);if(selected.length===256)break;}
const shape=r=>({id:r.id,image_url:r.image_url,source_hash:r.source_hash,slot:r.fashion_slot});
for(const [name,list] of [['shoes',shoes],['garments',selected]])fs.writeFileSync(path.join(dir,name+'-images.json'),JSON.stringify(list.map(shape)),{mode:0o600});
console.log({shoes:shoes.length,garments:selected.length,allFootwearPrioritized:true});
