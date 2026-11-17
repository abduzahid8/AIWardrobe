#!/usr/bin/env node
/** Enrich only the reviewed 2026-11-16 men's footwear import.
 * Use --apply for idempotent metadata-only writes. No merchant SKU, price,
 * photo, link, source or subscription record is ever altered.
 * Run: node --import tsx scripts/stylist6/enrich_imported_footwear.mjs
 */
import crypto from 'node:crypto';
import cp from 'node:child_process';
import {createRequire} from 'node:module';
import {fashionIntelligence, FASHION_INTELLIGENCE_VERSION} from '../../src/domain/garmentDescriptor.ts';
const require=createRequire(import.meta.url);
require('dotenv').config({path:'.env',quiet:true});
const {createClient}=require('@supabase/supabase-js');
export const SOURCE='elegant-mens-shoes-20261116-product-only';
export const hashCatalogEvidence=row=>crypto.createHash('md5').update('['+
  [row.name,row.description,row.brand,row.category,row.garment_type,row.image_url]
    .map(x=>JSON.stringify(x??null)).join(', ')+']').digest('hex');
/** Materialize the exact trusted merchant-name search terms as a PostgreSQL
 * tsvector literal; all values come from the active catalog, not the client.
 * This is a derived index, NEVER an alteration to merchant source records. */
export function searchVector(row){
  const text=[row.name,row.brand,row.description].filter(Boolean).join(' ');
  const words=[...text.toLowerCase().matchAll(/[\p{L}\p{N}]+/gu)].map(x=>x[0]);
  const seen=new Map();
  words.forEach((word,i)=>{if(!seen.has(word))seen.set(word,Math.min(i+1,16383));});
  return [...seen].map(([word,pos])=>`'${word.replaceAll("'","''")}':${pos}`).join(' ');
}
/** Only use a manufacturer's explicit footwear noun in their own product
 * URL when the title and description did not identify the type. A model code,
 * 'brogue' (decoration), or a generic 'shoe' is NOT enough to guess a last.
 */
export function manufacturerUrlSubtype(row){
  try{
    const url=new URL(String(row.source_url||''));
    if(url.protocol!=='https:' || !url.hostname || !url.pathname.includes('/products/'))return null;
    const slug=decodeURIComponent(url.pathname.toLowerCase()).replace(/[^a-z0-9]+/g,' ');
    const options=[
      ['oxford',/\b(oxfords?|richelieus?|balmorals?)\b/],
      ['derby',/\b(derb(?:y|ies)|bluchers?)\b/],
      ['loafer',/\b(loafers?|mocassins?|penny loafers?)\b/],
      ['chelsea',/\bchelsea\b/],
      ['boot',/\b(chukkas?|boots?|bottines?)\b/],
      ['monk',/\b(monk straps?|double monks?|monks?)\b/],
    ];
    return options.find(([,re])=>re.test(slug))?.[0]||null;
  }catch{return null;}
}
export function derivedFootwear(row){
  if(row?.source!==SOURCE || row.category!=='shoes' || row.garment_type!=='shoes' || !row.is_active)
    throw Error('unexpected_source_or_category');
  const evidence={...row,macroCategory:'shoes',type:'shoes',isShopItem:true};
  let intelligence=fashionIntelligence(evidence);
  const manufacturerType=intelligence.descriptor.subtype==='unknown'
    ? manufacturerUrlSubtype(row):null;
  if(manufacturerType){
    // Derive from a literal merchant URL shoe-type noun, never a style guess.
    const adjusted=fashionIntelligence({...evidence,
      description:`${row.description||''} Manufacturer product type: ${manufacturerType}.`});
    intelligence={...adjusted,descriptor:{...adjusted.descriptor,
      evidence:[...adjusted.descriptor.evidence,`source:manufacturer-product-url:${manufacturerType}`]}};
  }
  if(intelligence.descriptor.slot!=='shoes' || intelligence.version!==FASHION_INTELLIGENCE_VERSION)
    throw Error('invalid_metadata_slot');
  return {product_id:row.id,version:FASHION_INTELLIGENCE_VERSION,source_hash:hashCatalogEvidence(row),
    slot:'shoes',subtype:intelligence.descriptor.subtype,formality:intelligence.descriptor.formality,
    search_document:searchVector(manufacturerType?{...row,description:`${row.description||''} ${manufacturerType}`}:row),intelligence};
}
async function main(){
  const url=process.env.EXPO_PUBLIC_SUPABASE_URL;
  if(!/^https:\/\/[a-z0-9-]+\.supabase\.co$/.test(url||''))throw Error('missing_project_url');
  const key=JSON.parse(cp.execFileSync('npx',['--yes','supabase','projects','api-keys',
    '--project-ref',new URL(url).hostname.split('.')[0],'--output','json'],
    {encoding:'utf8',timeout:30000,stdio:['ignore','pipe','pipe']}))
    .find(x=>x.name==='service_role')?.api_key;
  if(!key)throw Error('no_authenticated_service_role');
  const admin=createClient(url,key,{auth:{persistSession:false,autoRefreshToken:false}});
  const rows=[];
  for(let from=0;;from+=250){
    const {data,error}=await admin.from('shop_catalog')
      .select('id,name,description,brand,category,garment_type,image_url,source_url,source,is_active')
      .eq('source',SOURCE).order('id').range(from,from+249);
    if(error)throw Error('catalog_read_failed:'+error.message);
    rows.push(...data);
    if(data.length<250)break;
  }
  if(rows.length!==1534 || rows.some(x=>!x.is_active))
    throw Error(`source_changed_expected_1534_received_${rows.length}`);
  const metadata=rows.map(derivedFootwear);
  const subtypes=Object.fromEntries([...new Set(metadata.map(x=>x.subtype))].sort()
    .map(k=>[k,metadata.filter(x=>x.subtype===k).length]));
  const formalities=Object.fromEntries([...new Set(metadata.map(x=>x.formality))].sort()
    .map(k=>[k,metadata.filter(x=>x.formality===k).length]));
  console.log(JSON.stringify({mode:process.argv.includes('--apply')?'apply':'preflight',
    source:SOURCE,items:metadata.length,subtypes,formalities,
    unknownSubtype:subtypes.unknown||0,photoAndPriceUnchanged:true}));
  if(!process.argv.includes('--apply'))return;
  // One extra source-specific integrity gate before metadata writes.
  const {count,error:existingError}=await admin.from('catalog_garment_intelligence')
    .select('product_id',{count:'exact',head:true}).in('product_id',rows.slice(0,10).map(x=>x.id));
  if(existingError)throw Error('metadata_check_failed:'+existingError.message);
  if(count!==0 && count!==10)throw Error('unexpected_partial_metadata_sample');
  for(let i=0;i<metadata.length;i+=100){
    const records=metadata.slice(i,i+100);
    const {error}=await admin.from('catalog_garment_intelligence').upsert(records,{onConflict:'product_id'});
    if(error)throw Error(`metadata_upsert_batch_${i}:`+error.message);
    console.log(`enriched ${Math.min(i+100,metadata.length)}/${metadata.length}`);
  }
  const {count:verified,error:verifyError}=await admin.from('catalog_garment_intelligence')
    .select('product_id',{head:true,count:'exact'})
    .in('product_id',rows.slice(0,10).map(x=>x.id));
  if(verifyError||verified!==10)throw Error('metadata_write_verification_failed');
  console.log(JSON.stringify({applied:true,source:SOURCE,rows:metadata.length,verifiedSample:verified}));
}
if(process.argv[1]?.endsWith('enrich_imported_footwear.mjs'))main().catch(e=>{
  console.error(e.message);process.exitCode=1;
});
