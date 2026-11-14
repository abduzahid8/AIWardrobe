/** Prepare one atomic derived-cache import. No prices/products are modified.
 * Local mode uses the isolated benchmark database. Production mode belongs to
 * the guarded release and is never invoked automatically by this script. */
import fs from 'node:fs';import path from 'node:path';import cp from 'node:child_process';
const dir=process.argv[2],mode=process.argv[3]||'prepare';if(!dir)throw Error('Provide private batch directory');
const inputs=process.argv.slice(4);const files=inputs.length?inputs:['shoes.jsonl','garments.jsonl'];
if(files.length>8)throw Error('At most eight bounded batch journals per release');
const rows=new Map();
for(const name of files){const bytes=fs.readFileSync(path.join(dir,name),'utf8');const lines=bytes.trim().split('\n').filter(Boolean);if(Buffer.byteLength(bytes)>16*1024*1024||lines.length>256)throw Error('batch journal exceeds bounds');for(const line of lines){
 const r=JSON.parse(line),p=r.profile;
 if(!r.product_id||p.version!==1||!(/^[a-f0-9]{40}$/.test(p.revision))||!(/^[a-f0-9]{64}$/.test(p.imageSha256))||!(/^[a-f0-9]{32}$/.test(p.sourceHash))||!Array.isArray(r.embedding)||r.embedding.length!==768||r.embedding.some(n=>!Number.isFinite(n))||JSON.stringify(p).length>16000)throw Error('invalid image evidence');
 rows.set(r.product_id,r);
}}
const q=value=>"'"+JSON.stringify(value).replaceAll("'","''")+"'::jsonb";
const sql=`create temporary table imported on commit drop as select * from jsonb_to_recordset(${q([...rows.values()])}) r(product_id text,profile jsonb,embedding real[]);
insert into public.catalog_image_embeddings(image_sha256,model_revision,embedding)
 select distinct on(profile->>'imageSha256',profile->>'revision') profile->>'imageSha256',profile->>'revision',embedding from imported
 on conflict(image_sha256,model_revision) do nothing;
insert into public.catalog_visual_fashion(product_id,version,source_hash,image_url,image_sha256,model,model_revision,profile)
 select r.product_id,1,r.profile->>'sourceHash',r.profile->>'imageUrl',r.profile->>'imageSha256',r.profile->>'model',r.profile->>'revision',r.profile
 from imported r join public.shop_catalog s on s.id=r.product_id where s.is_active
 and r.profile->>'sourceHash'=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
 and r.profile->>'imageUrl'=s.image_url
 on conflict(product_id,version) do update set source_hash=excluded.source_hash,image_url=excluded.image_url,image_sha256=excluded.image_sha256,model=excluded.model,model_revision=excluded.model_revision,profile=excluded.profile,created_at=now();
-- Evidence capsule is advisory and explicitly machine assessed. It gives a
-- small retrieval preference, never a monopoly, price waiver or source edit.
insert into public.catalog_stylist_capsule(product_id,source_hash,image_sha256,archetype,evidence)
 select v.product_id,v.source_hash,v.image_sha256,i.subtype,
 jsonb_build_object('review','machine','taxonomy',v.profile->>'taxonomyRevision','sourceSlotVerified',true,'humanFashionApproval',false)
 from public.catalog_visual_fashion v join public.catalog_garment_intelligence i on i.product_id=v.product_id
 join public.shop_catalog s on s.id=v.product_id
 where s.is_active and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
 and v.profile#>>'{attributes,slot,value}'=i.slot
 and (v.profile#>>'{attributes,slot,confidence}')::numeric>=0.8
 and v.profile#>>'{attributes,subtype,value}'=i.subtype
 and (v.profile#>>'{attributes,subtype,confidence}')::numeric>=0.8
 and v.profile#>>'{imageKind,value}'='product'
 and (v.profile#>>'{imageQuality,width}')::integer>=256 and (v.profile#>>'{imageQuality,height}')::integer>=256
 on conflict(product_id) do update set source_hash=excluded.source_hash,image_sha256=excluded.image_sha256,archetype=excluded.archetype,evidence=excluded.evidence;
`;
const file=path.join(dir,'cache-import.sql');fs.writeFileSync(file,sql,{mode:0o600});
if(mode==='local')cp.execFileSync('/opt/homebrew/opt/postgresql@16/bin/psql',['-h','127.0.0.1','-p','55440','-d','postgres','-X','-q','-v','ON_ERROR_STOP=1','-1','-f',file],{stdio:['ignore','ignore','pipe']});
else if(mode!=='prepare')throw Error('Only prepare/local supported. Production requires guarded release.');
console.log({validatedProfiles:rows.size,mode,productionWrites:0});
