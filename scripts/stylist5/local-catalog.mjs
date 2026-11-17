/** Isolated PostgreSQL fixture from frozen private catalog; no production writes. */
import fs from 'node:fs';import path from 'node:path';import cp from 'node:child_process';
const dir=process.argv[2];if(!dir)throw Error('Provide private freeze directory');
const psql='/opt/homebrew/opt/postgresql@16/bin/psql',args=['-h','127.0.0.1','-p','55440','-d','postgres','-X','-q','-v','ON_ERROR_STOP=1'];
export const localQuery=sql=>cp.execFileSync(psql,[...args,'-A','-t','-c',sql],{encoding:'utf8',maxBuffer:64*1024*1024});
const q=s=>"'"+String(s).replaceAll("'","''")+"'";
const rows=JSON.parse(fs.readFileSync(path.join(dir,'catalog.json')));
let sql=`create role anon;create role authenticated;create role service_role bypassrls;
create schema auth;create table auth.users(id uuid primary key);
create function auth.uid() returns uuid language sql stable as $$select nullif(current_setting('request.jwt.claim.sub',true),'')::uuid$$;
grant usage on schema auth to public;grant execute on function auth.uid() to public;
create table public.shop_catalog(id text primary key,name text,brand text,description text,category text,garment_type text,image_url text,price numeric,currency text,source text,source_url text,external_id text,is_active boolean);
alter table public.shop_catalog enable row level security;
create policy active_read on public.shop_catalog for select using(is_active);
grant select on public.shop_catalog to anon,authenticated,service_role;
create table public.saved_outfits(id uuid primary key,user_id uuid not null,items jsonb);
alter table public.saved_outfits enable row level security;
create policy own on public.saved_outfits to authenticated using(user_id=auth.uid()) with check(user_id=auth.uid());
grant all on public.saved_outfits to authenticated,service_role;
`;
// Only the cache schema portion; legacy discovery definition comes from the baseline source migration.
const textMigration=fs.readFileSync('backend/supabase/migrations/20261107210000_fashion_catalog_intelligence.sql','utf8');
sql+=textMigration.slice(0,textMigration.indexOf('-- ACTIVATE RETRIEVAL'));
const imageMigration=fs.readFileSync('backend/supabase/migrations/20261107230000_visual_fashion_concepts.sql','utf8');
sql+=imageMigration.slice(0,imageMigration.indexOf('create function public.catalog_stylist_candidates'))+'commit;\n';
sql+=imageMigration.slice(imageMigration.indexOf('-- Anonymous to other users;'));
sql+=fs.readFileSync('backend/supabase/migrations/20261107233000_stylist_feedback_request_keys.sql','utf8');
sql+=`insert into public.shop_catalog select id,name,brand,description,category,garment_type,image_url,price,currency,source,source_url,external_id,is_active from jsonb_to_recordset(${q(JSON.stringify(rows))}::jsonb) r(id text,name text,brand text,description text,category text,garment_type text,image_url text,price numeric,currency text,source text,source_url text,external_id text,is_active boolean);
insert into public.catalog_garment_intelligence(product_id,version,source_hash,slot,subtype,formality,search_document,intelligence)
select r->>'id',1,r->>'source_hash',r->>'fashion_slot',r#>>'{fashion_intelligence,descriptor,subtype}',r#>>'{fashion_intelligence,descriptor,formality}',''::tsvector,r->'fashion_intelligence' from jsonb_array_elements(${q(JSON.stringify(rows))}::jsonb) r;
insert into public.catalog_visual_fashion(product_id,version,source_hash,image_url,image_sha256,model,model_revision,profile)
select r->>'id',1,r->>'source_hash',r->>'image_url',r#>>'{visual_fashion,imageSha256}',r#>>'{visual_fashion,model}',r#>>'{visual_fashion,revision}',r->'visual_fashion' from jsonb_array_elements(${q(JSON.stringify(rows.filter(r=>r.visual_fashion)))}::jsonb) r;
`;
const old=fs.readFileSync('backend/supabase/migrations/20261108003000_stylist_set_retrieval.sql','utf8');
sql+=old.replaceAll('public.catalog_stylist_candidates(', 'public.catalog_stylist_candidates_v200(');
sql+=fs.readFileSync('backend/supabase/migrations/20261108040000_stylist5_editorial.sql','utf8');
const file=path.join(dir,'..','fixture.sql');fs.writeFileSync(file,sql,{mode:0o600});
cp.execFileSync(psql,[...args,'-f',file],{stdio:['ignore','ignore','pipe']});console.log({localProducts:rows.length,productionWrites:0});
