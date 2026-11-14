import assert from 'node:assert/strict'
import {execFileSync, spawnSync} from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

const tools = ['initdb', 'pg_ctl', 'pg_isready', 'psql']
const available = tools.every(tool => spawnSync('sh', ['-lc', `command -v ${tool}`]).status === 0)

test('compact retrieval migration preserves database contracts', {skip: !available}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'stylist5-contract-'))
  const port = String(55460 + Math.floor(Math.random() * 100))
  const migration = path.resolve('backend/supabase/migrations/20261108140000_stylist5_compact_retrieval.sql')
  const psql = sql => execFileSync('psql', ['-h', root, '-p', port, '-d', 'postgres', '-v', 'ON_ERROR_STOP=1', '-At', '-c', sql], {encoding: 'utf8'})
    .trim().split('\n').pop()
  const run = (file, args) => execFileSync(file, args, {stdio: 'ignore'})
  try {
    run('initdb', ['-D', root, '-A', 'trust', '--no-locale'])
    run('pg_ctl', ['-D', root, '-o', `-p ${port} -k ${root}`, '-l', path.join(root, 'postgres.log'), 'start'])
    for (let i = 0; i < 30; i++) {
      if (spawnSync('pg_isready', ['-h', root, '-p', port]).status === 0) break
      execFileSync('sleep', ['1'])
    }
    execFileSync('psql', ['-h', root, '-p', port, '-d', 'postgres', '-v', 'ON_ERROR_STOP=1', '-c', `
      create role service_role;
      create role anon;
      create role authenticated;
      create table public.shop_catalog(id text primary key,name text,brand text,description text,category text,garment_type text,image_url text,price numeric,currency text,source text,source_url text,external_id text,is_active boolean);
      create table public.saved_outfits(id uuid primary key,user_id uuid,items jsonb);
      create table public.catalog_garment_intelligence(product_id text,version int,slot text,subtype text,formality text,intelligence jsonb);
      create table public.catalog_visual_fashion(product_id text,version int,image_url text,source_hash text,image_sha256 text,model_revision text,profile jsonb);
      create table public.catalog_image_embeddings(image_sha256 text,model_revision text,embedding jsonb);
      create table public.catalog_stylist_capsule(product_id text,source_hash text,image_sha256 text,archetype text,evidence jsonb,created_at timestamptz);
      grant all on all tables in schema public to service_role;
    `], {stdio: 'ignore'})
    execFileSync('psql', ['-h', root, '-p', port, '-d', 'postgres', '-v', 'ON_ERROR_STOP=1', '-f', migration], {stdio: 'ignore'})
    execFileSync('psql', ['-h', root, '-p', port, '-d', 'postgres', '-v', 'ON_ERROR_STOP=1', '-c', `
      set role service_role;
      insert into public.shop_catalog values
        ('cardigan-1','Merino cardigan','Test','product only plain background','outerwear','outerwear','https://x/card.jpg',120000,'UZS','product-only','https://x/card','card-1',true),
        ('blazer-1','Wool blazer','Test','product only plain background','outerwear','outerwear','https://x/blazer.jpg',80,'USD','product-only','https://x/blazer','blazer-1',true);
      insert into public.catalog_garment_intelligence values
        ('cardigan-1',1,'outerwear','cardigan','smart','{"version":1,"descriptor":{"slot":"outerwear","subtype":"cardigan","color":"navy","volume":"regular","evidence":["text:catalog"]}}'),
        ('blazer-1',1,'outerwear','blazer','smart','{"version":1,"descriptor":{"slot":"outerwear","subtype":"blazer","color":"navy","volume":"regular","evidence":["text:catalog"]}}');
      insert into public.catalog_visual_fashion
        (product_id,version,image_url,source_hash,image_sha256,model_revision,profile)
      values
        ('cardigan-1',1,'https://x/card.jpg',
          md5(jsonb_build_array('Merino cardigan','product only plain background','Test','outerwear','outerwear','https://x/card.jpg')::text),
          repeat('b',64),repeat('c',40),
          '{"version":1,"sourceHash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","imageUrl":"https://x/card.jpg","imageSha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","model":"siglip2","revision":"cccccccccccccccccccccccccccccccccccccccc","attributes":{"color":{"value":"navy","confidence":0.95,"margin":0.2,"evidence":"image-model"}}}');
    `], {stdio: 'ignore'})
    assert.equal(psql(`set role service_role;
      select count(*) from public.catalog_stylist_candidates_compact(
        array['outerwear'],'contract',
        '{"budget":15,"currency":"USD","rates":{"UZS":1,"USD":10000,"EUR":11000,"GBP":13000},"concepts":[{"roles":{"outerwear":["cardigan"]}}]}',
        100,0,null,null);`).trim(), '1')
    assert.equal(psql(`set role service_role;
      select count(*) from public.catalog_stylist_candidates_compact(
        array['outerwear'],'contract',
        '{"concepts":[{"roles":{"outerwear":["cardigan"]}}]}',
        100,0,null,null) r where r->>'subtype'='cardigan';`).trim(), '1')
    assert.equal(psql(`set role service_role;
      select (r->'price_peers' ? 'normalizedMedian')::text
      from public.catalog_stylist_candidates_compact(array['outerwear'],'contract','{}',100,0,null,null) r limit 1;`).trim(), 'true')
    assert.equal(psql(`set role service_role;
      select ((r->>'id'='cardigan-1') and (r->'price_peers' ? 'normalizedCount'))::text
      from public.catalog_stylist_candidate_hydration(array['cardigan-1'],'{"currency":"USD","rates":{"UZS":1,"USD":10000,"EUR":11000,"GBP":13000}}') r;`).trim(), 'true')
    assert.equal(psql(`set role service_role;
      select (jsonb_typeof(i.intelligence->'descriptor'->'evidence')='array'
        and jsonb_typeof(v.profile->'attributes'->'color')='object')::text
      from public.catalog_garment_intelligence i
      join public.catalog_visual_fashion v on v.product_id=i.product_id
      where i.product_id='cardigan-1';`).trim(), 'true')
  } finally {
    spawnSync('pg_ctl', ['-D', root, 'stop', '-m', 'immediate'], {stdio: 'ignore'})
    fs.rmSync(root, {recursive: true, force: true})
  }
})
