/// <reference types="node" />
/* Offline metadata enrichment and guarded schema activation. No image/provider
 * calls; original products are read only. Credentials never enter artifacts. */
import fs from 'node:fs';
import cp from 'node:child_process';
import crypto from 'node:crypto';
import {
  fashionIntelligence,
  FASHION_INTELLIGENCE_VERSION,
} from '../../src/domain/garmentDescriptor.ts';
import { detectSlotMismatch } from '../../src/domain/occasionRules.ts';
const token =
  process.env.SUPABASE_ACCESS_TOKEN ||
  fs.readFileSync(process.env.SUPABASE_ACCESS_TOKEN_FILE!, 'utf8').trim();
const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
const base = 'https://api.supabase.com/v1/projects/fyqpifmrsftsfqibhwhy';
const path = 'backend/supabase/migrations/20261107210000_fashion_catalog_intelligence.sql';
const [schema, activation] = fs.readFileSync(path, 'utf8').split('-- ACTIVATE RETRIEVAL:');
const quote = (value: unknown) => "'" + JSON.stringify(value).replaceAll("'", "''") + "'::jsonb";
async function query(sql: string) {
  const response = await fetch(base + '/database/query', {
    method: 'POST',
    headers,
    body: JSON.stringify({ query: sql }),
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok)
    throw Error(`database HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
  return response.json();
}
async function baseline(allowCurrentRelease = false) {
  const response = await fetch(base + '/functions', { headers });
  if (!response.ok) throw Error('function baseline unavailable');
  const rows = await response.json(),
    live = rows.find((row: any) => row.slug === 'generate-outfits');
  const deployedPath = 'docs/reports/stylist-3-20261107/deployment.json';
  const deployed =
    allowCurrentRelease && fs.existsSync(deployedPath)
      ? JSON.parse(fs.readFileSync(deployedPath, 'utf8'))
      : null;
  const matchesRelease =
    deployed && live?.version === deployed.version && live?.ezbr_sha256 === deployed.ezbr_sha256;
  if (
    !matchesRelease &&
    (live?.version !== 195 ||
      live?.ezbr_sha256 !== '81994ebfb9ae3eb1f3276ea41f0c5df6a2c70dcaa0cf49cd1fcd24c4006dae19')
  )
    throw Error('production baseline drift');
}
async function main() {
  const mode = process.argv[2];
  if (!['stage', 'enrich', 'activate', 'verify'].includes(mode))
    throw Error('use stage/enrich/activate/verify');
  if (mode !== 'verify') {
    cp.execFileSync('git', [
      'diff',
      '--quiet',
      'HEAD',
      '--',
      path,
      'scripts/validation/stylist-3-catalog.ts',
      'src/domain',
    ]);
    await baseline(mode === 'enrich');
    const rollback = 'backend/supabase/rollback/generate-outfits-stylist-3-20261107';
    const manifest = JSON.parse(fs.readFileSync(rollback + '/manifest.json', 'utf8'));
    if (
      crypto
        .createHash('sha256')
        .update(fs.readFileSync(rollback + '/v195-source.tar.gz'))
        .digest('hex') !== manifest.rollbackArchiveSha256
    )
      throw Error('rollback archive drift');
  }
  if (mode === 'stage') {
    const state = await query(
      "select to_regclass('public.catalog_garment_intelligence') as table_name;"
    );
    if (state[0].table_name) throw Error('metadata table already exists; reconcile before staging');
    await query(schema);
    console.log(JSON.stringify({ staged: true, originalCatalogUnchanged: true }));
    return;
  }
  if (mode === 'enrich') {
    let after = '',
      total = 0;
    for (;;) {
      const rows =
        await query(`select s.id,s.name,s.brand,s.description,s.category,s.garment_type,s.image_url,
        md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) as source_hash
        from public.shop_catalog s where s.is_active and s.id>'${after.replaceAll("'", "''")}' order by s.id limit 500`);
      if (!rows.length) break;
      const records = rows.map((row: any) => {
        const initial =
          row.garment_type === 'shoes' || row.category === 'shoes'
            ? 'shoes'
            : row.garment_type === 'lower_body' || row.category === 'bottoms'
              ? 'bottom'
              : row.garment_type === 'outerwear' || row.category === 'outerwear'
                ? 'outerwear'
                : 'top';
        let slot: string =
          detectSlotMismatch(initial, row.name || '', row.garment_type || '') || initial;
        if (row.garment_type === 'accessory' || row.garment_type === 'outfit')
          slot = row.garment_type === 'outfit' ? 'outfits' : 'accessory';
        if (
          !['top', 'outerwear', 'bottom', 'shoes', 'dresses', 'accessory', 'outfits'].includes(slot)
        )
          slot = 'other';
        const intelligence = fashionIntelligence({
          ...row,
          macroCategory: slot,
          type: row.garment_type,
          isShopItem: true,
        });
        return {
          product_id: row.id,
          source_hash: row.source_hash,
          slot,
          subtype: intelligence.descriptor.subtype,
          formality: intelligence.descriptor.formality,
          intelligence,
        };
      });
      await query(`insert into public.catalog_garment_intelligence(product_id,version,source_hash,slot,subtype,formality,search_document,intelligence)
        select r.product_id,${FASHION_INTELLIGENCE_VERSION},r.source_hash,r.slot,r.subtype,r.formality,
          to_tsvector('simple',concat_ws(' ',s.name,s.brand,s.description)),r.intelligence
        from jsonb_to_recordset(${quote(records)}) r(product_id text,source_hash text,slot text,subtype text,formality text,intelligence jsonb)
        join public.shop_catalog s on s.id=r.product_id
        where r.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
        on conflict(product_id) do update set version=excluded.version,source_hash=excluded.source_hash,slot=excluded.slot,
          subtype=excluded.subtype,formality=excluded.formality,search_document=excluded.search_document,intelligence=excluded.intelligence,updated_at=now()`);
      total += rows.length;
      after = rows.at(-1).id;
    }
    await query('analyze public.catalog_garment_intelligence');
    console.log(
      JSON.stringify({
        enriched: total,
        version: FASHION_INTELLIGENCE_VERSION,
        noProviderCalls: true,
      })
    );
    return;
  }
  if (mode === 'activate') {
    const verify = `do $$ begin
      if exists(select 1 from public.shop_catalog s left join public.catalog_garment_intelligence i on i.product_id=s.id
        where s.is_active and (i.product_id is null or i.version<>1 or i.source_hash<>md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text))) then
        raise exception 'catalog intelligence coverage incomplete';end if;end $$;`;
    const sql = '-- ACTIVATE RETRIEVAL:' + activation;
    await query(
      sql
        .replace('begin;', `begin;\n${verify}`)
        .replace(
          "notify pgrst,'reload schema';",
          "insert into supabase_migrations.schema_migrations(version,name,statements) values('20261107210000','fashion_catalog_intelligence',array['Managed guarded additive catalog intelligence rollout']);\nnotify pgrst,'reload schema';"
        )
    );
    console.log(
      JSON.stringify({ activated: true, contract: 'catalog_discovery_candidates', version: 1 })
    );
  }
  const result =
    await query(`select jsonb_build_object('active',(select count(*) from public.shop_catalog where is_active),
    'enriched',(select count(*) from public.catalog_garment_intelligence i join public.shop_catalog s on s.id=i.product_id where s.is_active and i.version=1),
    'rls',(select relrowsecurity from pg_class where oid='public.catalog_garment_intelligence'::regclass),
    'functionSecurity',(select jsonb_build_object('invoker',not prosecdef,'searchPath',proconfig) from pg_proc where oid='public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)'::regprocedure),
    'clientCanWrite',has_table_privilege('authenticated','public.catalog_garment_intelligence','INSERT'),
    'rpcRoles',jsonb_build_object('anon',has_function_privilege('anon','public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)','EXECUTE'),
      'authenticated',has_function_privilege('authenticated','public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)','EXECUTE')),
    'slots',(select jsonb_object_agg(slot,n) from(select slot,count(*) n from public.catalog_garment_intelligence group by slot)t)) as verification`);
  console.log(JSON.stringify(result));
}
main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
