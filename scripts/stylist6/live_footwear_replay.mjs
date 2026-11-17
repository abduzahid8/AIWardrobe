/**
 * Real live-RLS catalog (frozen read-only) versus hypothetical inactive
 * footwear candidates. NO cloud inference, DB writes or photo requests.
 *
 * 1) AIW_S6_ENV_FILE=<existing .env path> node scripts/stylist6/frozen_catalog.mjs
 *      .private/stylist6-live/catalog.json --per-slot 96
 * 2) node scripts/stylist6/live_footwear_replay.mjs
 *      --catalog .private/stylist6-live/catalog.json
 *      --out .private/stylist6-live/comparison
 *
 * Add --exclude-brands ECCO to model a user who doesn't want ECCO.
 *
 * The only synthetic assumption is that FOUR already-curated Tommy Hilfiger
 * review rows were hypothetically activated IN PRIVATE MEMORY. Those items
 * remain inactive in real Supabase and must not be purchased/advertised based
 * on this synthetic replay.
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const sha = data => createHash('sha256').update(data).digest('hex');
const scenarioIds = Object.freeze([
  'S6-01', 'S6-02', 'S6-04', 'S6-05', 'S6-07', 'S6-08', 'S6-B01', 'S6-B03',
]);

const readJson = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const makePrivateDirectory = pathname => {
  const dir = path.resolve(pathname);
  // Never accidentally emit protected merchant catalog photos/links in
  // tracked docs, source folders or an arbitrary user-home directory.
  const repoPrivate = path.join(root, '.private') + path.sep;
  const privateTemp = path.resolve(os.tmpdir()) + path.sep;
  if (!dir.startsWith(repoPrivate) && !dir.startsWith(privateTemp) &&
      !dir.startsWith('/private/tmp/')) {
    throw new Error('output_must_be_under_gitignored_private_or_system_temp');
  }
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);
  return dir;
};

export function verifyLivePreviewRows(catalog, preview, { excludedBrands = [] } = {}) {
  if (!Array.isArray(catalog) || catalog.length < 30 || catalog.length > 5000 ||
      new Set(catalog.map(x => x?.id)).size !== catalog.length ||
      catalog.some(x => !x?.id || x.is_active !== true || x.currency !== 'UZS')) {
    throw new Error('invalid_frozen_live_catalog');
  }
  const ignored = new Set(excludedBrands.map(x=>x.toUpperCase()));
  const base = catalog.filter(x=> !ignored.has(String(x.brand||'').toUpperCase()));
  if (!Array.isArray(preview?.rows) || preview.rows.length !== 4 ||
    new Set(preview.rows.map(x=>x.id)).size !== 4) throw new Error('invalid_four_row_preview');
  const existing = new Set(catalog.map(x=>x.id));
  for (const row of preview.rows) {
    if (existing.has(row.id) || row.brand !== 'Tommy Hilfiger' ||
      row.is_active !== false || row.currency !== 'UZS' ||
      row.category !== 'shoes' || row.garment_type !== 'shoes' ||
      !Number.isSafeInteger(row.price) || row.price <= 0 ||
      !row.external_id || row.source !== 'frgroup-uz-review-not-active' ||
      !String(row.source_url).startsWith('https://frgroup.uz/ru/store/') ||
      !String(row.image_url).startsWith('https://media.frgroup.uz/')) {
      throw new Error('untrusted_candidate_in_preview');
    }
  }
  return { base, shadow: [
    ...base, ...preview.rows.map(row=>({...row,is_active:true})),
  ] };
}

export function comparePilotResults(baseline, shadow, previewIds) {
  if (!Array.isArray(baseline?.runs) || baseline.runs.length !== scenarioIds.length ||
      !Array.isArray(shadow?.runs) || shadow.runs.length !== scenarioIds.length) {
    throw new Error('pilot_runs_missing');
  }
  if (!(previewIds instanceof Set) || previewIds.size !== 4) throw new Error('preview_ids_invalid');
  const count = data => {
    const shoes = [], completeLooks = [];
    const perScenario = {};
    for (const run of data.runs) {
      if (!scenarioIds.includes(run.id) || Object.hasOwn(perScenario,run.id))
        throw new Error('unexpected_pilot_scenario');
      const arms = {};
      for (const arm of ['reference-first','creative-first']) {
        const looks=run.arms?.[arm]?.looks;
        if (!Array.isArray(looks) || looks.length > 3) throw new Error('invalid_pilot_looks');
        arms[arm] = looks.length;
        for (const look of looks) {
          const items=look?.items;
          if (!Array.isArray(items) || items.length < 3 || items.length > 4) throw new Error('invalid_pilot_items');
          const footwear=items.filter(x=>x.macroCategory==='shoes');
          if (footwear.length !== 1) throw new Error('pilot_shoe_slot_mismatch');
          shoes.push(footwear[0].id);
          completeLooks.push({
            key: run.id + '/' + arm + '/' + completeLooks.filter(x=>x.key.startsWith(run.id+'/'+arm+'/')).length,
            ids:items.map(x=>x.id),
          });
        }
      }
      perScenario[run.id]=arms;
    }
    return { shoes, completeLooks, perScenario,
      totalLooks: completeLooks.length,
      uniqueShoes: new Set(shoes).size,
      previewShoeSelections: shoes.filter(x=>previewIds.has(x)).length,
    };
  };
  const before=count(baseline), after=count(shadow);
  if (before.completeLooks.length !== after.completeLooks.length) {
    // Different cardinality is valid; match keyed positions and count extras
  }
  const a=new Map(before.completeLooks.map(x=>[x.key,x.ids]));
  const b=new Map(after.completeLooks.map(x=>[x.key,x.ids]));
  let changedLooks=0;
  for (const key of new Set([...a.keys(),...b.keys()])) {
    if (JSON.stringify(a.get(key))!==JSON.stringify(b.get(key))) changedLooks++;
  }
  return {
    before:{totalLooks:before.totalLooks,uniqueShoes:before.uniqueShoes,perScenario:before.perScenario},
    after:{totalLooks:after.totalLooks,uniqueShoes:after.uniqueShoes,perScenario:after.perScenario},
    previewShoeSelections: after.previewShoeSelections,
    changedLooks,
    nvidiaCalls:0, photoRequests:0,
    productionChanges:0,
    evaluation:'retrieval_only_not_human_fashion_or_checkout_proof',
  };
}

function argValue(key, defaults=null) {
  const i=process.argv.indexOf('--'+key);
  return i < 0 ? defaults : process.argv[i+1] || defaults;
}

async function main() {
  const input=argValue('catalog');
  const output=argValue('out',path.join(root,'.private','stylist6-live','comparison'));
  if (!input) throw new Error('usage: --catalog PRIVATE_JSON [--out PRIVATE_FOLDER] [--exclude-brands ECCO]');
  const excludedBrands=(argValue('exclude-brands','')||'').split(',').map(s=>s.trim()).filter(Boolean);
  const allowedExclusions=new Set(['ECCO','CABANI','BASCONI']);
  if (excludedBrands.some(x=>!allowedExclusions.has(x.toUpperCase())))
    throw new Error('unapproved_excluded_brand');
  const privateDir=makePrivateDirectory(output);
  const catalog=readJson(path.resolve(input));
  const preview=readJson(path.join(root,'benchmarks/stylist6/FOOTWEAR_UZ_INACTIVE_PREVIEW_20261113.json'));
  const {base,shadow}=verifyLivePreviewRows(catalog,preview,{excludedBrands});
  const writePrivate=(filename,object)=>{
    const bytes=JSON.stringify(object);
    fs.writeFileSync(filename,bytes+'\n',{mode:0o600});
    return sha(bytes+'\n');
  };
  const paths={
    base: path.join(privateDir,'base'),
    shadow:path.join(privateDir,'shadow'),
  };
  for (const p of Object.values(paths)) fs.mkdirSync(p,{recursive:true,mode:0o700});
  const baseHash=writePrivate(path.join(paths.base,'catalog.json'),base);
  const shadowHash=writePrivate(path.join(paths.shadow,'catalog.json'),shadow);

  for(const label of ['base','shadow']){
    const experimentDir=path.join(privateDir,'runs-'+label);
    fs.mkdirSync(experimentDir,{mode:0o700});
    const args=[
      path.join(root,'scripts/stylist6/pilot_12.mjs'),
      '--snapshot-dir',paths[label],
      '--out',experimentDir,
      '--report-out',path.join(experimentDir,'report.json'),
      '--scenarios',scenarioIds.join(','),
      '--skip-v5','true',
    ];
    // Child script uses only local JSON/catalog metadata; it never downloads
    // images or runs a hosted critic. Limit runtime and log safely.
    const outputText=execFileSync(process.execPath,args,{cwd:root,encoding:'utf8',timeout:180000});
    if(!outputText.includes('"scenarios":8')) throw new Error('incomplete_'+label+'_pilot');
  }

  const before=readJson(path.join(privateDir,'runs-base','pilot-private.json'));
  const after=readJson(path.join(privateDir,'runs-shadow','pilot-private.json'));
  const result=comparePilotResults(before,after,new Set(preview.rows.map(x=>x.id)));
  const summary={
    version:1,
    ...result,
    inputCatalogRows:catalog.length,
    originalEligibleRows:base.length,
    hypotheticallyAdded:preview.rows.length,
    shadowEligibleRows:shadow.length,
    excludedBrands,
    sourceCatalogSha256:sha(fs.readFileSync(path.resolve(input))),
    replayBaseSha256:baseHash,
    replayShadowSha256:shadowHash,
    scenarioIds,
    note:'Comparative offline replay on fresh live Supabase public read-only catalog sampling. No stock guarantees, photo permissions or production changes.',
  };
  writePrivate(path.join(privateDir,'summary.json'),summary);
  console.log(JSON.stringify(summary,null,2));
}

const direct=process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url);
if(direct) main().catch(e=>{console.error('LIVE_FOOTWEAR_REPLAY_REFUSED '+e.message);process.exitCode=1});
