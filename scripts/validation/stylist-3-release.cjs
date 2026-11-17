/* Release only generate-outfits after reconciling live source. Credentials stay
 * in SUPABASE_ACCESS_TOKEN or SUPABASE_ACCESS_TOKEN_FILE, never in artifacts. */
const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const report = path.join(root, 'docs/reports/stylist-3-20261107');
const record = path.join(root, 'backend/supabase/rollback/generate-outfits-stylist-3-20261107');
const base = 'https://api.supabase.com/v1/projects/fyqpifmrsftsfqibhwhy/functions';
const baselineCommit = '7624a89e44c67600a14b2895b7b1855c083105e4';
const baselineSource = '820bbdf336d7d48cebc0e7415618fc437618a644cabc1e97d2014ee86bb15552';
const hash = b => crypto.createHash('sha256').update(b).digest('hex');
const git = (...args) => cp.execFileSync('git', args, {cwd: root, encoding: 'utf8'}).trim();
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n');
const summary = f => ({slug: f.slug, version: f.version, status: f.status, verify_jwt: f.verify_jwt, ezbr_sha256: f.ezbr_sha256});
const others = rows => rows.filter(f => f.slug !== 'generate-outfits').map(summary).sort((a, b) => a.slug.localeCompare(b.slug));
const token = process.env.SUPABASE_ACCESS_TOKEN || fs.readFileSync(process.env.SUPABASE_ACCESS_TOKEN_FILE, 'utf8').trim();
const headers = {Authorization: 'Bearer ' + token};
async function list() {
  const r = await fetch(base, {headers, signal: AbortSignal.timeout(30000)});
  assert(r.ok, 'list HTTP ' + r.status);
  return r.json();
}
async function source() {
  const r = await fetch(base + '/generate-outfits/body', {headers, signal: AbortSignal.timeout(30000)});
  assert(r.ok, 'body HTTP ' + r.status);
  const bytes = Buffer.from(await r.arrayBuffer()), key = Buffer.from('"sourcesContent":[');
  const position = bytes.indexOf(key);
  assert(position >= 0, 'embedded original source missing');
  const start = position + key.length;
  let end = start + 1;
  for (; end < bytes.length; end++) {
    if (bytes[end] === 92) {end++; continue;}
    if (bytes[end] === 34) break;
  }
  return JSON.parse(bytes.subarray(start, end + 1).toString());
}
function authBlock(s) {
  const a = s.indexOf("        const authHeader = req.headers.get('Authorization')");
  const b = s.indexOf('        // Provided clothing controls', a);
  assert(a > 0 && b > a, 'auth boundaries missing');
  return s.slice(a, b);
}
async function build() {
  return require('esbuild').build({absWorkingDir: root,
    entryPoints: ['backend/supabase/functions/generate-outfits/index.ts'], bundle: true,
    format: 'esm', platform: 'neutral', target: 'es2022', external: ['https://*'],
    write: false, metafile: true, logLevel: 'silent'});
}
async function smoke() {
  const url = 'https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/generate-outfits';
  const reports = [];
  for (const init of [{method: 'OPTIONS'}, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}, {method: 'POST', headers: {'Content-Type': 'application/json', Authorization: 'Bearer invalid-release-probe'}, body: '{}'}]) {
    const r = await fetch(url, {...init, signal: AbortSignal.timeout(30000)});
    reports.push({method: init.method, authorization: init.headers?.Authorization ? 'invalid' : 'absent', status: r.status, cors: r.headers.get('access-control-allow-origin')});
    assert.equal(r.status, init.method === 'OPTIONS' ? 200 : 401);
    await r.text();
  }
  return reports;
}
async function main() {
  const mode = process.argv[2];
  assert(['prepare', 'deploy', 'verify', 'rollback'].includes(mode), 'use prepare, deploy, verify or rollback');
  cp.execFileSync('git', ['merge-base', '--is-ancestor', baselineCommit, 'HEAD'], {cwd: root});
  if (mode === 'prepare') {
    const before = await list(), live = before.find(f => f.slug === 'generate-outfits');
    assert.equal(live.version, 195, 'production changed; reconcile again');
    assert.equal(live.status, 'ACTIVE');
    const original = await source();
    assert.equal(hash(original), baselineSource, 'v195 original source drift');
    const entry = 'backend/supabase/functions/generate-outfits/index.ts';
    assert.equal(authBlock(fs.readFileSync(path.join(root, entry), 'utf8')), authBlock(git('show', baselineCommit + ':' + entry)));
    const ts = require('typescript');
    const functions = source => {
      const tree = ts.createSourceFile('index.ts', source, ts.ScriptTarget.Latest, true);
      const result = new Map();
      function visit(node) {
        if ((ts.isFunctionDeclaration(node) || ts.isVariableDeclaration(node)) && node.name && ts.isIdentifier(node.name)) result.set(node.name.text,node.getText(tree));
        ts.forEachChild(node,visit);
      }
      visit(tree); return result;
    };
    const current = functions(fs.readFileSync(path.join(root,entry),'utf8'));
    const prior = functions(git('show',baselineCommit+':'+entry));
    for(const name of ['shopPriceOf','outfitShopSpend','validateOutfitAgainstStyle','validateOutfitCompatibility'])
      assert(current.has(name) && current.get(name) === prior.get(name), 'critical safeguard changed: '+name);
    const rateLimit = 'backend/supabase/functions/_shared/rateLimit.ts';
    assert.equal(fs.readFileSync(path.join(root, rateLimit), 'utf8').trim(), git('show', baselineCommit + ':' + rateLimit));
    fs.mkdirSync(record, {recursive: true});
    const staging = fs.mkdtempSync('/tmp/aiwardrobe-stylist-2-rollback-');
    fs.writeFileSync(path.join(staging, 'index.ts'), original);
    cp.execFileSync('tar', ['-czf', path.join(record, 'v195-source.tar.gz'), '-C', staging, 'index.ts']);
    fs.rmSync(staging, {recursive: true});
    const built = await build(), bundle = built.outputFiles[0].contents;
    const manifest = {function: 'generate-outfits', preparedFromCommit: git('rev-parse', 'HEAD'),
      branch: git('branch', '--show-current'), baseline: {...summary(live), sourceCommit: baselineCommit, sourceSha256: hash(original)},
      bundleSha256: hash(bundle), bundleBytes: bundle.length, authSourceUnchanged: true, rateLimitSourceUnchanged: true, criticalFunctionBodiesUnchanged: true,
      rollbackArchiveSha256: hash(fs.readFileSync(path.join(record, 'v195-source.tar.gz'))),
      inputs: Object.keys(built.metafile.inputs).map(p => ({path: p, sha256: hash(fs.readFileSync(path.join(root, p)))}))};
    write(path.join(record, 'manifest.json'), manifest);
    write(path.join(report, 'functions-before.json'), before.map(summary));
    console.log(JSON.stringify({prepared: true, baseline: live.version, sourceMatchesV195: true, bundleSha256: manifest.bundleSha256}));
    return;
  }
  const manifest = JSON.parse(fs.readFileSync(path.join(record, 'manifest.json')));
  if (mode === 'rollback') {
    const deployment = JSON.parse(fs.readFileSync(path.join(report, 'deployment.json')));
    const before = await list(), live = before.find(f => f.slug === 'generate-outfits');
    assert.equal(live.version, deployment.version, 'reconcile newer deployment before rollback');
    assert.equal(live.ezbr_sha256, deployment.ezbr_sha256);
    assert.equal(hash(await source()), manifest.bundleSha256);
    const archive = path.join(record, 'v195-source.tar.gz');
    assert.equal(hash(fs.readFileSync(archive)), manifest.rollbackArchiveSha256);
    const restored = cp.execFileSync('tar', ['-xOzf', archive, 'index.ts']);
    assert.equal(hash(restored), manifest.baseline.sourceSha256);
    const form = new FormData();
    form.append('metadata', JSON.stringify({name: 'generate-outfits', entrypoint_path: 'index.ts', verify_jwt: manifest.baseline.verify_jwt}));
    form.append('file', new Blob([restored], {type: 'application/typescript'}), 'index.ts');
    const r = await fetch(base + '/deploy?slug=generate-outfits', {method: 'POST', headers, body: form, signal: AbortSignal.timeout(120000)});
    assert(r.ok, 'rollback HTTP ' + r.status);
    const deployed = await r.json(), after = await list(), active = after.find(f => f.slug === 'generate-outfits');
    assert.equal(active.version, deployed.version);
    assert.equal(active.status, 'ACTIVE');
    assert.equal(active.verify_jwt, manifest.baseline.verify_jwt);
    assert.equal(hash(await source()), manifest.baseline.sourceSha256);
    assert.deepEqual(others(after), others(before));
    const verification = {...summary(active), restoresSourceVersion: 195, sourceSha256: hash(restored), liveHealth: await smoke()};
    write(path.join(report, 'rollback-verification.json'), verification);
    console.log(JSON.stringify(verification));
    return;
  }
  const built = await build(), bundle = built.outputFiles[0].contents;
  assert.equal(hash(bundle), manifest.bundleSha256);
  for (const input of manifest.inputs) {
    assert.equal(hash(fs.readFileSync(path.join(root, input.path))), input.sha256, 'working input changed: ' + input.path);
    assert.equal(hash(cp.execFileSync('git', ['show', 'HEAD:' + input.path], {cwd: root})), input.sha256, 'uncommitted input: ' + input.path);
  }
  assert.equal(hash(cp.execFileSync('tar', ['-xOzf', path.join(record, 'v195-source.tar.gz'), 'index.ts'])), manifest.baseline.sourceSha256);
  const before = await list(), live = before.find(f => f.slug === 'generate-outfits');
  if (mode === 'deploy') {
    const query = `select jsonb_build_object('migration',exists(select 1 from supabase_migrations.schema_migrations where version='20261107210000'),
      'rls',(select relrowsecurity from pg_class where oid='public.catalog_garment_intelligence'::regclass),
      'complete',not exists(select 1 from public.shop_catalog s left join public.catalog_garment_intelligence i on i.product_id=s.id
        where s.is_active and (i.product_id is null or i.version<>1 or i.source_hash<>md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text))),
      'invoker',(select not prosecdef from pg_proc where oid='public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb)'::regprocedure),
      'clientCannotWrite',not has_table_privilege('authenticated','public.catalog_garment_intelligence','INSERT')) as state`;
    const catalogResponse = await fetch('https://api.supabase.com/v1/projects/fyqpifmrsftsfqibhwhy/database/query', {method:'POST',headers:{...headers,'Content-Type':'application/json'},body:JSON.stringify({query}),signal:AbortSignal.timeout(30000)});
    assert(catalogResponse.ok,'catalog preflight HTTP '+catalogResponse.status);
    const state=(await catalogResponse.json())[0].state;
    assert(Object.values(state).every(value=>value===true),'catalog/security preflight failed');

    const branch = git('branch', '--show-current'), head = git('rev-parse', 'HEAD');
    assert.equal(git('ls-remote', '--heads', 'origin', branch).split(/\s/)[0], head, 'push reviewed source first');
    assert.equal(live.version, manifest.baseline.version, 'production version drift');
    assert.equal(live.ezbr_sha256, manifest.baseline.ezbr_sha256, 'production fingerprint drift');
    assert.equal(live.verify_jwt, manifest.baseline.verify_jwt, 'auth config drift');
    assert.equal(live.status, 'ACTIVE');
    assert.equal(hash(await source()), manifest.baseline.sourceSha256);
    const form = new FormData();
    form.append('metadata', JSON.stringify({name: 'generate-outfits', entrypoint_path: 'index.ts', verify_jwt: live.verify_jwt}));
    form.append('file', new Blob([bundle], {type: 'application/typescript'}), 'index.ts');
    const r = await fetch(base + '/deploy?slug=generate-outfits', {method: 'POST', headers, body: form, signal: AbortSignal.timeout(120000)});
    assert(r.ok, 'deploy HTTP ' + r.status);
    const deployed = await r.json();
    write(path.join(report, 'deployment.json'), {...summary(deployed), sourceCommit: head, branch});
  }
  const after = await list(), active = after.find(f => f.slug === 'generate-outfits');
  const deployment = JSON.parse(fs.readFileSync(path.join(report, 'deployment.json')));
  assert.equal(active.version, deployment.version);
  assert.equal(active.ezbr_sha256, deployment.ezbr_sha256);
  assert.equal(active.status, 'ACTIVE');
  assert.equal(active.verify_jwt, manifest.baseline.verify_jwt);
  assert.equal(hash(await source()), manifest.bundleSha256, 'deployed source differs');
  assert.deepEqual(others(after), others(JSON.parse(fs.readFileSync(path.join(report, 'functions-before.json')))));
  const verification = {checkedAt: new Date().toISOString(), ...summary(active), sourceCommit: deployment.sourceCommit,
    branch: deployment.branch, sourceSha256: manifest.bundleSha256, deployedSourceMatchesCommittedInputs: true,
    authenticationUnchanged: true, rateLimitUnchanged: true, otherFunctionsUnchanged: true,
    rollback: {version: 195, sourceSha256: manifest.baseline.sourceSha256, path: path.relative(root, record)},
    liveHealth: await smoke(), noAuthenticatedLiveGeneration: true, noUserDataOrSchemaChanges: true};
  write(path.join(report, 'verification.json'), verification);
  console.log(JSON.stringify(verification));
}
main().catch(e => {console.error(e.message); process.exitCode = 1;});
