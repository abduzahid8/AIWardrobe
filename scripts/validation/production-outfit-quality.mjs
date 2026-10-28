/** Three representative live requests, confined to a temporary trial account. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile, readFile } from 'node:fs/promises';
import ts from 'typescript';
import {catalogOutfitSpends} from './outfit-quality-budget.mjs';
import {loadBudgetHelpers} from './load-outfit-budget-helpers.mjs';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary-account setup credential required');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const reportOverride = process.env.OUTFIT_QUALITY_REPORT;
const budgetOverride = process.env.OUTFIT_QUALITY_BUDGET_USD === undefined ? null : Number(process.env.OUTFIT_QUALITY_BUDGET_USD);
if (budgetOverride !== null && (!Number.isFinite(budgetOverride) || budgetOverride <= 0 || budgetOverride > 10000)) throw new Error('Invalid scenario budget');
if (reportOverride && !/^[a-z0-9-]+\.json$/.test(reportOverride)) throw new Error('Invalid report filename');
const ratesOnly=process.env.OUTFIT_RATES_ONLY==='1';
const diagnosticRegion=process.env.OUTFIT_QUALITY_REGION;
if(diagnosticRegion&&!['eu-central-1','ap-south-1','ap-southeast-1'].includes(diagnosticRegion))throw Error('Invalid diagnostic region');
const report = { measuredAt: new Date().toISOString(), scope: 'Live NVIDIA category/budget scenarios; not exhaustive accuracy or capacity certification', checks: [] };
const call = async (path, headers, body, method = 'POST') => {
  const start = performance.now();
  const response = await fetch(new URL(path, base), { method, headers:diagnosticRegion&&path.startsWith('/functions/')?{...headers,'x-region':diagnosticRegion}:headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(120000), redirect: 'error' });
  let data; try { data = await response.json(); } catch { data = null; }
  return { status: response.status, ok: response.ok, data, region:response.headers.get('x-sb-edge-region'), latencyMs: Math.round(performance.now() - start) };
};
let owner, authHeaders;
try {
  const suffix = randomUUID();
  const email = `readiness-${suffix}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
  const created = await call('/auth/v1/admin/users', admin, { email, password, email_confirm: true, app_metadata: { readiness_audit: suffix }, user_metadata: { username: `readiness_${suffix.replaceAll('-','').slice(0,20)}`, readiness_test: true } });
  if (!created.ok || !created.data?.id) throw new Error(`Temporary account creation failed: HTTP ${created.status}`);
  owner = created.data.id;
  report.fixtureUserId = owner;
  const session = await call('/auth/v1/token?grant_type=password', { apikey: anon, 'Content-Type': 'application/json' }, { email, password });
  if (!session.ok || !session.data?.access_token) throw new Error('Temporary account login failed');
  const headers = { apikey: anon, Authorization: `Bearer ${session.data.access_token}`, 'Content-Type': 'application/json' };
  authHeaders = headers;
  const trial = await call(`/rest/v1/profiles?id=eq.${owner}`, admin, { trial_started_at: new Date().toISOString() }, 'PATCH');
  if (!trial.ok) throw new Error(`Trial initialization failed: HTTP ${trial.status}`);
  const {core,client} = await loadBudgetHelpers();
  const rates = await call('/functions/v1/budget-rates',headers,{});
  report.budgetRates = {status:rates.status,latencyMs:rates.latencyMs,success:rates.data?.success,error:rates.data?.error,snapshotPresent:!!rates.data?.snapshot,region:rates.region,requestedRegion:diagnosticRegion??null};
  const snapshot = core.validateBudgetFxSnapshot(rates.data?.snapshot);
  report.checks.push({name:'authenticated_budget_reference_available',status:rates.status,passed:rates.ok&&!!snapshot});
  if(ratesOnly)report.scope='Authenticated live reference-rate diagnostic only; no outfit or provider execution';
  const fx = core.budgetFxContext(snapshot);
  report.reference = snapshot;
  if (process.env.PRODUCTION_OUTFIT_ONLY !== '1') {
  const chat = await call('/functions/v1/ai-chat', headers, { query: 'Give one short styling tip for pairing a navy shirt with beige trousers.' });
  report.checks.push({ name: 'chat_returns_text', status: chat.status, latencyMs: chat.latencyMs, passed: chat.ok && typeof chat.data?.text === 'string' && chat.data.text.trim().length > 10, provider: chat.data?.provider || null });
  }
  const judge = {};
  const structure = {};
  for (const [name, target] of [['occasionRules', judge], ['outfitStructure', structure]]) {
    const source = await readFile(`backend/supabase/functions/_shared/${name}.ts`, 'utf8');
    new Function('exports', ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText)(target);
  }
  const scenarios = process.env.OUTFIT_QUALITY_REMAINING === '1' ? [
    { name: 'date_old_money_250_usd', stylePreferences: 'old_money', occasion: 'date', prompt: 'Create one complete outfit for date night. Budget under $250.', capUzs: 3150000 },
    { name: 'formal_semi_classic_250_usd', stylePreferences: 'semi_classic', occasion: 'formal', prompt: 'Create one complete outfit for a formal event. Budget under $250.', capUzs: 3150000 },
    { name: 'travel_minimalist_250_usd', stylePreferences: 'minimalist', occasion: 'travel', prompt: 'Create one complete comfortable outfit for travel. Budget under $250.', capUzs: 3150000 },
  ] : [
    { name: 'casual_250_usd', stylePreferences: 'casual', occasion: 'Everyday', prompt: 'Create one complete casual outfit. Budget under $250.', capUzs: 3150000 },
    { name: 'business_250_usd', stylePreferences: 'business_casual', occasion: 'Team Collaboration', prompt: 'Create one complete professional office outfit. Budget under $250.', capUzs: 3150000 },
    { name: 'workout_250_usd', stylePreferences: 'sport', occasion: 'Gym workout', prompt: 'Create one complete outfit suitable for a gym workout, including athletic shoes. Budget under $250.', capUzs: 3150000 },
  ];
  let suppliedPool;
  if(process.env.OUTFIT_PROVIDED_CATALOG==='1'){
    const fixtureIds=['md-uz-62246011','zara-uz-557829873','zara-uz-545409746'];
    const catalog=await call(`/rest/v1/shop_catalog?select=id,name,brand,price,currency,image_url,source_url,is_active&id=in.(${fixtureIds.join(',')})`,admin,undefined,'GET');
    if(!catalog.ok||catalog.data?.length!==3||catalog.data.some(row=>row.is_active!==true))throw new Error('Active supplied catalog fixture unavailable');
    suppliedPool=catalog.data.map(row=>({...row,id:`shop_${row.id}`,macroCategory:row.id===fixtureIds[0]?'top':row.id===fixtureIds[1]?'bottom':'shoes',isShopItem:true,isWardrobe:false,imageUrl:row.image_url,shopUrl:row.source_url}));
    report.scope='Live authenticated supplied-catalog path; not exhaustive accuracy or capacity certification';
  }
  const selectedScenarios=ratesOnly?[]:scenarios.filter(scenario => !process.env.OUTFIT_QUALITY_CASE || scenario.name === process.env.OUTFIT_QUALITY_CASE);
  if(!ratesOnly&&!selectedScenarios.length)throw Error('No matching outfit scenarios');
  for (const scenario of selectedScenarios) {
    if (budgetOverride !== null) {
      scenario.prompt = scenario.prompt.replace('$250', `$${budgetOverride}`);
      scenario.name = scenario.name.replace('250_usd', `${budgetOverride}_usd`);
    }
    const budget = client.parseBudgetFromText(scenario.prompt,fx);
    if(!budget)throw new Error('Scenario budget missing');
    scenario.capUzs = core.referenceAmount(budget.amount,budget.currency,'UZS',snapshot,fx.checkedAt);
    report.pendingScenario = scenario.name;
    const generated = await call('/functions/v1/generate-outfits', headers, { ...scenario, preferences:{maxShopBudget:{amount:budget.amount,currency:budget.currency},budgetRatesId:snapshot.id}, weather: { temp: 20, condition: 'Clear' }, limit: 1, ...(suppliedPool?{wardrobeOnly:true,useProvidedWardrobeOnly:true,wardrobeItems:suppliedPool}:{}) });
    const outfits = generated.data?.outfits;
    const items = Array.isArray(outfits) ? outfits.flatMap(o => o.items || []) : [];
    const ids = items.map(i => String(i.id || '').replace(/^shop_/, ''));
    const catalog = ids.length ? await call(`/rest/v1/shop_catalog?select=id,name,brand,price,currency,category,garment_type,source_url&id=in.(${ids.map(encodeURIComponent).join(',')})`, admin, undefined, 'GET') : null;
    const rows = new Map((Array.isArray(catalog?.data) ? catalog.data : []).map(row => [String(row.id), row]));
    const reasons = [];
    if (!generated.ok) reasons.push(`generation_http_${generated.status}`);
    if (!Array.isArray(outfits)) reasons.push('invalid_outfits');
    else if (outfits.length !== 1) reasons.push('wrong_outfit_count');
    for (const outfit of Array.isArray(outfits) ? outfits : []) {
      const structural = structure.outfitStructureReason(outfit.items, !!generated.data?.layered);
      const style = judge.styleJudgeReason(outfit.items, scenario.stylePreferences, scenario.occasion, {temp:20});
      if (structural) reasons.push(structural);
      if (style) reasons.push(style);
    }
    const outfitSpendsUzs = catalogOutfitSpends(outfits, rows);
    const spend = outfitSpendsUzs.length && outfitSpendsUzs.every(value => value !== null) ? Math.max(...outfitSpendsUzs) : null;
    for (const item of items) {
      const row = rows.get(String(item.id || '').replace(/^shop_/, ''));
      if (!row) { reasons.push('item_not_in_authoritative_catalog'); continue; }
      if (row.currency !== 'UZS' || !Number.isFinite(row.price) || row.price <= 0) reasons.push('invalid_catalog_price');
      if (Number(item.price) !== Number(row.price)) reasons.push('returned_price_differs_from_catalog');
      if (item.name !== row.name) reasons.push('returned_name_differs_from_catalog');
      if (item.currency !== row.currency) reasons.push('returned_currency_differs_from_catalog');
      if ((item.brand || '') !== (row.brand || '')) reasons.push('returned_brand_differs_from_catalog');
      if (item.shopUrl !== row.source_url) reasons.push('returned_url_differs_from_catalog');
    }
    if (outfitSpendsUzs.some(value => value === null)) reasons.push('unverified_outfit_spend');
    if (outfitSpendsUzs.some(value => value !== null && !core.fitsBudgetAmount(value,scenario.capUzs))) reasons.push('over_budget');
    if(generated.ok && generated.data?.budgetReference?.id!==snapshot.id)reasons.push('unacknowledged_reference_snapshot');
    report.checks.push({name:scenario.name,status:generated.status,error:generated.data?.error,code:generated.data?.code,minimumShopSpend:generated.data?.minimumShopSpend,latencyMs:generated.latencyMs,passed:reasons.length===0 && items.length>0,reasons,outfitCount:Array.isArray(outfits)?outfits.length:null,outfitSpendsUzs,spendUzs:spend,capUzs:scenario.capUzs,provider:generated.data?.provider,source:generated.data?.source,items:items.map(i=>({id:i.id,name:i.name,macroCategory:i.macroCategory,price:i.price,currency:i.currency,shopUrl:i.shopUrl})),providerAttempts:generated.data?.providerAttempts});
    delete report.pendingScenario;
  }
  if (process.env.PRODUCTION_OUTFIT_ONLY !== '1') {
  const catalog = await call('/rest/v1/shop_catalog?select=image_url,category,garment_type&is_active=eq.true&limit=1', { apikey: anon, Authorization: `Bearer ${anon}` }, undefined, 'GET');
  const sample = catalog.data?.[0];
  if (!sample?.image_url || !sample.image_url.startsWith('https://')) throw new Error('Public catalog image unavailable');
  const picture = await fetch(sample.image_url, { signal: AbortSignal.timeout(20000) });
  if (!picture.ok) throw new Error(`Public catalog image failed: HTTP ${picture.status}`);
  const bytes = Buffer.from(await picture.arrayBuffer());
  if (bytes.length > 7 * 1024 * 1024) throw new Error('Public sample exceeds test image budget');
  const image = `data:${picture.headers.get('content-type')?.split(';')[0] || 'image/jpeg'};base64,${bytes.toString('base64')}`;
  const scan = await call('/functions/v1/ai-process', headers, { image, operation: 'classify' });
  const expected = sample.category || sample.garment_type;
  const section = scan.data?.classification?.section;
  report.checks.push({ name: 'catalog_garment_classification', status: scan.status, latencyMs: scan.latencyMs, passed: scan.ok && !!scan.data?.classification && section === expected && !scan.data?._nvidiaError, expectedCatalogCategory: expected, classification: scan.data?.classification || null, providerReportedError: !!scan.data?._nvidiaError });
  }
} catch (error) {
  if (error.name === 'TypeError') report.transportDiagnostic = { fetchFailed: error.message === 'fetch failed', causeCode: typeof error.cause?.code === 'string' && /^[A-Z0-9_]{1,40}$/.test(error.cause.code) ? error.cause.code : null };
  report.failure = /^[A-Za-z ]+failed: HTTP \d+$/.test(error.message) ? error.message : (error.name || 'Error');
  process.exitCode = 1;
} finally {
  report.cleanup = {};
  if (authHeaders) {
    try { const revoked = await call('/auth/v1/logout?scope=global', authHeaders, {}); report.cleanup.refreshSessionsRevoked = revoked.ok; }
    catch { report.cleanup.refreshSessionsRevoked = false; }
    if (!report.cleanup.refreshSessionsRevoked) process.exitCode = 1;
  }
  if (owner) {
    try { const removed = await call(`/auth/v1/admin/users/${owner}`, admin, undefined, 'DELETE'); report.cleanup.temporaryUserDeleted = removed.ok; report.cleanup.status = removed.status; }
    catch { report.cleanup.temporaryUserDeleted = false; }
    if (!report.cleanup.temporaryUserDeleted) process.exitCode = 1;
  }
  const filename = reportOverride || (process.env.OUTFIT_QUALITY_CASE ? 'nvidia-date-budget-correction.json' : process.env.OUTFIT_QUALITY_REMAINING === '1' ? 'nvidia-category-budget-remaining.json' : 'nvidia-category-budget-matrix.json');
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
if (report.checks.some(c => !c.passed)) process.exitCode = 1;
