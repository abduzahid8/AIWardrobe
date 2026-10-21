/** Live budget preflight confined to a disposable account; no model calls expected. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary-account setup credential required');
const reportName=process.env.PRODUCTION_BUDGET_REPORT || 'production-budget-preflight.json';
if(!/^[a-z0-9-]+\.json$/.test(reportName))throw new Error('Invalid report filename');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const report = { measuredAt: new Date().toISOString(), scope: 'Production budget preflight: input and empty-pool decisions, not generation or capacity certification', checks: [], rateLimitWaits: [] };
const call = async (path, headers, body, method = 'POST') => {
  const start = performance.now();
  const response = await fetch(new URL(path, base), { method, headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(90000), redirect: 'error' });
  let data; try { data = await response.json(); } catch { data = null; }
  return { status: response.status, ok: response.ok, data, retryAfterSeconds: Number(response.headers.get('retry-after') || data?.retry_after_sec || 0), latencyMs: Math.round(performance.now() - start) };
};
let owner;
try {
  const suffix = randomUUID();
  const email = `readiness-${suffix}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
  const created = await call('/auth/v1/admin/users', admin, { email, password, email_confirm: true, app_metadata: { readiness_audit: suffix }, user_metadata: { username: `readiness_${suffix}`, readiness_test: true } });
  if (!created.ok || !created.data?.id) throw new Error(`Temporary account creation failed: HTTP ${created.status}`);
  owner = created.data.id;
  const session = await call('/auth/v1/token?grant_type=password', { apikey: anon, 'Content-Type': 'application/json' }, { email, password });
  if (!session.ok || !session.data?.access_token) throw new Error('Temporary account login failed');
  const headers = { apikey: anon, Authorization: `Bearer ${session.data.access_token}`, 'Content-Type': 'application/json' };
  const trial = await call(`/rest/v1/profiles?id=eq.${owner}`, headers, { trial_started_at: new Date().toISOString() }, 'PATCH');
  if (!trial.ok) throw new Error(`Trial initialization failed: HTTP ${trial.status}`);
  const rates = await call('/functions/v1/budget-rates', headers, {});
  const snapshot = rates.data?.snapshot;
  if (!rates.ok || !snapshot?.id || snapshot.source !== 'CBU') throw new Error('Reference snapshot unavailable');
  report.reference = snapshot;
  const usdUZS = snapshot.uzsPerUnit.USD;
  const eurLimit = Math.floor(95 * usdUZS / snapshot.uzsPerUnit.EUR * 100) / 100;
  const gbpLimit = Math.ceil(120 * usdUZS / snapshot.uzsPerUnit.GBP * 100) / 100;
  const catalog = await call('/rest/v1/shop_catalog?select=id,name,category,garment_type,price,currency,image_url&is_active=eq.true&price=gt.12600&limit=1', admin, undefined, 'GET');
  const row = catalog.data?.[0];
  if (!catalog.ok || !row) throw new Error('Catalog fixture unavailable');
  const shopping = [{...row,id:`shop_${row.id}`,macroCategory:'top',imageUrl:row.image_url,isShopItem:true,isWardrobe:false}];
  const completePool=['top','bottom','shoes'].map(macroCategory=>({id:`shop_budget_probe_${macroCategory}`,name:`Test ${macroCategory}`,macroCategory,isShopItem:true,isWardrobe:false,price:100,currency:'USD',imageUrl:'https://example.invalid/test.jpg'}));
  const mixedPool = completePool.map((item, index) => ({ ...item, price: index === 0 ? 100000 : 20, currency: index === 0 ? 'UZS' : 'USD' }));
  const cases = [
    {name:'eur_reference_over_usd_cap',selected:{amount:100,currency:'USD'},prompt:'',items:completePool.map(item=>({...item,price:100,currency:'EUR'})),status:422,code:'no_items_within_budget',budget:{amount:100,currency:'USD'}},
    {name:'required_eur_piece_with_owned_remaining_slots',selected:{amount:100,currency:'USD'},prompt:'',items:completePool.map(item=>item.macroCategory==='top'?{...item,price:100,currency:'EUR'}:{...item,id:`owned_${item.macroCategory}`,isShopItem:false,isWardrobe:true}),status:422,code:'no_outfit_within_budget',budget:{amount:100,currency:'USD'},minimumShopSpend:100*snapshot.uzsPerUnit.EUR/usdUZS},
    {name:'grouped_uzs_budget_text',selected:null,prompt:'Under 1 000 000 UZS',items:completePool.map(item=>({...item,price:2000000,currency:'UZS'})),status:422,code:'no_items_within_budget',budget:{amount:1000000,currency:'UZS'}},
    ...[undefined, null, 0, -1].map((price, index) => ({name:`unpriced_shop_pool_${index}`,selected:{amount:250,currency:'USD'},prompt:'',items:completePool.map(item=>({...item,price})),status:422,code:'unverified_shop_pricing',budget:{amount:250,currency:'USD'}})),
    {name:'unpriced_required_slot',selected:{amount:250,currency:'USD'},prompt:'',items:completePool.map(item=>item.macroCategory==='top'?{...item,price:null}:item),status:422,code:'unverified_shop_pricing',budget:{amount:250,currency:'USD'}},
    {name:'unsupported_shop_currency',selected:{amount:250,currency:'USD'},prompt:'',items:completePool.map(item=>({...item,currency:'XYZ'})),status:422,code:'unverified_shop_pricing',budget:{amount:250,currency:'USD'}},
    {name:'mixed_currency_minimum_exceeds_cap',selected:{amount:400000,currency:'UZS'},prompt:'',items:mixedPool,status:422,code:'no_outfit_within_budget',budget:{amount:400000,currency:'UZS'},minimumShopSpend:100000+40*usdUZS},
    {name:'live_gym_catalog_cannot_meet_cap',selected:{amount:250,currency:'USD'},prompt:'Create a gym workout outfit',occasion:'Gym workout',style:'sport',providedOnly:false,wardrobeOnly:false,items:[],status:422,code:'no_outfit_within_budget',budget:{amount:250,currency:'USD'}},
    {name:'complete_outfit_minimum_exceeds_cap',selected:{amount:250,currency:'USD'},prompt:'',items:completePool,status:422,code:'no_outfit_within_budget',budget:{amount:250,currency:'USD'},minimumShopSpend:300},
    {name:'mixed_currency_usd_stricter',selected:null,prompt:`Maximum $100 or £${gbpLimit}`,items:[],status:422,code:'no_matching_clothing',budget:{amount:100,currency:'USD'}},
    {name:'mixed_currency_eur_stricter',selected:null,prompt:`Maximum $100 or €${eurLimit}`,items:[],status:422,code:'no_matching_clothing',budget:{amount:eurLimit,currency:'EUR'}},
    {name:'selected_cap_cannot_be_raised',selected:{amount:250,currency:'USD'},prompt:'Budget under $1000',items:[],status:422,code:'no_matching_clothing',budget:{amount:250,currency:'USD'}},
    {name:'prompt_can_lower_cap',selected:{amount:250,currency:'USD'},prompt:'Budget under $100',items:[],status:422,code:'no_matching_clothing',budget:{amount:100,currency:'USD'}},
    {name:'invalid_budget_rejected',selected:{amount:-1,currency:'USD'},prompt:'',items:[],status:400,error:'invalid_outfit_budget'},
    {name:'impossible_budget_distinguished',selected:{amount:1,currency:'USD'},prompt:'',items:shopping,status:422,code:'no_items_within_budget',budget:{amount:1,currency:'USD'}},
    {name:'zero_text_budget_preserved',selected:null,prompt:'Budget under $0',items:shopping,status:422,code:'no_items_within_budget',budget:{amount:0,currency:'USD'}},
  ];
  for (const test of cases.filter(test=>!process.env.PRODUCTION_BUDGET_CASE || test.name===process.env.PRODUCTION_BUDGET_CASE)) {
    const attempt=()=>call('/functions/v1/generate-outfits',headers,{weather:{temp:20,condition:'Clear'},prompt:test.prompt,occasion:test.occasion||'Everyday',stylePreferences:test.style||'casual',limit:1,wardrobeOnly:test.wardrobeOnly??true,useProvidedWardrobeOnly:test.providedOnly??true,wardrobeItems:test.items,preferences:{maxShopBudget:test.selected,budgetRatesId:snapshot.id}});
    let result = await attempt();
    if (result.status === 429) {
      const seconds = result.retryAfterSeconds;
      if (!Number.isInteger(seconds) || seconds < 1 || seconds > 60) throw new Error('Invalid rate-limit retry window');
      report.rateLimitWaits.push({case:test.name,status:429,waitSeconds:seconds});
      // Pace this audit using the existing guard; never disable production limits.
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
      result = await attempt();
    }
    const budgetMatches=!test.budget || result.data?.budget?.amount===test.budget.amount && result.data?.budget?.currency===test.budget.currency;
    report.checks.push({name:test.name,status:result.status,latencyMs:result.latencyMs,code:result.data?.code,error:result.data?.error,budget:result.data?.budget,minimumShopSpend:result.data?.minimumShopSpend,passed:result.status===test.status && (!test.code||result.data?.code===test.code) && (!test.error||result.data?.error===test.error) && budgetMatches && (!test.minimumShopSpend || Math.abs(result.data?.minimumShopSpend-test.minimumShopSpend)<1e-6) && result.data?.success===false && Array.isArray(result.data?.outfits) && result.data.outfits.length===0});
  }
} catch (error) {
  report.failure = /^[A-Za-z ]+failed: HTTP \d+$/.test(error.message) ? error.message : (error.name || 'Error');
  process.exitCode = 1;
} finally {
  if (owner) {
    const removed = await call(`/auth/v1/admin/users/${owner}`, admin, undefined, 'DELETE');
    report.cleanup = { temporaryUserDeleted: removed.ok, status: removed.status };
    if (!removed.ok) process.exitCode = 1;
  }
  const filename = reportName;
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
if (!report.checks.length || report.checks.some(c => !c.passed)) process.exitCode = 1;
