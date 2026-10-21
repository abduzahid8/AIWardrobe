/** Three representative live requests, confined to a temporary trial account. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary-account setup credential required');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const report = { measuredAt: new Date().toISOString(), scope: 'Three representative production AI requests; not provider capacity or universal output accuracy', checks: [] };
const call = async (path, headers, body, method = 'POST') => {
  const start = performance.now();
  const response = await fetch(new URL(path, base), { method, headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(90000), redirect: 'error' });
  let data; try { data = await response.json(); } catch { data = null; }
  return { status: response.status, ok: response.ok, data, latencyMs: Math.round(performance.now() - start) };
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
  const trial = await call(`/rest/v1/profiles?id=eq.${owner}`, admin, { trial_started_at: new Date().toISOString() }, 'PATCH');
  if (!trial.ok) throw new Error(`Trial initialization failed: HTTP ${trial.status}`);
  if (process.env.PRODUCTION_OUTFIT_ONLY !== '1') {
  const chat = await call('/functions/v1/ai-chat', headers, { query: 'Give one short styling tip for pairing a navy shirt with beige trousers.' });
  report.checks.push({ name: 'chat_returns_text', status: chat.status, latencyMs: chat.latencyMs, passed: chat.ok && typeof chat.data?.text === 'string' && chat.data.text.trim().length > 10, provider: chat.data?.provider || null });
  }
  const generated = await call('/functions/v1/generate-outfits', headers, { occasion: 'Everyday', stylePreferences: 'Casual', prompt: 'Create one complete casual outfit for mild weather.', weather: { temp: 20, condition: 'Clear' }, limit: 1, ...(process.env.PRODUCTION_OUTFIT_PROVIDER === 'gemini' ? { provider: 'gemini' } : {}) });
  const outfits = generated.data?.outfits;
  report.checks.push({ name: 'generate_outfit_has_items', status: generated.status, latencyMs: generated.latencyMs, passed: generated.ok && Array.isArray(outfits) && outfits.length === 1 && outfits.every(o => Array.isArray(o.items) && o.items.length >= 3 && o.items.every(i => i.id && (i.image || i.imageUrl || i.image_url))), outfitCount: Array.isArray(outfits) ? outfits.length : null, itemCounts: Array.isArray(outfits) ? outfits.map(o => o.items?.length || 0) : [], source: generated.data?.source || null, provider: generated.data?.provider || null, items: Array.isArray(outfits) ? outfits.flatMap(o => (o.items || []).map(i => ({ category: i.category, type: i.type, name: i.name }))) : [] });
  report.checks.push({ name: 'primary_ai_outfit_generation', status: generated.status, passed: generated.ok && generated.data?.source === 'ai', observedSource: generated.data?.source || null });
  if (process.env.PRODUCTION_OUTFIT_PROVIDER === 'gemini') report.checks.push({ name: 'gemini_generation', status: generated.status, passed: generated.ok && generated.data?.source === 'ai' && String(generated.data?.provider || '').startsWith('gemini:'), provider: generated.data?.provider || null });
  report.providerAttempts = (generated.data?.providerAttempts || []).map(({ provider, model, status }) => ({ provider, model, status }));
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
  report.failure = /^[A-Za-z ]+failed: HTTP \d+$/.test(error.message) ? error.message : (error.name || 'Error');
  process.exitCode = 1;
} finally {
  if (owner) {
    const removed = await call(`/auth/v1/admin/users/${owner}`, admin, undefined, 'DELETE');
    report.cleanup = { temporaryUserDeleted: removed.ok, status: removed.status };
    if (!removed.ok) process.exitCode = 1;
  }
  const filename = process.env.PRODUCTION_AI_REPORT || (process.env.PRODUCTION_OUTFIT_PROVIDER === 'gemini' ? 'production-gemini-probe.json' : process.env.PRODUCTION_OUTFIT_ONLY === '1' ? 'production-outfit-diagnostic.json' : 'production-ai-smoke.json');
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
if (report.checks.some(c => !c.passed)) process.exitCode = 1;
