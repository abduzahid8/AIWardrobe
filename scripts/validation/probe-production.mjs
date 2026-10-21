/** Bounded read-only production probe. Does not call Auth/admin, paid AI, or write endpoints. */
import 'dotenv/config';
const base = new URL(process.env.EXPO_PUBLIC_SUPABASE_URL);
if (base.protocol !== 'https:' || !base.hostname.endsWith('.supabase.co')) throw new Error('Expected Supabase HTTPS host');
const anonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if (!anonKey) throw new Error('Missing public anon key');
const paths = [
  '/rest/v1/shop_catalog?select=id,name,garment_type,category,price,currency,image_url&is_active=eq.true&limit=1000',
  '/rest/v1/featured_capsules?select=id&limit=1',
  '/rest/v1/clothing_items?select=id&limit=1',
  '/rest/v1/profiles?select=id&limit=1',
];
const report = { measuredAt: new Date().toISOString(), scope: 'four bounded anonymous reads; no authenticated isolation or load certification', checks: [] };
for (const path of paths) {
  const start = performance.now();
  try {
    const response = await fetch(new URL(path, base), { headers: { apikey: anonKey, Authorization: `Bearer ${anonKey}` }, signal: AbortSignal.timeout(15000), redirect: 'error' });
    const rows = await response.json();
    const table = path.split('?')[0].split('/').pop();
    const check = { table, status: response.status, latencyMs: Math.round(performance.now() - start), rows: Array.isArray(rows) ? rows.length : null };
    if (Array.isArray(rows) && table === 'shop_catalog') {
      check.categories = Object.fromEntries([...new Set(rows.map(r => r.category || r.garment_type || 'unknown'))].map(category => [category, rows.filter(r => (r.category || r.garment_type || 'unknown') === category).length]));
      check.missingImage = rows.filter(r => !r.image_url).length;
      check.invalidPrice = rows.filter(r => !Number.isFinite(Number(r.price)) || Number(r.price) <= 0).length;
    }
    report.checks.push(check);
  } catch { report.checks.push({ table: path.split('?')[0].split('/').pop(), failure: 'network_or_invalid_json' }); }
}
console.log(JSON.stringify(report, null, 2));
if (report.checks.some(c => c.status !== 200 || c.rows === null || (['profiles', 'clothing_items'].includes(c.table) && c.rows !== 0))) process.exitCode = 1;
