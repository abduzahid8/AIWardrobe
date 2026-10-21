/** Public anon JWT checks: gateway-valid credential without a user; no valid image or provider work requested. */
import 'dotenv/config';
import { writeFile } from 'node:fs/promises';
const checks = [];
for (const name of ['generate-outfits', 'ai-process', 'try-on', 'ai-chat', 'analyze-outfit', 'create-trip-plan', 'sync-subscription']) {
  const start = performance.now();
  try {
    const response = await fetch(new URL(`/functions/v1/${name}`, process.env.EXPO_PUBLIC_SUPABASE_URL), {
      method: 'POST', headers: { apikey: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY, Authorization: `Bearer ${process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(name==='ai-process'?{image:'AA==',operation:'classify'}:{}), signal: AbortSignal.timeout(15000), redirect: 'error',
    });
    await response.arrayBuffer();
    checks.push({ endpoint: name, status: response.status, outcome: response.status===404?'not_deployed':[401,403].includes(response.status)?'unauthenticated_denied':'unexpected', deniedUnauthenticated: [401, 403].includes(response.status), latencyMs: Math.round(performance.now() - start) });
  } catch { checks.push({ endpoint: name, failure: 'network_or_timeout' }); }
}
const report = { measuredAt: new Date().toISOString(), scope: 'Gateway-valid public anon JWT, without a user identity; valid request shape where required. A gateway-only401 is insufficient to establish that a function exists.', checks };
await writeFile(new URL('../../docs/reports/readiness-evidence/production-edge-boundaries.json', import.meta.url), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (checks.some(c => !c.deniedUnauthenticated)) process.exitCode = 1;
