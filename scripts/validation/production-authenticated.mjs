/** Controlled production reads using the repository's dedicated E2E account.
 * Never writes session tokens or credentials to the report. */
import 'dotenv/config';
import { readFile, writeFile } from 'node:fs/promises';
import { runLoad } from '../load/readiness.mjs';
import { randomUUID } from 'node:crypto';

const baseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
const apiKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const source = await readFile(new URL('../../.maestro/scenarios/_sign_in.yaml', import.meta.url), 'utf8');
const inputs = [...source.matchAll(/inputText: "([^"\n]+)"/g)].map(match => match[1]);
if (inputs.length !== 2 || !inputs[0].startsWith('e2e.')) throw new Error('Dedicated E2E credentials not found');
const signIn = (email, password) => fetch(new URL('/auth/v1/token?grant_type=password', baseUrl), {
  method: 'POST', headers: { apikey: apiKey, 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password }), signal: AbortSignal.timeout(20000),
});
let auth = await signIn(inputs[0], inputs[1]);
let createdUserId;
const adminHeaders = { apikey: process.env.SUPABASE_SERVICE_KEY, Authorization: `Bearer ${process.env.SUPABASE_SERVICE_KEY}`, 'Content-Type': 'application/json' };
if (!auth.ok) {
  const failure = await auth.json();
  console.log(JSON.stringify({ stage: 'existing_test_account_sign_in', status: auth.status, code: failure.error_code || failure.code }));
  if (!process.env.SUPABASE_SERVICE_KEY) throw new Error('Dedicated test account unavailable; admin credential missing');
  const email = `readiness-${randomUUID()}@aiwardrobe.club`;
  const password = `${randomUUID()}Aa!9`;
  const created = await fetch(new URL('/auth/v1/admin/users', baseUrl), { method: 'POST', headers: adminHeaders, body: JSON.stringify({ email, password, email_confirm: true, user_metadata: { readiness_test: true } }), signal: AbortSignal.timeout(20000) });
  if (!created.ok) throw new Error(`Dedicated account creation failed: HTTP ${created.status}`);
  createdUserId = (await created.json()).id;
  if (!createdUserId) throw new Error('Created account ID missing');
  auth = await signIn(email, password);
}
try {
if (!auth.ok) throw new Error(`Dedicated test sign-in failed: HTTP ${auth.status}`);
const session = await auth.json();
if (!session.user?.id || !session.access_token) throw new Error('Invalid test session');
const headers = { apikey: apiKey, Authorization: `Bearer ${session.access_token}`, 'Content-Type': 'application/json' };
const report = { measuredAt: new Date().toISOString(), scope: 'Live production; one dedicated test account; authenticated read paths only', checks: [], loads: [] };
for (const table of ['profiles', 'clothing_items', 'saved_outfits']) {
  const owner = table === 'profiles' ? 'id' : 'user_id';
  const response = await fetch(new URL(`/rest/v1/${table}?select=${owner}&${owner}=neq.${session.user.id}&limit=1`, baseUrl), { headers, signal: AbortSignal.timeout(15000) });
  const rows = await response.json();
  report.checks.push({ name: `${table}_other_owner_hidden`, status: response.status, passed: response.ok && Array.isArray(rows) && rows.length === 0 });
}
const rpc = await fetch(new URL('/rest/v1/rpc/get_subscription_status', baseUrl), { method: 'POST', headers, body: '{}', signal: AbortSignal.timeout(15000) });
const rpcBody = await rpc.json();
report.checks.push({ name: 'subscription_status_rpc', status: rpc.status, passed: rpc.ok, ...(rpc.ok ? {} : { code: rpcBody.code }) });
console.log(JSON.stringify({ stage: 'authenticated_checks', checks: report.checks }));
const users = [{ id: session.user.id, accessToken: session.access_token }];
const baseline = await runLoad({ baseUrl, apiKey, users, rps: 5, durationSec: 30 });
report.loads.push({ stage: 'baseline', ...baseline });
console.log(JSON.stringify({ stage: 'baseline', ...baseline }));
if (baseline.passed && report.checks.filter(c => c.name !== 'subscription_status_rpc').every(c => c.passed)) {
  const durationSec = Number(process.env.PRODUCTION_LOAD_DURATION_SEC || 600);
  console.log(JSON.stringify({ stage: 'peak_started', rps: 40, durationSec }));
  report.loads.push({ stage: 'peak', ...await runLoad({ baseUrl, apiKey, users, rps: 40, durationSec }) });
}
await writeFile(new URL('../../docs/reports/readiness-evidence/production-authenticated.json', import.meta.url), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (report.checks.some(c => !c.passed) || report.loads.some(l => !l.passed)) process.exitCode = 1;
} finally {
  if (createdUserId) {
    const cleanup = await fetch(new URL(`/auth/v1/admin/users/${createdUserId}`, baseUrl), { method: 'DELETE', headers: adminHeaders, signal: AbortSignal.timeout(20000) });
    console.log(JSON.stringify({ stage: 'dedicated_test_account_cleanup', passed: cleanup.ok, status: cleanup.status }));
    if (!cleanup.ok) throw new Error(`Dedicated test account cleanup failed: ${createdUserId}`);
  }
}
