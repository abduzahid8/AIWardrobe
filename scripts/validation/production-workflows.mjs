/** Live CRUD/RLS checks confined to temporary users and explicitly created IDs. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const key = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Production admin credential required for temporary-account setup/cleanup');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const users = [];
const report = { measuredAt: new Date().toISOString(), scope: 'Production temporary-account wardrobe, outfit, calendar CRUD and two-user isolation; no provider calls or existing-user writes', checks: [], cleanup: [] };
const request = async (path, headers, method = 'GET', body) => {
  const response = await fetch(new URL(path, base), { method, headers: { ...headers, Prefer: 'return=representation' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(20000), redirect: 'error' });
  const raw = await response.text();
  let data; try { data = JSON.parse(raw); } catch { data = null; }
  return { status: response.status, ok: response.ok, data };
};
const check = (name, response, passed) => report.checks.push({ name, status: response.status, passed, ...(!response.ok && response.data?.code ? { code: response.data.code } : {}) });
try {
  for (let i = 0; i < 2; i++) {
    const suffix = randomUUID();
    const email = `readiness-${suffix}@aiwardrobe.club`;
    const password = `${randomUUID()}Aa!9`;
    const created = await request('/auth/v1/admin/users', admin, 'POST', { email, password, email_confirm: true, user_metadata: { username: `readiness_${suffix}`, readiness_test: true } });
    if (!created.ok || !created.data?.id) throw new Error(`Temporary account setup failed: ${created.status}`);
    const user = { id: created.data.id }; users.push(user);
    const session = await request('/auth/v1/token?grant_type=password', { apikey: key, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!session.ok || !session.data?.access_token) throw new Error(`Temporary account sign-in failed: ${session.status}`);
    user.headers = { apikey: key, Authorization: `Bearer ${session.data.access_token}`, 'Content-Type': 'application/json' };
  }
  const [a, b] = users;
  for (const table of (process.env.PRODUCTION_SECURITY_ONLY === '1' ? [] : ['clothing_items', 'saved_outfits', 'wear_logs'])) {
    const id = randomUUID();
    const payload = { id, user_id: a.id, ...(table === 'clothing_items' ? { category: 'top', name: 'Readiness test garment', image_url: 'https://example.invalid/readiness-test.png' } : table === 'saved_outfits' ? { name: 'Readiness test outfit', items: [] } : { date: '2026-10-15', item_ids: [], occasion: 'readiness test' }) };
    const path = `/rest/v1/${table}?id=eq.${id}`;
    const added = await request(`/rest/v1/${table}`, a.headers, 'POST', payload);
    check(`${table}_create`, added, added.ok && added.data?.[0]?.id === id);
    if (!added.ok) continue;
    const own = await request(path, a.headers);
    check(`${table}_persists`, own, own.ok && own.data?.length === 1 && own.data[0].user_id === a.id);
    const other = await request(path, b.headers);
    check(`${table}_other_user_cannot_read`, other, other.ok && other.data?.length === 0);
    const field = table === 'wear_logs' ? 'occasion' : 'name';
    const attack = await request(path, b.headers, 'PATCH', { [field]: 'Unauthorized change' });
    check(`${table}_other_user_cannot_update`, attack, (!attack.ok || attack.data?.length === 0));
    const attackDelete = await request(path, b.headers, 'DELETE');
    const remains = await request(path, a.headers);
    check(`${table}_other_user_cannot_delete`, attackDelete, remains.ok && remains.data?.length === 1 && remains.data[0][field] === payload[field]);
    const spoof = await request(`/rest/v1/${table}`, b.headers, 'POST', { ...payload, id: randomUUID() });
    check(`${table}_other_user_cannot_insert_for_owner`, spoof, !spoof.ok && [401, 403].includes(spoof.status));
    const updated = await request(path, a.headers, 'PATCH', { [field]: 'Readiness updated' });
    const persisted = await request(path, a.headers);
    check(`${table}_owner_update_persists`, updated, updated.ok && persisted.data?.[0]?.[field] === 'Readiness updated');
    const removed = await request(path, a.headers, 'DELETE');
    const absent = await request(path, a.headers);
    check(`${table}_owner_delete_persists`, removed, removed.ok && absent.ok && absent.data?.length === 0);
  }
  const promoted = await request(`/rest/v1/profiles?id=eq.${a.id}`, a.headers, 'PATCH', { subscription_tier: 'premium' });
  const profile = await request(`/rest/v1/profiles?id=eq.${a.id}&select=subscription_tier`, a.headers);
  check('profile_client_cannot_self_grant_paid_tier', promoted, profile.ok && profile.data?.[0]?.subscription_tier === 'free');
  report.checks.at(-1).observedTier = profile.data?.[0]?.subscription_tier || null;
  // Restore only this synthetic account immediately, regardless of whether the attack succeeded.
  await request(`/rest/v1/profiles?id=eq.${a.id}`, admin, 'PATCH', { subscription_tier: 'free' });
  if (process.env.PRODUCTION_SECURITY_ONLY === '1') {
    const adminAttack = await request(`/rest/v1/profiles?id=eq.${a.id}`, a.headers, 'PATCH', { is_admin: true, admin_role: 'super_admin' });
    const authority = await request(`/rest/v1/profiles?id=eq.${a.id}&select=is_admin,admin_role`, a.headers);
    check('client_cannot_self_grant_admin', adminAttack, [401, 403].includes(adminAttack.status) && authority.ok && authority.data?.[0]?.is_admin !== true && authority.data?.[0]?.admin_role !== 'super_admin');
    if (adminAttack.ok) await request(`/rest/v1/profiles?id=eq.${a.id}`, admin, 'PATCH', { is_admin: false, admin_role: null });
    const rpc = await request('/rest/v1/rpc/get_subscription_status', a.headers, 'POST', {});
    check('owner_subscription_status_rpc', rpc, rpc.ok && rpc.data?.length === 1 && rpc.data[0].tier === 'free');
    const subscriptionId = randomUUID();
    const forged = await request('/rest/v1/subscriptions', a.headers, 'POST', { id: subscriptionId, user_id: a.id, tier: 'premium', status: 'active' });
    check('client_cannot_insert_paid_subscription', forged, [401, 403].includes(forged.status));
    if (forged.ok) await request(`/rest/v1/subscriptions?id=eq.${subscriptionId}`, admin, 'DELETE');
    const changed = await request(`/rest/v1/profiles?id=eq.${a.id}`, a.headers, 'PATCH', { username: `readiness_updated_${randomUUID()}` });
    check('ordinary_owner_profile_edit_allowed', changed, changed.ok && changed.data?.length === 1);
    const trial = await request(`/rest/v1/profiles?id=eq.${a.id}`, a.headers, 'PATCH', { trial_started_at: '2099-01-01T00:00:00.000Z' });
    const trialRead = await request(`/rest/v1/profiles?id=eq.${a.id}&select=trial_started_at`, a.headers);
    const startedAt = Date.parse(trialRead.data?.[0]?.trial_started_at || '');
    check('initial_trial_uses_server_time', trial, trial.ok && Number.isFinite(startedAt) && Math.abs(Date.now() - startedAt) < 60000);
    const reset = await request(`/rest/v1/profiles?id=eq.${a.id}`, a.headers, 'PATCH', { trial_started_at: '2099-01-02T00:00:00.000Z' });
    check('trial_cannot_be_reset', reset, [401, 403].includes(reset.status));
    const publicHeaders = { apikey: key, Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };
    for (const [name, args] of [
      ['increment_rate_bucket', { p_user: a.id, p_feature: 'readiness:boundary' }],
      ['prune_processed_webhooks', {}],
      ['handle_failed_login', { user_email: 'readiness-boundary@example.invalid' }],
      ['handle_successful_login', { user_email: 'readiness-boundary@example.invalid', client_ip: null }],
    ]) {
      for (const [role, authHeaders] of [['anonymous', publicHeaders], ['authenticated', a.headers]]) {
        const denied = await request(`/rest/v1/rpc/${name}`, authHeaders, 'POST', args);
        check(`${role}_cannot_call_${name}`, denied, [401, 403].includes(denied.status));
      }
    }
    const serviceCounter = await request('/rest/v1/rpc/increment_rate_bucket', admin, 'POST', { p_user: a.id, p_feature: 'readiness:boundary' });
    check('service_rate_counter_remains_available', serviceCounter, serviceCounter.ok && serviceCounter.data === 1);
  }
} catch (error) {
  report.failure = error.message;
  process.exitCode = 1;
} finally {
  for (const user of users) {
    const removed = await request(`/auth/v1/admin/users/${user.id}`, admin, 'DELETE');
    report.cleanup.push({ temporaryUserDeleted: removed.ok, status: removed.status });
    if (!removed.ok) process.exitCode = 1;
  }
  const filename = process.env.PRODUCTION_SECURITY_ONLY === '1' ? 'production-profile-authority.json' : 'production-workflows.json';
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
if (report.checks.some(c => !c.passed)) process.exitCode = 1;
