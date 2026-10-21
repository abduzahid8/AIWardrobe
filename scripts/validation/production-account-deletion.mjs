/** Account deletion test confined to newly created disposable users/files. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary account setup credential required');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const users = [], objects = [];
const report = { measuredAt: new Date().toISOString(), scope: 'Disposable production accounts and explicitly created files only; no existing-user changes', checks: [], cleanup: [] };
const call = async (path, headers, method = 'GET', body) => {
  const r = await fetch(new URL(path, base), { method, headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(60000), redirect: 'error' });
  const data = await r.json().catch(() => null);
  return { status: r.status, ok: r.ok, data };
};
const check = (name, r, passed) => report.checks.push({ name, status: r.status, passed });
const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jY1EAAAAASUVORK5CYII=', 'base64');
try {
  for (let i = 0; i < 2; i++) {
    const suffix = randomUUID(), email = `readiness-${suffix}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await call('/auth/v1/admin/users', admin, 'POST', { email, password, email_confirm: true, user_metadata: { username: `readiness_${suffix}`, readiness_test: true } });
    if (!created.ok || !created.data?.id) throw new Error('Temporary user creation failed');
    const user = { id: created.data.id }; users.push(user);
    const session = await call('/auth/v1/token?grant_type=password', { apikey: anon, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!session.ok || !session.data?.access_token) throw new Error('Temporary user login failed');
    user.headers = { apikey: anon, Authorization: `Bearer ${session.data.access_token}`, 'Content-Type': 'application/json' };
  }
  const [owner, control] = users;
  if (process.env.DELETION_CHECK_METHODS === '1') {
    const readOnly = await call('/functions/v1/delete-account', owner.headers);
    check('get_cannot_delete_account', readOnly, readOnly.status === 405);
    for (const [role, headers] of [['anonymous', { apikey: anon, Authorization: `Bearer ${anon}`, 'Content-Type': 'application/json' }], ['authenticated', owner.headers]]) {
      const denied = await call('/rest/v1/rpc/list_account_deletion_objects', headers, 'POST', { p_user_id: control.id });
      check(`${role}_cannot_enumerate_storage`, denied, [401, 403].includes(denied.status));
    }
  }
  for (const [user, bucket, name] of [
    [owner, 'user_uploads', `${owner.id}/readiness-root.png`],
    [owner, 'user_uploads', `${owner.id}/nested/readiness.png`],
    [owner, 'AIWARDROBE', `readiness-${owner.id}.png`],
    [control, 'user_uploads', `${control.id}/readiness-control.png`],
  ]) {
    const r = await fetch(new URL(`/storage/v1/object/${bucket}/${name}`, base), { method: 'POST', headers: { ...user.headers, 'Content-Type': 'image/png' }, body: image, signal: AbortSignal.timeout(20000) });
    if (!r.ok) throw new Error(`Fixture upload failed: HTTP ${r.status}`);
    objects.push({ bucket, name });
  }
  const garment = await call('/rest/v1/clothing_items', owner.headers, 'POST', { id: randomUUID(), user_id: owner.id, category: 'top', name: 'Readiness deletion fixture', image_url: 'https://example.invalid/deletion.png' });
  if (!garment.ok) throw new Error('Garment fixture failed');
  const denied = await call('/functions/v1/delete-account', { apikey: anon, 'Content-Type': 'application/json' }, 'POST', {});
  check('anonymous_delete_denied', denied, denied.status === 401);
  const removed = await call('/functions/v1/delete-account', owner.headers, 'POST', { user_id: control.id });
  check('owner_delete_success', removed, removed.ok && removed.data?.success === true);
  const auth = await call(`/auth/v1/admin/users/${owner.id}`, admin);
  check('owner_auth_removed', auth, auth.status === 404);
  const profile = await call(`/rest/v1/profiles?id=eq.${owner.id}&select=id`, admin);
  check('owner_profile_removed', profile, profile.ok && profile.data?.length === 0);
  const item = await call(`/rest/v1/clothing_items?user_id=eq.${owner.id}&select=id`, admin);
  check('owner_garments_removed', item, item.ok && item.data?.length === 0);
  for (const file of objects) {
    const r = await fetch(new URL(`/storage/v1/object/authenticated/${file.bucket}/${file.name}`, base), { headers: admin, signal: AbortSignal.timeout(20000) });
    await r.arrayBuffer();
    const isControl = file.name.startsWith(control.id + '/');
    check(isControl ? 'other_user_file_preserved' : `owner_file_removed_${file.bucket}_${file.name.includes('/nested/') ? 'nested' : 'root'}`, r, isControl ? r.ok : !r.ok && [400, 404].includes(r.status));
  }
  const other = await call(`/auth/v1/admin/users/${control.id}`, admin);
  check('body_cannot_target_other_user', other, other.ok && other.data?.id === control.id);
  if (removed.ok && removed.data?.success === true) {
    const staleToken = await call('/auth/v1/user', owner.headers);
    check('deleted_account_token_cannot_authenticate', staleToken, [401, 403].includes(staleToken.status));
    if (process.env.DELETION_CHECK_STALE_WRITES === '1') {
      for (const bucket of ['avatars', 'user_uploads']) {
        const name = `${owner.id}/readiness-after-deletion.png`;
        objects.push({ bucket, name });
        const response = await fetch(new URL(`/storage/v1/object/${bucket}/${name}`, base), {
          method: 'POST', headers: { ...owner.headers, 'Content-Type': 'image/png' }, body: image, signal: AbortSignal.timeout(20000),
        });
        await response.arrayBuffer();
        check(`deleted_token_cannot_upload_${bucket}`, response, !response.ok && [400, 401, 403].includes(response.status));
      }
      const row = await call('/rest/v1/clothing_items', owner.headers, 'POST', { id: randomUUID(), user_id: owner.id, category: 'top', name: 'Deleted identity probe' });
      check('deleted_token_cannot_create_garment', row, !row.ok && [400, 401, 403, 409].includes(row.status));
    }

  }
} catch (e) {
  report.failure = e instanceof Error ? e.message.replace(/(?:eyJ|AIza|AQ\.)\S+/g, '[REDACTED]') : 'Error';
  process.exitCode = 1;
} finally {
  for (const bucket of [...new Set(objects.map(o => o.bucket))]) {
    const removed = await call(`/storage/v1/object/${bucket}`, admin, 'DELETE', { prefixes: objects.filter(o => o.bucket === bucket).map(o => o.name) });
    report.cleanup.push({ kind: 'fixture_files', bucket, status: removed.status, passed: removed.ok });
    if (!removed.ok) process.exitCode = 1;
  }
  for (const user of users) {
    const removed = await call(`/auth/v1/admin/users/${user.id}`, admin, 'DELETE');
    report.cleanup.push({ kind: 'temporary_account', status: removed.status, passed: removed.ok || removed.status === 404 });
    if (!removed.ok && removed.status !== 404) process.exitCode = 1;
  }
  const filename = process.env.DELETION_CHECK_STALE_WRITES === '1' ? 'production-deleted-token-writes.json' : process.env.DELETION_EVIDENCE_BEFORE === '1' ? 'production-account-deletion-before.json' : 'production-account-deletion.json';
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report, null, 2));
  process.stdout.write(JSON.stringify(report, null, 2) + '\n');
}
if (report.checks.some(c => !c.passed)) process.exitCode = 1;
