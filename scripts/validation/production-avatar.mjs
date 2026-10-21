/** Avatar lifecycle/RLS checks using only disposable production users and files. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL, anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary account setup credential required');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const users = [], files = [];
const report = { measuredAt: new Date().toISOString(), scope: 'Production avatar HTTP lifecycle using disposable accounts only; excludes native image-picker UI', checks: [], cleanup: [] };
const request = async (path, headers, method = 'GET', body, binary = false) => {
  const r = await fetch(new URL(path, base), { method, headers, ...(body === undefined ? {} : { body: binary ? body : JSON.stringify(body) }), signal: AbortSignal.timeout(30000), redirect: 'error' });
  const bytes = await r.arrayBuffer();
  let data; try { data = JSON.parse(Buffer.from(bytes).toString()); } catch { data = null; }
  return { ok: r.ok, status: r.status, data, byteLength: bytes.byteLength };
};
const check = (name, r, passed) => report.checks.push({ name, status: r.status, passed });
const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jY1EAAAAASUVORK5CYII=', 'base64');
try {
  for (let i = 0; i < 2; i++) {
    const suffix = randomUUID(), email = `readiness-${suffix}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await request('/auth/v1/admin/users', admin, 'POST', { email, password, email_confirm: true, user_metadata: { username: `readiness_${suffix}`, readiness_test: true } });
    if (!created.ok || !created.data?.id) throw new Error('Temporary user setup failed');
    const user = { id: created.data.id }; users.push(user);
    const login = await request('/auth/v1/token?grant_type=password', { apikey: anon, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!login.data?.access_token) throw new Error('Temporary sign-in failed');
    user.headers = { apikey: anon, Authorization: `Bearer ${login.data.access_token}`, 'Content-Type': 'application/json' };
  }
  const [owner, other] = users, path = `${owner.id}/avatar_readiness.png`;
  const upload = (headers, name = path, content = image, type = 'image/png', upsert = false) => request(`/storage/v1/object/avatars/${name}`, { ...headers, 'Content-Type': type, ...(upsert ? { 'x-upsert': 'true' } : {}) }, 'POST', content, true);
  const added = await upload(owner.headers);
  check('owner_upload', added, added.ok);
  if (!added.ok) throw new Error(`Avatar upload failed: HTTP ${added.status}`);
  files.push(path);
  const avatarUrl = new URL(`/storage/v1/object/authenticated/avatars/${path}`, base).href;
  const publicRead = await request(`/storage/v1/object/public/avatars/${path}`, {});
  check('avatar_not_publicly_readable', publicRead, !publicRead.ok);
  const otherRead = await request(`/storage/v1/object/authenticated/avatars/${path}`, other.headers);
  check('other_cannot_read_private_avatar', otherRead, !otherRead.ok);
  const downloaded = await request(`/storage/v1/object/authenticated/avatars/${path}`, owner.headers);
  check('avatar_image_downloads', downloaded, downloaded.ok && downloaded.byteLength === image.length);
  const changed = await request(`/rest/v1/profiles?id=eq.${owner.id}`, { ...owner.headers, Prefer: 'return=representation' }, 'PATCH', { profile_image: avatarUrl });
  check('profile_photo_saved', changed, changed.ok && changed.data?.[0]?.profile_image === avatarUrl);
  const persisted = await request(`/rest/v1/profiles?id=eq.${owner.id}&select=profile_image`, owner.headers);
  check('profile_photo_persists', persisted, persisted.ok && persisted.data?.[0]?.profile_image === avatarUrl);
  const ownerUpsert = await upload(owner.headers, path, image, 'image/png', true);
  check('owner_can_replace_avatar', ownerUpsert, ownerUpsert.ok);
  const spoof = await upload(other.headers, `${owner.id}/spoof.png`);
  if (spoof.ok) files.push(`${owner.id}/spoof.png`);
  check('other_cannot_upload_to_owner_folder', spoof, !spoof.ok);
  const overwritten = await upload(other.headers, path, image, 'image/png', true);
  check('other_cannot_overwrite_avatar', overwritten, !overwritten.ok);
  const erased = await request('/storage/v1/object/avatars', other.headers, 'DELETE', { prefixes: [path] });
  const stillThere = await request(`/storage/v1/object/authenticated/avatars/${path}`, owner.headers);
  check('other_cannot_delete_avatar', erased, stillThere.ok);
  const listed = await request('/storage/v1/object/list/avatars', other.headers, 'POST', { prefix: `${owner.id}/`, limit: 100 });
  check('other_cannot_list_owner_files', listed, listed.ok && Array.isArray(listed.data) && listed.data.length === 0);
  const badType = await upload(owner.headers, `${owner.id}/invalid.txt`, Buffer.from('test'), 'text/plain');
  if (badType.ok) files.push(`${owner.id}/invalid.txt`);
  check('non_image_upload_denied', badType, !badType.ok);
  const anonymous = await upload({ apikey: anon, Authorization: `Bearer ${anon}` }, `${owner.id}/anonymous.png`);
  if (anonymous.ok) files.push(`${owner.id}/anonymous.png`);
  check('anonymous_upload_denied', anonymous, !anonymous.ok);
  try {
    const oversized = await upload(owner.headers, `${owner.id}/oversized.png`, Buffer.alloc(5 * 1024 * 1024 + 1));
    if (oversized.ok) files.push(`${owner.id}/oversized.png`);
    check('oversized_upload_denied', oversized, !oversized.ok);
  } catch (error) {
    report.checks.push({ name: 'oversized_upload_denied', passed: false, status: null, errorType: error.name, causeCode: error.cause?.code || null });
  }
  const removed = await request('/functions/v1/delete-account', owner.headers, 'POST', {});
  check('account_deletion_success', removed, removed.ok && removed.data?.success === true);
  const absent = await request(`/storage/v1/object/authenticated/avatars/${path}`, admin);
  check('deleted_account_avatar_removed', absent, !absent.ok && [400,404].includes(absent.status));
  const intact = await request(`/auth/v1/admin/users/${other.id}`, admin);
  check('other_account_preserved', intact, intact.ok && intact.data?.id === other.id);
} catch (e) { report.failure = e instanceof Error ? e.message : 'Error'; process.exitCode = 1; }
finally {
  if (files.length) {
    const removed = await request('/storage/v1/object/avatars', admin, 'DELETE', { prefixes: files });
    report.cleanup.push({ kind: 'fixture_files', status: removed.status, passed: removed.ok });
    if (!removed.ok) process.exitCode = 1;
  }
  for (const user of users) {
    const removed = await request(`/auth/v1/admin/users/${user.id}`, admin, 'DELETE');
    report.cleanup.push({ kind: 'temporary_account', status: removed.status, passed: removed.ok || removed.status === 404 });
    if (!removed.ok && removed.status !== 404) process.exitCode = 1;
  }
  const filename = process.env.AVATAR_EVIDENCE_BEFORE === '1' ? 'production-avatar-before.json' : 'production-avatar.json';
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report,null,2));
  process.stdout.write(JSON.stringify(report,null,2) + '\n');
}
if (report.checks.some(c => !c.passed)) process.exitCode = 1;
