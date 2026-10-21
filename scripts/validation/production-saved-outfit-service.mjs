/** Execute the actual app service with installed Supabase JS and disposable users. */
import 'dotenv/config';
import { loadAppMediaForNode } from './app-media-node.mjs';
import { createClient } from '@supabase/supabase-js';
import { transpileModule, ModuleKind } from 'typescript';
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID, createHash } from 'node:crypto';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL, anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!base || !anon || !service) throw new Error('Fixture setup credential required');
const admin = createClient(base, service, { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } });
const options = { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } };
const report = { measuredAt: new Date().toISOString(), scope: 'Actual savedOutfitService source via installed Supabase JS; two disposable accounts and 251-row owner collection; not native runtime/capacity', checks: [], cleanup: [] };
const users = [];
const compile = async path => transpileModule(await readFile(new URL(path, import.meta.url), 'utf8'), { compilerOptions: { module: ModuleKind.CommonJS } }).outputText;
const events = {}, dates = {};
new Function('exports', await compile('../../store/authEvents.ts'))(events);
new Function('exports', await compile('../../src/utils/calendarDate.ts'))(dates);
const code = await compile('../../src/services/savedOutfitService.ts');
report.executedServiceJavaScriptSha256 = createHash('sha256').update(code).digest('hex');
async function serviceFor(client) {
  const appMedia = await loadAppMediaForNode(client, events, base);
  report.executedMediaJavaScriptSha256 = appMedia.hashes;
  const exports = {};
  new Function('exports', 'require', code)(exports, name => {
    if (name === '../../lib/supabase') return { supabase: client };
    if (name === '../../store/authEvents') return events;
    if (name === '../utils/calendarDate') return dates;
    if (name === './wardrobeItemMedia') return appMedia.itemMedia;
    throw new Error('Unexpected service dependency');
  });
  return Object.assign(exports, { media: appMedia.media });
}
function check(name, passed) {
  report.checks.push({ name, passed: Boolean(passed) });
  if (!passed) throw new Error('AssertionError');
}
try {
  for (let index = 0; index < 2; index++) {
    const tag = randomUUID(), email = `readiness-${tag}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await admin.auth.admin.createUser({ email, password, email_confirm: true,
      user_metadata: { username: `readiness_${tag}`, readiness_test: true } });
    if (created.error || !created.data.user) throw new Error('FixtureSetupError');
    const user = { id: created.data.user.id, client: createClient(base, anon, options) }; users.push(user);
    const login = await user.client.auth.signInWithPassword({ email, password });
    if (login.error || !login.data.session) throw new Error('FixtureLoginError');
    user.service = await serviceFor(user.client);
  }
  const [owner, other] = users;
  const rows = Array.from({ length: 251 }, () => ({ id: randomUUID(), user_id: owner.id,
    date: '2026-10-16', occasion: 'casual', name: 'Readiness saved-outfit fixture',
    items: [{ id: randomUUID(), name: 'Synthetic fixture garment', imageUrl: 'https://example.invalid/fixture.jpg' }] }));
  const inserted = await owner.client.from('saved_outfits').insert(rows);
  if (inserted.error) throw new Error('FixtureInsertError');
  events.authEvents.emitLogin(owner.id);
  let collection = await owner.service.listCloudOutfits();
  const expected = new Set(rows.map(row => row.id));
  check('actual_service_reads_all_251_rows', collection.length === 251 && new Set(collection.map(row => row.id)).size === 251
    && collection.every(row => expected.has(row.id)));
  const input = { id: rows[0].id, date: rows[0].date, occasion: 'formal', caption: 'Readiness retry fixture', items: rows[0].items };
  await owner.service.saveCloudOutfit(input); await owner.service.saveCloudOutfit(input);
  collection = await owner.service.listCloudOutfits();
  check('actual_service_repeated_save_keeps_one_stable_row', collection.length === 251 && collection.filter(row => row.id === input.id).length === 1);
  const persisted = await owner.client.from('saved_outfits').select('items,date,occasion,caption').eq('id', input.id).single();
  check('saved_fields_persist_exactly', !persisted.error && persisted.data.date === input.date && persisted.data.occasion === input.occasion
    && persisted.data.caption === input.caption && JSON.stringify(persisted.data.items) === JSON.stringify(input.items));
  events.authEvents.emitLogin(other.id);
  check('second_account_cannot_list_owner_collection', (await other.service.listCloudOutfits()).length === 0);
  let rejected = false;
  try { await other.service.saveCloudOutfit(input); } catch { rejected = true; }
  check('second_account_cannot_overwrite_owner_id', rejected);
  await other.service.removeCloudOutfit(input.id);
  events.authEvents.emitLogin(owner.id);
  check('second_account_delete_did_not_remove_owner_row', (await owner.service.listCloudOutfits()).some(row => row.id === input.id));
  await owner.service.removeCloudOutfit(input.id); await owner.service.removeCloudOutfit(input.id);
  collection = await owner.service.listCloudOutfits();
  check('actual_service_repeat_removal_is_idempotent', collection.length === 250 && !collection.some(row => row.id === input.id));
  events.authEvents.emitLogin(other.id);
  const otherInput = { ...input, id: randomUUID() };
  await other.service.saveCloudOutfit(otherInput);
  check('second_account_can_save_own_outfit', (await other.service.listCloudOutfits()).map(row => row.id).join() === otherInput.id);
  events.authEvents.emitLogin(owner.id);
  check('first_account_cannot_list_second_outfit', !(await owner.service.listCloudOutfits()).some(row => row.id === otherInput.id));
} catch (error) { report.failure = error.name; process.exitCode = 1; }
finally {
  for (const user of users) {
    try { await user.client.auth.signOut(); } catch { /* Admin cleanup still required. */ }
    let result;
    for (let attempt = 0; attempt < 3; attempt++) {
      try { result = await admin.auth.admin.deleteUser(user.id); } catch { result = { error: true }; }
      if (!result.error) break;
    }
    const passed = !result.error;
    report.cleanup.push({ temporaryAccountDeleted: passed });
    if (!passed) process.exitCode = 1;
  }
  report.passed = report.checks.length === 9 && report.checks.every(check => check.passed) && report.cleanup.length === 2
    && report.cleanup.every(check => check.temporaryAccountDeleted) && !report.failure;
  if (!report.passed) process.exitCode = 1;
  await writeFile(new URL('../../docs/reports/readiness-evidence/production-saved-outfit-service.json', import.meta.url), JSON.stringify(report, null, 2)+'\n');
  process.stdout.write(JSON.stringify(report)+'\n');
}
