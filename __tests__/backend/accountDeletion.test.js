/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/delete-account/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
function load({ authenticated = true, enumerateError = null, removeError = null, files = [], stuck = false } = {}) {
  let handler, page = 0;
  const events = [];
  const rpc = jest.fn(async () => {
    events.push('enumerate');
    return { data: stuck || page++ === 0 ? files : [], error: enumerateError };
  });
  const remove = jest.fn(async () => { events.push('remove'); return { error: removeError }; });
  const deleteUser = jest.fn(async () => { events.push('deleteUser'); return { error: null }; });
  const createClient = () => ({
    auth: { getUser: async () => ({ data: { user: authenticated ? { id: 'owner' } : null }, error: null }), admin: { deleteUser } },
    rpc,
    storage: { from: bucket => ({ remove: paths => remove({ bucket, paths }) }) },
  });
  new Function('require', 'exports', 'Deno', compiled)(() => ({ createClient }), {}, {
    serve: fn => { handler = fn; }, env: { get: () => 'test-configuration' },
  });
  return { handler, rpc, remove, deleteUser, events };
}
const request = (method = 'POST') => new Request('https://example.test/delete-account', {
  method, headers: { Authorization: 'Bearer user-jwt' },
  ...(method === 'POST' ? { body: JSON.stringify({ user_id: 'other-user' }) } : {}),
});
test('deletes only authenticated identity after nested and cross-bucket files are removed', async () => {
  const { handler, rpc, remove, deleteUser, events } = load({ files: [
    { bucket_id: 'user_uploads', name: 'owner/nested/photo.png' },
    { bucket_id: 'AIWARDROBE', name: 'legacy-photo.png' },
  ] });
  expect((await handler(request())).status).toBe(200);
  expect(rpc.mock.calls.every(([name, args]) => name === 'list_account_deletion_objects' && args.p_user_id === 'owner')).toBe(true);
  expect(remove.mock.calls.map(([args]) => args)).toEqual([
    { bucket: 'user_uploads', paths: ['owner/nested/photo.png'] },
    { bucket: 'AIWARDROBE', paths: ['legacy-photo.png'] },
  ]);
  expect(deleteUser).toHaveBeenCalledWith('owner');
  expect(events.at(-1)).toBe('deleteUser');
});
test.each([{ enumerateError: { code: 'unavailable' } }, { removeError: { status: 503 }, files: [{ bucket_id: 'user_uploads', name: 'owner/photo.png' }] }])('storage failure preserves the account and reports failure', async options => {
  const { handler, deleteUser } = load(options);
  expect((await handler(request())).status).toBe(503);
  expect(deleteUser).not.toHaveBeenCalled();
});
test('bounded cleanup never reports success if storage continues returning files', async () => {
  const { handler, deleteUser } = load({ stuck: true, files: [{ bucket_id: 'user_uploads', name: 'owner/photo.png' }] });
  expect((await handler(request())).status).toBe(503);
  expect(deleteUser).not.toHaveBeenCalled();
});
test('unauthenticated calls never enumerate or delete anything', async () => {
  const { handler, rpc, deleteUser } = load({ authenticated: false });
  expect((await handler(request())).status).toBe(401);
  expect(rpc).not.toHaveBeenCalled();
  expect(deleteUser).not.toHaveBeenCalled();
});
test('GET cannot trigger deletion', async () => {
  const { handler, rpc, deleteUser } = load();
  expect((await handler(request('GET'))).status).toBe(405);
  expect(rpc).not.toHaveBeenCalled();
  expect(deleteUser).not.toHaveBeenCalled();
});
