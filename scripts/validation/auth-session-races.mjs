import { createClient } from '@supabase/supabase-js';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import ts from 'typescript';
const require = createRequire(import.meta.url);

// Actual installed SDK, synthetic responses only. No credentials or live network.
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};
const response = owner => new Response(JSON.stringify({
  access_token: `synthetic-${owner}`, refresh_token: `synthetic-refresh-${owner}`,
  token_type: 'bearer', expires_in: 3600,
  user: { id: owner, aud: 'authenticated', role: 'authenticated', email: `${owner}@example.invalid` },
}), { status: 200, headers: { 'Content-Type': 'application/json' } });

function fixture(name) {
  const disk = new Map(), events = [], pending = [];
  const client = createClient('https://auth-races.invalid', 'synthetic-anon', {
    auth: { persistSession: true, storageKey: `readiness-${name}`, autoRefreshToken: false,
      detectSessionInUrl: false, storage: {
        getItem: async key => disk.get(key) ?? null,
        setItem: async (key, value) => { disk.set(key, value); },
        removeItem: async key => { disk.delete(key); },
      } },
    global: { fetch: async (url, init) => {
      const target = new URL(url);
      if (target.hostname !== 'auth-races.invalid') throw new Error('Unexpected network target');
      if (target.pathname.endsWith('/token') || target.pathname.endsWith('/logout')) {
        const gate = deferred();
        pending.push({ path: target.pathname, body: init?.body ? JSON.parse(init.body) : null, gate });
        return gate.promise;
      }
      throw new Error(`Unexpected synthetic route: ${target.pathname}`);
    } },
  });
  client.auth.onAuthStateChange((event, session) => { events.push({ event, owner: session?.user.id ?? null }); });
  const owner = () => {
    const stored = disk.get(`readiness-${name}`);
    return stored ? JSON.parse(stored).user.id : null;
  };
  const waitRequest = async count => {
    const deadline = Date.now() + 3000;
    while (pending.length < count) {
      if (Date.now() >= deadline) throw new Error('Synthetic request did not start');
      await new Promise(done => setTimeout(done, 1));
    }
    return pending[count - 1];
  };
  return { client, events, pending, owner, waitRequest };
}

async function loginAfterLogout() {
  const f = fixture('login-after-logout');
  await f.client.auth.getSession();
  const login = f.client.auth.signInWithPassword({ email: 'old@example.invalid', password: 'synthetic' });
  const request = await f.waitRequest(1);
  const logout = await f.client.auth.signOut();
  if (logout.error) throw logout.error;
  const ownerAfterLogout = f.owner();
  request.gate.resolve(response('old'));
  const result = await login;
  if (result.error) throw result.error;
  return { scenario: 'sign-in response arrives after completed logout', expectedOwner: null,
    ownerAfterLogout, actualOwner: f.owner(), passed: f.owner() === null, events: f.events };
}
async function reversedLogins() {
  const f = fixture('reversed-logins');
  await f.client.auth.getSession();
  const old = f.client.auth.signInWithPassword({ email: 'old@example.invalid', password: 'synthetic' });
  const first = await f.waitRequest(1);
  const newer = f.client.auth.signInWithPassword({ email: 'new@example.invalid', password: 'synthetic' });
  const second = await f.waitRequest(2);
  second.gate.resolve(response('new'));
  if ((await newer).error) throw new Error('Synthetic new login failed');
  const ownerBeforeOldResponse = f.owner();
  first.gate.resolve(response('old'));
  if ((await old).error) throw new Error('Synthetic old login failed');
  return { scenario: 'older sign-in response arrives after newer sign-in', expectedOwner: 'new',
    ownerBeforeOldResponse, actualOwner: f.owner(), passed: f.owner() === 'new', events: f.events };
}
async function delayedLogout() {
  const f = fixture('delayed-logout');
  await f.client.auth.getSession();
  const initial = f.client.auth.signInWithPassword({ email: 'old@example.invalid', password: 'synthetic' });
  (await f.waitRequest(1)).gate.resolve(response('old')); await initial;
  const logout = f.client.auth.signOut();
  const request = await f.waitRequest(2);
  const newer = f.client.auth.signInWithPassword({ email: 'new@example.invalid', password: 'synthetic' });
  (await f.waitRequest(3)).gate.resolve(response('new')); await newer;
  const ownerBeforeLogoutResponse = f.owner();
  request.gate.resolve(new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }));
  if ((await logout).error) throw new Error('Synthetic logout failed');
  return { scenario: 'old logout response arrives after newer sign-in', expectedOwner: 'new',
    ownerBeforeLogoutResponse, actualOwner: f.owner(), passed: f.owner() === 'new', events: f.events };
}
async function sequentialControl() {
  const f = fixture('sequential-control');
  await f.client.auth.getSession();
  const login = f.client.auth.signInWithPassword({ email: 'old@example.invalid', password: 'synthetic' });
  (await f.waitRequest(1)).gate.resolve(response('old'));
  if ((await login).error) throw new Error('Synthetic control login failed');
  const signedInOwner = f.owner();
  const logout = f.client.auth.signOut();
  (await f.waitRequest(2)).gate.resolve(new Response('{}', { status: 200 }));
  if ((await logout).error) throw new Error('Synthetic control logout failed');
  return { scenario: 'sequential sign-in then sign-out control', expectedOwner: null,
    signedInOwner, actualOwner: f.owner(), passed: signedInOwner === 'old' && f.owner() === null, events: f.events };
}
const queueSource = await readFile(new URL('../../src/lib/authOperationQueue.ts', import.meta.url), 'utf8');
const compiledQueue = ts.transpileModule(queueSource, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { createAuthOperationQueue } = await import(`data:text/javascript;base64,${Buffer.from(compiledQueue).toString('base64')}`);
async function coordinated(firstKind, secondKind) {
  const f = fixture(`coordinated-${firstKind}-${secondKind}`), queue = createAuthOperationQueue();
  await f.client.auth.getSession();
  const execute = kind => kind === 'logout' ? f.client.auth.signOut()
    : f.client.auth.signInWithPassword({ email: `${kind}@example.invalid`, password: 'synthetic' });
  if (firstKind === 'logout') {
    const initial = execute('old'); (await f.waitRequest(1)).gate.resolve(response('old')); await initial;
  }
  const offset = f.pending.length;
  const first = queue.run(() => execute(firstKind));
  const firstRequest = await f.waitRequest(offset + 1);
  const second = queue.run(() => execute(secondKind));
  // Flush a full event-loop turn to detect an incorrectly concurrent SDK request.
  await new Promise(done => setImmediate(done));
  const noOverlap = f.pending.length === offset + 1;
  firstRequest.gate.resolve(firstKind === 'logout' ? new Response('{}', { status: 200 }) : response(firstKind));
  if ((await first).error) throw new Error('First coordinated operation failed');
  const secondRequest = await f.waitRequest(offset + 2);
  secondRequest.gate.resolve(secondKind === 'logout' ? new Response('{}', { status: 200 }) : response(secondKind));
  if ((await second).error) throw new Error('Second coordinated operation failed');
  const expectedOwner = secondKind === 'logout' ? null : secondKind;
  return { scenario: `coordinated ${firstKind} then ${secondKind}`, noOverlappingSdkRequest: noOverlap,
    expectedOwner, actualOwner: f.owner(), passed: noOverlap && expectedOwner === f.owner(), events: f.events };
}
const coordinatedMode = process.argv.includes('--coordinated');
const cases = [await sequentialControl()];
if (coordinatedMode) {
  for (const pair of [['old', 'logout'], ['old', 'new'], ['logout', 'new']]) cases.push(await coordinated(...pair));
} else {
  for (const run of [loginAfterLogout, reversedLogins, delayedLogout]) cases.push(await run());
}
const sdkPackage = JSON.parse(await readFile(require.resolve('@supabase/auth-js/package.json'), 'utf8'));
const report = { mode: coordinatedMode ? 'candidate-coordination-not-integrated' : 'sdk-baseline', queueSourceSha256: createHash('sha256').update(queueSource).digest('hex'), sdkVersion: sdkPackage.version, sourceSha256: createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex'), scope: 'Actual installed Supabase SDK, isolated asynchronous in-memory storage and synthetic fetch; no live network or UI',
  passed: cases.every(value => value.passed), cases };
const destination = process.argv.slice(2).find(value => value !== '--coordinated');
if (destination) { await mkdir(dirname(resolve(destination)), { recursive: true }); await writeFile(destination, `${JSON.stringify(report, null, 2)}\n`); }
process.stdout.write(`${JSON.stringify(report)}\n`);
if (!report.passed) process.exitCode = 2;
