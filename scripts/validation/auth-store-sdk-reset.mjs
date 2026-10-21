import { createClient } from '@supabase/supabase-js';
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const root = new URL('../../', import.meta.url), sources = {};
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
async function load(path, dependencies) {
    const source = await readFile(new URL(path, root), 'utf8');
    sources[path] = createHash('sha256').update(source).digest('hex');
    const js = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
    const module = { exports: {} };
    new Function('require', 'module', 'exports', js)(name => {
        if (!(name in dependencies)) throw new Error(`Unexpected diagnostic dependency: ${name}`);
        return dependencies[name];
    }, module, module.exports);
    return module.exports;
}
let { authEvents } = await load('store/authEvents.ts', {});
const { createAuthOperationQueue } = await load('src/lib/authOperationQueue.ts', {});
const disk = new Map(), events = [], requests = [], tokenGrants = [];
const response = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
let failSessionRemoval = false;
function makeSdk(autoRefresh = false) { return createClient('https://auth-store-sdk.invalid', 'synthetic-anon', {
    auth: { persistSession: true, autoRefreshToken: autoRefresh, detectSessionInUrl: false, storageKey: 'synthetic-session', storage: {
        getItem: async key => disk.get(key) ?? null,
        setItem: async (key, value) => { disk.set(key, value); },
        removeItem: async key => {
            if (key === 'synthetic-session' && failSessionRemoval) throw new Error('Synthetic SDK session removal failure');
            disk.delete(key);
        },
    } },
    global: { fetch: async (url, init) => {
        const target = new URL(url);
        if (target.hostname !== 'auth-store-sdk.invalid') throw new Error('Unexpected network target');
        requests.push(target.pathname);
        if (target.pathname === '/auth/v1/token') {
            const grant = target.searchParams.get('grant_type'); tokenGrants.push(grant);
            if (grant === 'refresh_token' && rejectExpiredRefresh) return response({ message: 'Synthetic refresh token rejected', code: 'refresh_token_not_found' }, 400);
            const body = JSON.parse(init.body);
            const owner = grant === 'refresh_token' ? body.refresh_token.replace(/^synthetic-refresh-/, '') : body.email.split('@')[0];
            return response({ access_token: `synthetic-${owner}`, refresh_token: `synthetic-refresh-${owner}`, token_type: 'bearer', expires_in: 3600,
                user: { id: owner, email: `${owner}@example.invalid`, aud: 'authenticated', role: 'authenticated' } });
        }
        if (target.pathname === '/rest/v1/profiles') {
            const owner = target.searchParams.get('id')?.replace(/^eq\./, '');
            return response({ id: owner, email: `${owner}@example.invalid`, username: owner });
        }
        if (target.pathname === '/auth/v1/logout') return response({});
        throw new Error(`Unexpected synthetic route: ${target.pathname}`);
    } },
}); }
let sdk = makeSdk();
sdk.auth.onAuthStateChange((event, session) => { events.push({ event, owner: session?.user.id ?? null }); });
async function makeIntents() {
    return load('src/lib/signOutIntent.ts', { '@react-native-async-storage/async-storage': { __esModule: true, default: {
        getItem: async key => disk.get(key) ?? null,
        setItem: async (key, value) => { disk.set(key, value); },
        removeItem: async key => { disk.delete(key); },
    } } });
}
let intents = await makeIntents();
let resetGate = null, resetStarted = null, heldResets = 0;
const noop = () => undefined, asyncNoop = async () => undefined;
async function makeStore() { return load('store/auth.ts', {
    zustand: require('zustand'), 'react-native': { Platform: { OS: 'android' } },
    'expo-web-browser': {}, 'expo-linking': { createURL: () => 'synthetic://auth/callback' },
    '../lib/supabase': { supabase: sdk },
    '../src/lib/requestDeadline': { withRequestDeadline: work => work(new AbortController().signal) },
    '../src/services/analyticsService': { analyticsService: { trackLogin: noop, trackSignup: noop, trackEvent: noop, setUserId: noop, clearUserId: noop } },
    '../src/services/crashReporting': { crashReporting: { setUser: noop, clearUser: noop } },
    '../src/services/iapService': { iapService: { identify: asyncNoop, logout: asyncNoop } },
    './subscriptionStore': { __esModule: true, default: { getState: () => ({ verifySubscriptionFromServer: asyncNoop, initializeTrial: asyncNoop }) } },
    '../src/utils/logger': { createLogger: () => ({ debug: noop, info: noop, warn: noop, error: noop }) },
    '../src/lib/persistence': { ensurePreviousAccountCleanup: asyncNoop, clearAllPersistedUserData: asyncNoop, resetInMemoryUserState: () => {
        if (resetGate) { heldResets++; resetStarted.resolve(); return resetGate.promise; }
        return Promise.resolve();
    } },
    './authEvents': { authEvents }, '../src/services/uploadQueue': { uploadQueue: { destroy: noop, init: noop } },
    '../src/lib/authOperationQueue': { createAuthOperationQueue },
    '../src/lib/signOutIntent': intents,
    './wardrobeStore': { __esModule: true, default: { getState: () => ({ rehydrateFromCloud: asyncNoop }) } },
}); }
let module = await makeStore();
let store = module.default;
const watchdog = setTimeout(() => { process.stderr.write('Actual SDK/store diagnostic exceeded 10 seconds\n'); process.exit(2); }, 10000);
const failureMode = process.argv.includes('--logout-storage-failure');
const rejectExpiredRefresh = process.argv.includes('--expired-refresh-rejected');
const expiredIntentMode = process.argv.includes('--expired-signout-intent') || rejectExpiredRefresh;
const destination = process.argv.slice(2).find(value => !['--logout-storage-failure', '--expired-signout-intent', '--expired-refresh-rejected'].includes(value));
async function emit(report) {
    report.sdkVersion = require('@supabase/auth-js/package.json').version;
    report.diagnosticSha256 = createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex');
    if (destination) await writeFile(destination, `${JSON.stringify(report, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify(report)}\n`);
    if (!report.passed) process.exitCode = 2;
}
try {
    if (expiredIntentMode) {
        await store.getState().initializeAuth();
        await store.getState().login('old-owner@example.invalid', 'synthetic-password');
        await intents.beginSignOutIntent();
        const saved = JSON.parse(disk.get('synthetic-session'));
        saved.expires_at = Math.floor(Date.now() / 1000) - 120;
        disk.set('synthetic-session', JSON.stringify(saved));
        module.cleanupAuthSubscription(); await sdk.auth.stopAutoRefresh();
        ({ authEvents } = await load('store/authEvents.ts', {}));
        sdk = makeSdk(true); intents = await makeIntents(); module = await makeStore(); store = module.default;
        const refreshesBeforeAppInit = tokenGrants.filter(grant => grant === 'refresh_token').length;
        const publishedOwners = [];
        const unsubscribe = store.subscribe(state => { if (state.isAuthenticated) publishedOwners.push(state.session?.user.id); });
        try { await store.getState().initializeAuth(); }
        finally { unsubscribe(); }
        const restoredOwner = store.getState().session?.user.id ?? null;
        await emit({ scope: 'Actual app store and installed SDK, expired synthetic persisted session plus saved sign-out intent, automatic SDK refresh enabled; memory storage and intercepted HTTP, no OS restart',
            passed: refreshesBeforeAppInit > 0 && tokenGrants.includes('refresh_token') && restoredOwner === null && publishedOwners.length === 0 && !disk.has('synthetic-session') && !disk.has('auth_signout_pending_v1'),
            restoredOwner, publishedOwners, refreshesBeforeAppInit, refreshResponse: rejectExpiredRefresh ? 'synthetic server rejection' : 'synthetic success', tokenGrants, requests, intentRemovedAfterRecovery: !disk.has('auth_signout_pending_v1'), sdkSessionRemovedAfterRecovery: !disk.has('synthetic-session'), sourceSha256: sources });
    } else if (failureMode) {
        await store.getState().initializeAuth();
        await store.getState().login('old-owner@example.invalid', 'synthetic-password');
        failSessionRemoval = true;
        let logoutResolved = false, logoutRejected = false;
        try { await store.getState().logout(); logoutResolved = true; } catch { logoutRejected = true; }
        const afterLogout = { logoutResolved, logoutRejected, appOwner: store.getState().session?.user.id ?? null, storedOwner: disk.has('synthetic-session') ? JSON.parse(disk.get('synthetic-session')).user.id : null, pendingIntentSaved: disk.get('auth_signout_pending_v1') === '1' };
        module.cleanupAuthSubscription();
        failSessionRemoval = false;
        ({ authEvents } = await load('store/authEvents.ts', {}));
        sdk = makeSdk(); intents = await makeIntents(); module = await makeStore(); store = module.default;
        await store.getState().initializeAuth();
        const restoredOwner = store.getState().session?.user.id ?? null;
        await emit({ scope: 'Actual app auth-store and installed SDK, transient synthetic session-removal failure then fresh SDK/store module state over the same memory disk; no OS restart or production calls', passed: afterLogout.logoutRejected && !afterLogout.logoutResolved && afterLogout.storedOwner === 'old-owner' && afterLogout.pendingIntentSaved && restoredOwner === null && !disk.has('synthetic-session') && !disk.has('auth_signout_pending_v1'), afterLogout, restoredOwner, intentRemovedAfterRecovery: !disk.has('auth_signout_pending_v1'), sdkSessionRemovedAfterRecovery: !disk.has('synthetic-session'), requests, sourceSha256: sources });
    } else {
    await store.getState().initializeAuth();
    await new Promise(done => setImmediate(done));
    resetGate = deferred(); resetStarted = deferred();
    let completed = false;
    const login = store.getState().login('new-owner@example.invalid', 'synthetic-password').then(() => { completed = true; });
    await resetStarted.promise;
    await new Promise(done => setImmediate(done));
    const during = { loginCompleted: completed, authenticated: store.getState().isAuthenticated, cacheOwner: authEvents.getUserId(), sdkSavedOwner: JSON.parse(disk.get('synthetic-session')).user.id };
    resetGate.resolve(); await login;
    const after = { authenticated: store.getState().isAuthenticated, owner: store.getState().session?.user.id, cacheOwner: authEvents.getUserId(), sdkSavedOwner: JSON.parse(disk.get('synthetic-session')).user.id, heldResets };
    const passed = !during.loginCompleted && !during.authenticated && during.cacheOwner === null && during.sdkSavedOwner === 'new-owner'
        && after.authenticated && after.owner === 'new-owner' && after.cacheOwner === 'new-owner' && after.sdkSavedOwner === 'new-owner' && heldResets === 1;
    const report = { scope: 'Actual app auth-store source and installed SDK, synthetic HTTP and asynchronous in-memory session storage; reset/IAP/UI services stubbed', passed, during, after, requests, events, sourceSha256: sources };
    await emit(report);
    }
} finally { module.cleanupAuthSubscription(); await sdk.auth.stopAutoRefresh(); clearTimeout(watchdog); }
