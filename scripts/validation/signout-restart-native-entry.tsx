/** Native protocol audit: actual SDK/storage and intent module, synthetic HTTP only. */
import 'react-native-url-polyfill/auto';
import React, { useEffect, useState } from 'react';
import { Text, Platform } from 'react-native';
import { registerRootComponent } from 'expo';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';
import { createClient } from '@supabase/supabase-js';
import { beginSignOutIntent, hasPendingSignOutIntent, completeSignOutIntent, SIGN_OUT_INTENT_KEY } from '../../src/lib/signOutIntent';
const phaseKey = 'readiness_native_signout_phase';
const sessionKey = 'readiness_native_signout_session';
const privateKey = '@ai_chat_history';
let started = false;
function Audit() {
    const [status, setStatus] = useState('Native sign-out restart audit running');
    useEffect(() => {
        if (started) return;
        started = true;
        void (async () => {
            let phase = 0, blockedFetchAttempts = 0, syntheticHttpRequests = 0, failRemoval = false;
            globalThis.fetch = async () => { blockedFetchAttempts++; throw new Error('Audit network disabled'); };
            const report: { phase: number; platform: string; passed: boolean; checks: string[]; blockedFetchAttempts: number; syntheticHttpRequests: number; failure?: string } = {
                phase, platform: Platform.OS, passed: false, checks: [], blockedFetchAttempts, syntheticHttpRequests,
            };
            const check = (value: boolean, label: string) => { if (!value) throw new Error(label); report.checks.push(label); };
            const client = createClient('https://native-signout.invalid', 'synthetic-anon', {
                auth: { persistSession: true, autoRefreshToken: false, detectSessionInUrl: false, storageKey: sessionKey,
                    storage: { getItem: key => AsyncStorage.getItem(key), setItem: (key, value) => AsyncStorage.setItem(key, value),
                        removeItem: key => failRemoval && key === sessionKey ? Promise.reject(new Error('Synthetic SDK deletion rejection')) : AsyncStorage.removeItem(key) } },
                global: { fetch: async (input) => {
                    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
                    if (url.hostname !== 'native-signout.invalid') throw new Error('Unexpected synthetic host');
                    syntheticHttpRequests++;
                    if (url.pathname === '/auth/v1/token') return new Response(JSON.stringify({
                        access_token: 'synthetic-native-access', refresh_token: 'synthetic-native-refresh', token_type: 'bearer', expires_in: 3600,
                        user: { id: 'native-old-owner', email: 'old@example.invalid', aud: 'authenticated', role: 'authenticated' },
                    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
                    if (url.pathname === '/auth/v1/logout') return new Response('{}', { status: 200 });
                    throw new Error('Unexpected synthetic HTTP route');
                } },
            });
            try {
                phase = Number(await AsyncStorage.getItem(phaseKey) || '1'); report.phase = phase;
                if (phase === 1) {
                    const result = await client.auth.signInWithPassword({ email: 'old@example.invalid', password: 'synthetic-password' });
                    if (result.error) throw result.error;
                    await AsyncStorage.setItem(privateKey, 'synthetic-old-account-chat');
                    await beginSignOutIntent();
                    failRemoval = true; let failed = false;
                    try { const result = await client.auth.signOut(); failed = !!result.error; } catch { failed = true; }
                    check(failed, 'SDK removal rejection is observable');
                    check(JSON.parse((await AsyncStorage.getItem(sessionKey))!).user.id === 'native-old-owner', 'old SDK session remains acknowledged in native storage');
                    check(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY) === '1', 'sign-out intent is acknowledged in native storage');
                    await AsyncStorage.setItem(phaseKey, '2');
                } else if (phase === 2) {
                    check(await hasPendingSignOutIntent(), 'fresh process reads persisted sign-out intent');
                    check(JSON.parse((await AsyncStorage.getItem(sessionKey))!).user.id === 'native-old-owner', 'residual SDK session survives force-stop');
                    const result = await client.auth.signOut(); if (result.error) throw result.error;
                    const { clearAllPersistedUserData } = require('../../src/lib/persistence') as typeof import('../../src/lib/persistence');
                    await clearAllPersistedUserData();
                    const state = await client.auth.getSession(); if (state.error) throw state.error;
                    check(state.data.session === null, 'SDK confirms session absent after recovery');
                    check(await AsyncStorage.getItem(privateKey) === null, 'recovery clears previous-account chat');
                    await completeSignOutIntent();
                    check(!await hasPendingSignOutIntent(), 'completed recovery clears intent');
                    await AsyncStorage.setItem(phaseKey, '3');
                } else if (phase === 3) {
                    check(!await hasPendingSignOutIntent(), 'cleared intent remains absent after second force-stop');
                    const state = await client.auth.getSession(); if (state.error) throw state.error;
                    check(state.data.session === null && await AsyncStorage.getItem(sessionKey) === null, 'old SDK session cannot return after second restart');
                    check(await AsyncStorage.getItem(privateKey) === null, 'old chat remains absent after restart');
                    check(syntheticHttpRequests === 0, 'clean startup makes no synthetic auth request');
                    await AsyncStorage.removeItem(phaseKey);
                    check(await AsyncStorage.getItem(phaseKey) === null, 'audit phase fixture removed');
                } else throw new Error('Unexpected phase');
                check(blockedFetchAttempts === 0, 'no global app fetch attempts');
                report.passed = true;
            } catch (error) { report.failure = error instanceof Error ? error.message : 'Unknown failure'; }
            await client.auth.stopAutoRefresh();
            report.blockedFetchAttempts = blockedFetchAttempts; report.syntheticHttpRequests = syntheticHttpRequests;
            await FileSystem.writeAsStringAsync(FileSystem.documentDirectory + `signout-restart-native-phase-${phase}.json`, JSON.stringify(report));
            console.warn('SIGNOUT_RESTART_NATIVE_AUDIT ' + JSON.stringify(report));
            setStatus(`Native sign-out restart phase ${phase}: ${report.passed ? 'passed' : 'failed'}`);
        })();
    }, []);
    return <Text style={{ padding: 40 }}>{status}</Text>;
}
registerRootComponent(Audit);
