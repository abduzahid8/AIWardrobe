/** Explicit native audit entry: synthetic keys, no credentials, all fetch blocked. */
import React, { useEffect, useState } from 'react';
import { Text, Platform } from 'react-native';
import { registerRootComponent } from 'expo';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';

const phaseKey = 'readiness_cleanup_restart_phase';
const privateKey = '@ai_chat_history';
const privatePlan = 'outfitLogs::11111111-1111-4111-8111-111111111111';
let started = false;
function Audit() {
    const [status, setStatus] = useState('Native cleanup restart audit running');
    useEffect(() => {
        if (started) return;
        started = true;
        void (async () => {
            let phase = 0, networkAttempts = 0;
            globalThis.fetch = async () => { networkAttempts++; throw new Error('Audit network disabled'); };
            const report: { phase: number; platform: string; passed: boolean; checks: string[]; networkAttempts: number; failure?: string } = {
                phase, platform: Platform.OS, passed: false, checks: [], networkAttempts,
            };
            const check = (value: boolean, message: string) => { if (!value) throw new Error(message); report.checks.push(message); };
            try {
                phase = Number(await AsyncStorage.getItem(phaseKey) || '1'); report.phase = phase;
                const { ensurePreviousAccountCleanup, clearAllPersistedUserData, ACCOUNT_CLEANUP_PENDING_KEY } = require('../../src/lib/persistence') as typeof import('../../src/lib/persistence');
                if (phase === 1) {
                    await AsyncStorage.setItem(privateKey, 'synthetic-previous-owner-chat');
                    await AsyncStorage.setItem(privatePlan, 'synthetic-previous-owner-plan');
                    const original = AsyncStorage.multiRemove.bind(AsyncStorage);
                    let injected = false, failed = false;
                    AsyncStorage.multiRemove = (...args: Parameters<typeof AsyncStorage.multiRemove>) => {
                        if (args[0].includes('wardrobe-storage') && !injected) {
                            injected = true; return Promise.reject(new Error('Synthetic registry operation rejection'));
                        }
                        return original(...args);
                    };
                    try { await clearAllPersistedUserData(); } catch { failed = true; }
                    finally { AsyncStorage.multiRemove = original; }
                    check(injected && failed, 'injected registry rejection is reported');
                    check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === '1', 'native cleanup marker acknowledged on disk');
                    check(await AsyncStorage.getItem(privateKey) === 'synthetic-previous-owner-chat', 'failed registry leaves fixture chat for restart recovery');
                    await AsyncStorage.setItem(phaseKey, '2');
                } else if (phase === 2) {
                    check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === '1', 'marker survives force-stop into a fresh native process');
                    check(await AsyncStorage.getItem(privateKey) === 'synthetic-previous-owner-chat', 'fixture chat survives with pending marker');
                    await ensurePreviousAccountCleanup();
                    check(await AsyncStorage.getItem(privateKey) === null, 'retry removes previous-owner chat');
                    check(await AsyncStorage.getItem(privatePlan) === null, 'retry removes previous-owner plan');
                    check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === null, 'retry removes marker after confirmed cleanup');
                    await AsyncStorage.setItem(phaseKey, '3');
                } else if (phase === 3) {
                    check(await AsyncStorage.getItem(privateKey) === null && await AsyncStorage.getItem(privatePlan) === null, 'clean fixture state survives a second force-stop');
                    check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === null, 'completed marker remains absent after restart');
                    const original = AsyncStorage.multiRemove.bind(AsyncStorage);
                    let registryOperations = 0;
                    AsyncStorage.multiRemove = (...args: Parameters<typeof AsyncStorage.multiRemove>) => { registryOperations++; return original(...args); };
                    try { await ensurePreviousAccountCleanup(); }
                    finally { AsyncStorage.multiRemove = original; }
                    check(registryOperations === 0, 'clean startup gate does not repeat bulk deletion');
                    await AsyncStorage.removeItem(phaseKey);
                    check(await AsyncStorage.getItem(phaseKey) === null, 'audit phase fixture removed');
                } else throw new Error('Unexpected phase');
                check(networkAttempts === 0, 'no fetch requests attempted');
                report.passed = true;
            } catch (error) { report.failure = error instanceof Error ? error.message : 'Unknown failure'; }
            report.networkAttempts = networkAttempts;
            await FileSystem.writeAsStringAsync(FileSystem.documentDirectory + `cleanup-restart-native-phase-${phase}.json`, JSON.stringify(report));
            console.warn('CLEANUP_RESTART_NATIVE_AUDIT ' + JSON.stringify(report));
            setStatus(`Cleanup restart phase ${phase}: ${report.passed ? 'passed' : 'failed'}`);
        })();
    }, []);
    return <Text style={{ padding: 40 }}>{status}</Text>;
}
registerRootComponent(Audit);
