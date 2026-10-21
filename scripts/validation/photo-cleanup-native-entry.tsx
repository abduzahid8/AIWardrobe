/** Explicit audit entry: disposable photo fixtures, real native filesystem/AsyncStorage. */
import React, { useEffect, useState } from 'react';
import { Text, Platform } from 'react-native';
import { registerRootComponent } from 'expo';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';

const phaseKey = 'readiness_photo_cleanup_phase';
let started = false;
function Audit() {
  const [status, setStatus] = useState('Native photo cleanup audit running');
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
        const { clearAllPersistedUserData, ensurePreviousAccountCleanup, ACCOUNT_CLEANUP_PENDING_KEY } = require('../../src/lib/persistence') as typeof import('../../src/lib/persistence');
        const root = FileSystem.documentDirectory + 'aiwardrobe-pending-media/';
        const first = root + '11111111-1111-4111-8111-111111111111/';
        const second = root + '22222222-2222-4222-8222-222222222222/';
        const files = [first + 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa.jpg', second + 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb.png', second + '.orphan.pending'];
        const unrelated = FileSystem.documentDirectory + 'readiness-unrelated-document.txt';
        if (phase === 1) {
          await FileSystem.makeDirectoryAsync(first, { intermediates: true });
          await FileSystem.makeDirectoryAsync(second, { intermediates: true });
          for (const file of files) await FileSystem.writeAsStringAsync(file, 'synthetic-private-fixture');
          await FileSystem.writeAsStringAsync(unrelated, 'keep-unrelated-document');
          // Inject a rejection at the cleanup service boundary. Expo's re-exported
          // filesystem methods are read-only; recovery calls the real filesystem
          // through the unchanged service implementation in a fresh process.
          const media = require('../../src/services/pendingWardrobeMedia') as typeof import('../../src/services/pendingWardrobeMedia');
          const original = media.clearAllPendingWardrobeImages;
          let injected = false, failed = false;
          media.clearAllPendingWardrobeImages = async () => {
            injected = true; throw new Error('Synthetic photo cleanup service rejection');
          };
          try { await clearAllPersistedUserData(); } catch { failed = true; }
          finally { media.clearAllPendingWardrobeImages = original; }
          check(injected && failed, 'photo deletion rejection reaches the account cleanup caller');
          check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === '1', 'photo deletion failure keeps the native cleanup marker');
          for (const file of files) check((await FileSystem.getInfoAsync(file)).exists, 'failed deletion retains fixture ' + file.slice(root.length));
          await AsyncStorage.setItem(phaseKey, '2');
        } else if (phase === 2) {
          check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === '1', 'pending photo cleanup marker survives process restart');
          for (const file of files) check((await FileSystem.getInfoAsync(file)).exists, 'private fixture survives until retry ' + file.slice(root.length));
          await ensurePreviousAccountCleanup();
          check(!(await FileSystem.getInfoAsync(root)).exists, 'real native retry deletes the whole managed photo directory');
          check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === null, 'marker clears after successful native deletion');
          check(await FileSystem.readAsStringAsync(unrelated) === 'keep-unrelated-document', 'unrelated Documents file is preserved');
          await AsyncStorage.setItem(phaseKey, '3');
        } else if (phase === 3) {
          check(!(await FileSystem.getInfoAsync(root)).exists, 'managed photos remain absent after a second process restart');
          check(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY) === null, 'completed cleanup marker remains absent');
          await ensurePreviousAccountCleanup();
          check(!(await FileSystem.getInfoAsync(root)).exists, 'clean startup does not recreate managed photos');
          check(await FileSystem.readAsStringAsync(unrelated) === 'keep-unrelated-document', 'unrelated file survives restart');
          await FileSystem.deleteAsync(unrelated, { idempotent: true });
          await AsyncStorage.removeItem(phaseKey);
        } else throw new Error('Unexpected audit phase');
        check(networkAttempts === 0, 'no fetch requests attempted');
        report.passed = true;
      } catch (error) { report.failure = error instanceof Error ? error.message : 'Unknown failure'; }
      report.networkAttempts = networkAttempts;
      await FileSystem.writeAsStringAsync(FileSystem.documentDirectory + `photo-cleanup-native-phase-${phase}.json`, JSON.stringify(report));
      setStatus(`Photo cleanup phase ${phase}: ${report.passed ? 'passed' : 'failed'}`);
    })();
  }, []);
  return <Text style={{ padding: 40 }}>{status}</Text>;
}
registerRootComponent(Audit);
