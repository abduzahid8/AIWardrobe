/** Actual App root in a separate, signed-out offline QA app. Never a shipping entry. */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { registerRootComponent } from 'expo';
import * as FileSystem from 'expo-file-system/legacy';

const started = performance.now();
const attempts: Array<{ transport: string; hostClass: string }> = [];
let localAssetRequests = 0;
const disabledServices: string[] = [];
const reportPath = `${FileSystem.documentDirectory}rc-offline-app.json`;
let readyMs: number | null = null;
function deny(transport: string, value?: unknown): never {
  let hostClass = 'unknown';
  try {
    const u = new URL(String(value));
    hostClass = ['localhost', '127.0.0.1', '::1', '[::1]'].includes(u.hostname)
      ? 'loopback'
      : 'external';
  } catch {
    /* Never record URLs, headers, passwords or bodies. */
  }
  attempts.push({ transport, hostClass });
  throw new TypeError('Network request failed');
}
const localAsset = (value: unknown) => /^(file|asset|data):/i.test(String(value));
const originalFetch = globalThis.fetch;
globalThis.fetch = (async (input, init) => {
  const url = typeof input === 'string' ? input : (input as { url?: string })?.url;
  if (localAsset(url)) {
    localAssetRequests++;
    return originalFetch(input, init);
  }
  return deny('fetch', url);
}) as typeof fetch;
const originalOpen = XMLHttpRequest.prototype.open;
XMLHttpRequest.prototype.open = function (_method: string, url: string | URL) {
  if (localAsset(url)) {
    localAssetRequests++;
    return originalOpen.call(this, _method, String(url));
  }
  deny('xhr', url);
};

// Change only this QA process. No env file or service configuration is written.
const config = require('../../src/config/env').Config as {
  supabase: { url: string; anonKey: string };
  api: { url: string };
  revenueCat: { apiKey: string };
  sentry: { dsn: string };
};
config.supabase.url = 'http://127.0.0.1:54321';
config.supabase.anonKey = 'local-public-qa-fixture';
config.api.url = 'http://127.0.0.1:54321';
config.revenueCat.apiKey = 'appl_local_qa_stub';
config.sentry.dsn = '';

const iap = require('../../src/services/iapService').iapService as Record<string, unknown>;
for (const method of ['initialize', 'identify', 'logout'])
  iap[method] = async () => {
    disabledServices.push(`iap.${method}`);
  };
iap.getProducts = async () => [];
for (const method of ['purchase', 'restorePurchases'])
  iap[method] = async () => ({
    success: false,
    error: 'Store actions disabled in local offline QA',
  });
for (const method of ['presentCodeRedemptionSheet', 'manageSubscriptions'])
  iap[method] = async () => {
    throw new Error('Store UI disabled in local offline QA');
  };
iap.verifyServerSubscription = async () => false;
const analytics = require('../../src/services/analyticsService').analyticsService;
void analytics.setEnabled(false);
analytics.initialize = () => {
  disabledServices.push('analytics.initialize');
};
analytics.flush = async () => {};
const notifications = require('../../src/services/notificationService').notificationService;
notifications.initialize = async () => {
  disabledServices.push('notifications.initialize');
};
const prewarm = require('../../src/services/tabPrewarmService');
prewarm.prewarmTabs = async () => {
  disabledServices.push('prewarmTabs');
};
prewarm.prewarmTab = () => {};
const ready = require('../../src/lib/appReady') as typeof import('../../src/lib/appReady');
ready.onAppReady(() => {
  readyMs = performance.now() - started;
});
const App = require('../../App').default as React.ComponentType;
const auth = require('../../store/auth').default as typeof import('../../store/auth').default;
const nav = require('../../navigation/navigationRef')
  .navigationRef as typeof import('../../navigation/navigationRef').navigationRef;
setInterval(() => {
  const state = auth.getState();
  void FileSystem.writeAsStringAsync(
    reportPath,
    JSON.stringify({
      schemaVersion: 1,
      scope:
        'Actual App root, signed out, local offline fixture; SDK/registration/sending/prewarm explicitly disabled',
      releaseBuild: !__DEV__,
      readyMs,
      isInitialized: state.isInitialized,
      authenticated: state.isAuthenticated,
      currentRoute: nav.isReady() ? nav.getCurrentRoute()?.name : null,
      config: {
        backend: 'loopback only',
        billing: 'stubbed; no store validation',
        analyticsSending: false,
        notificationRegistration: false,
        prewarm: false,
      },
      blockedRequests: attempts,
      localAssetRequests,
      disabledServiceCalls: [...new Set(disabledServices)],
    })
  ).catch(() => {});
}, 500);
function OfflineApp() {
  return (
    <View style={styles.root}>
      <App />
      <Text pointerEvents="none" style={styles.label}>
        LOCAL OFFLINE QA · billing/sending disabled
      </Text>
    </View>
  );
}
const styles = StyleSheet.create({
  root: { flex: 1 },
  label: {
    position: 'absolute',
    top: 54,
    alignSelf: 'center',
    fontSize: 10,
    color: '#8b0000',
    backgroundColor: '#fff',
    padding: 2,
  },
});
registerRootComponent(OfflineApp);
