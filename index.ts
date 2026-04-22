// Must run BEFORE any other import so we intercept third-party warnings
// that fire during module initialisation (e.g. react-native-css-interop
// reading the deprecated react-native SafeAreaView getter on import, and
// expo-notifications/expo-media-library posting Expo Go compatibility
// warnings at boot). These originate from transitive dependencies we do
// not own and do NOT affect production behaviour or App Store submission.
const SUPPRESSED_WARNING_PATTERNS = [
    'SafeAreaView has been deprecated',
    'expo-notifications',
    '`expo-notifications`',
    'Expo Go can no longer provide full access to the media library',
    'Android Push notifications (remote notifications) functionality',
];

function shouldSuppress(firstArg: unknown): boolean {
    try {
        const msg = typeof firstArg === 'string' ? firstArg : String(firstArg ?? '');
        return SUPPRESSED_WARNING_PATTERNS.some((pat) => msg.includes(pat));
    } catch {
        return false;
    }
}

const _originalWarn = console.warn;
console.warn = (...args: unknown[]) => {
    if (shouldSuppress(args[0])) return;
    _originalWarn(...(args as []));
};

// Some Expo/RN warnings are routed through console.error or log. Cover both.
const _originalError = console.error;
console.error = (...args: unknown[]) => {
    if (shouldSuppress(args[0])) return;
    _originalError(...(args as []));
};

// LogBox also displays warnings via its own pipeline (yellow-box banners).
// Register a matching ignore list so they do not surface on-device either.
try {
    // Lazy-require to keep this file runnable before react-native is fully
    // initialised in any odd environments.
    const { LogBox } = require('react-native');
    LogBox?.ignoreLogs?.(SUPPRESSED_WARNING_PATTERNS);
} catch {
    // Non-fatal: if LogBox is unavailable for any reason, the console
    // interceptors above still filter the messages.
}

import { registerRootComponent } from 'expo';

import App from './App';

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately
registerRootComponent(App);
