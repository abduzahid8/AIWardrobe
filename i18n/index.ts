import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import en from './locales/en.json';
// NOTE: ru/uz are NOT statically imported — that forced ~208KB of JSON
// parsing on every cold start. They load lazily via ensureLocaleBundle()
// (idle preload + on-demand guard in languageStore). en stays eager as the
// default + fallback language.

const LANGUAGE_STORAGE_KEY = '@app_language';

const resources = {
    en: { translation: en },
};

// Language detector plugin
const languageDetector = {
    type: 'languageDetector',
    async: true,
    detect: async (callback: (lang: string) => void) => {
        try {
            const savedLanguage = await AsyncStorage.getItem(LANGUAGE_STORAGE_KEY);
            if (savedLanguage) {
                callback(savedLanguage);
            } else {
                callback('en'); // Default language
            }
        } catch (error) {
            console.error('Error reading language', error);
            callback('en');
        }
    },
    init: () => { },
    // Intentionally a no-op: languageStore is the single writer of
    // '@app_language'. If this cached on every i18n.changeLanguage(),
    // mere previews (first-run gate row taps, init fallbacks) would
    // persist and silently skip the first-run language gate on relaunch.
    cacheUserLanguage: async (_language: string) => { },
};

i18n
    .use(languageDetector as any)
    .use(initReactI18next)
    .init({
        resources,
        fallbackLng: 'en',
        compatibilityJSON: 'v4',
        interpolation: {
            escapeValue: false,
        },
        react: {
            useSuspense: false,
        },
    });

/**
 * Ensure a locale's resource bundle is registered with i18next.
 * No-op for 'en' (bundled at init) and for already-loaded locales.
 * Uses explicit literal require() calls (Metro cannot resolve dynamic
 * paths) and caches by checking hasResourceBundle first.
 */
export function ensureLocaleBundle(lng: string): void {
    if (!lng || lng === 'en' || i18n.hasResourceBundle(lng, 'translation')) return;
    try {
        let data: unknown = null;
        if (lng === 'ru') {
            // eslint-disable-next-line @typescript-eslint/no-var-requires
            const mod = require('./locales/ru.json');
            data = mod?.default ?? mod;
        } else if (lng === 'uz') {
            // eslint-disable-next-line @typescript-eslint/no-var-requires
            const mod = require('./locales/uz.json');
            data = mod?.default ?? mod;
        } else {
            return;
        }
        if (data) i18n.addResourceBundle(lng, 'translation', data, true, true);
    } catch {
        // Missing/corrupt locale — fallbackLng ('en') covers rendering.
    }
}

// Idle-preload the non-default locales off the critical path so a later
// language switch is instant. A switch racing this timer is covered by the
// ensureLocaleBundle() guard at each changeLanguage() call site.
if (typeof setTimeout !== 'undefined') {
    setTimeout(() => {
        try {
            ensureLocaleBundle('ru');
            ensureLocaleBundle('uz');
        } catch {
            // Never block startup on locale preload.
        }
    }, 2000);
}

export default i18n;
