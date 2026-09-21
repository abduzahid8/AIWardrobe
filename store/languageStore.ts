import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import i18n from '../i18n';
import { createLogger } from '../src/utils/logger';

const log = createLogger('LanguageStore');

const LANGUAGE_STORAGE_KEY = '@app_language';

export type Language = 'en' | 'ru' | 'uz';

export interface LanguageState {
    currentLanguage: Language;
    isLoading: boolean;
    /** True once language hydration from storage has finished. */
    isInitialized: boolean;
    /**
     * True only after the user has explicitly chosen a language
     * (first-run LanguageSelectionScreen or later change in settings).
     * False on fresh install — used by RootNavigator to show the
     * language-only gate before anything else.
     */
    hasSelectedLanguage: boolean;
}

export interface LanguageActions {
    setLanguage: (language: Language) => Promise<void>;
    initializeLanguage: () => Promise<void>;
}

export type LanguageStore = LanguageState & LanguageActions;

const SUPPORTED_LANGUAGES: Language[] = ['en', 'ru', 'uz'];

const LANGUAGE_NAMES: Record<Language, string> = {
    en: 'English',
    ru: 'Русский',
    uz: 'O\'zbek',
};

// Module-level in-flight guard: bootstrapStores() (App module load) and
// RootNavigator both call initializeLanguage(). Without this, two
// concurrent calls both pass the isInitialized check and hydrate twice.
let initializePromise: Promise<void> | null = null;

const useLanguageStore = create<LanguageStore>((set, get) => ({
    currentLanguage: 'en',
    isLoading: false,
    isInitialized: false,
    hasSelectedLanguage: false,

    setLanguage: async (language: Language) => {
        if (!SUPPORTED_LANGUAGES.includes(language)) {
            log.error('Unsupported language', { language });
            return;
        }

        set({ isLoading: true });

        try {
            // Save to AsyncStorage
            await AsyncStorage.setItem(LANGUAGE_STORAGE_KEY, language);

            // Update i18n
            await i18n.changeLanguage(language);

            set({ currentLanguage: language, isLoading: false, hasSelectedLanguage: true, isInitialized: true });
            log.info('Language changed', { language });
        } catch (error) {
            log.error('Failed to set language', error);
            set({ isLoading: false });
        }
    },

    initializeLanguage: async () => {
        // Avoid double-hydration (bootstrapStores + RootNavigator both call this),
        // including the concurrent case where the first call is still in flight.
        if (get().isInitialized) return;
        if (initializePromise) {
            await initializePromise;
            return;
        }

        initializePromise = (async () => {
            set({ isLoading: true });

            try {
                const savedLanguage = await AsyncStorage.getItem(LANGUAGE_STORAGE_KEY);

                if (savedLanguage && SUPPORTED_LANGUAGES.includes(savedLanguage as Language)) {
                    await i18n.changeLanguage(savedLanguage as Language);
                    set({ currentLanguage: savedLanguage as Language, isLoading: false, hasSelectedLanguage: true, isInitialized: true });
                    log.info('Language restored from storage', { language: savedLanguage });
                } else {
                    // First run — no explicit choice yet. Do NOT persist anything:
                    // RootNavigator will show the language-only gate. Use the
                    // device locale (or English fallback) purely as a preview
                    // language so the gate itself is readable.
                    // (Safe: the i18n detector's cacheUserLanguage is a no-op,
                    // so preview changeLanguage() calls never touch storage.)
                    const deviceLanguage = i18n.language || 'en';
                    const supportedLang = SUPPORTED_LANGUAGES.find(lang =>
                        deviceLanguage.startsWith(lang)
                    ) || 'en';

                    await i18n.changeLanguage(supportedLang);
                    set({ currentLanguage: supportedLang, isLoading: false, hasSelectedLanguage: false, isInitialized: true });
                    log.info('Language initialized (no saved choice — awaiting first-run selection)', { language: supportedLang });
                }
            } catch (error) {
                log.error('Failed to initialize language', error);
                set({ currentLanguage: 'en', isLoading: false, hasSelectedLanguage: false, isInitialized: true });
            }
        })();

        await initializePromise;
    },
}));

export { SUPPORTED_LANGUAGES, LANGUAGE_NAMES };
export default useLanguageStore;
