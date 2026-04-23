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

const useLanguageStore = create<LanguageStore>((set, get) => ({
    currentLanguage: 'en',
    isLoading: false,

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

            set({ currentLanguage: language, isLoading: false });
            log.info('Language changed', { language });
        } catch (error) {
            log.error('Failed to set language', error);
            set({ isLoading: false });
        }
    },

    initializeLanguage: async () => {
        set({ isLoading: true });

        try {
            const savedLanguage = await AsyncStorage.getItem(LANGUAGE_STORAGE_KEY);

            if (savedLanguage && SUPPORTED_LANGUAGES.includes(savedLanguage as Language)) {
                await i18n.changeLanguage(savedLanguage as Language);
                set({ currentLanguage: savedLanguage as Language, isLoading: false });
                log.info('Language restored from storage', { language: savedLanguage });
            } else {
                // Use device locale or default to English
                const deviceLanguage = i18n.language || 'en';
                const supportedLang = SUPPORTED_LANGUAGES.find(lang => 
                    deviceLanguage.startsWith(lang)
                ) || 'en';

                await i18n.changeLanguage(supportedLang);
                set({ currentLanguage: supportedLang, isLoading: false });
                log.info('Language initialized', { language: supportedLang });
            }
        } catch (error) {
            log.error('Failed to initialize language', error);
            set({ currentLanguage: 'en', isLoading: false });
        }
    },
}));

export { SUPPORTED_LANGUAGES, LANGUAGE_NAMES };
export default useLanguageStore;
