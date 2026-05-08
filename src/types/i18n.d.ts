/**
 * i18n type definitions for strict type checking
 * Ensures all translation keys are validated at compile time
 */

declare module 'react-i18next' {
  interface CustomTypeOptions {
    defaultNS: 'translation';
    resources: {
      translation: {
        // Navigation
        navigation: {
          home: string;
          closet: string;
          ai: string;
          inspo: string;
          profile: string;
        };
        // Home
        home: {
          title: string;
          greeting: string;
          noItems: string;
          addFirstItem: string;
        };
        // Auth
        auth: {
          login: string;
          register: string;
          logout: string;
          email: string;
          password: string;
        };
        // Subscription
        subscription: {
          free: string;
          premium: string;
          vip: string;
          upgrade: string;
          manage: string;
        };
        // Admin
        admin: {
          title: string;
          subtitle: string;
          accessDenied: string;
          adminPrivilegesRequired: string;
          tabs: {
            add: string;
            manage: string;
            inspo: string;
            guide: string;
          };
        };
        // Paywall
        paywall: {
          title: string;
          subtitle: string;
          monthly: string;
          yearly: string;
          restore: string;
          terms: string;
        };
        // Common
        common: {
          cancel: string;
          save: string;
          delete: string;
          edit: string;
          done: string;
          loading: string;
          error: string;
          retry: string;
        };
      };
    };
  }
}

// Strict typing for t() function
export type TranslationKey = 
  | `navigation.${keyof CustomTypeOptions['resources']['translation']['navigation']}`
  | `home.${keyof CustomTypeOptions['resources']['translation']['home']}`
  | `auth.${keyof CustomTypeOptions['resources']['translation']['auth']}`
  | `subscription.${keyof CustomTypeOptions['resources']['translation']['subscription']}`
  | `admin.${keyof CustomTypeOptions['resources']['translation']['admin']}`
  | `admin.tabs.${keyof CustomTypeOptions['resources']['translation']['admin']['tabs']}`
  | `paywall.${keyof CustomTypeOptions['resources']['translation']['paywall']}`
  | `common.${keyof CustomTypeOptions['resources']['translation']['common']}`;
