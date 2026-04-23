import useLanguageStore from './languageStore';

let initialized = false;

export function bootstrapStores() {
  if (initialized) {
    return;
  }

  initialized = true;

  // Initialize language store
  void useLanguageStore.getState().initializeLanguage();
}
