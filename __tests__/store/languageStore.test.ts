const LANGUAGE_KEY = '@app_language';

/**
 * Isolated setup per test.
 *
 * IMPORTANT: jest.resetModules() gives every require() a FRESH
 * AsyncStorage mock instance (empty storage). The store, i18n, and the
 * test must all share the instance obtained AFTER the reset — the
 * top-level import would point at a stale, disconnected instance.
 */
const setup = async (seed?: string) => {
  jest.resetModules();
  const asyncStorageMod = require('@react-native-async-storage/async-storage');
  // The jest mock has no .default (babel adds interop for `import`
  // in source files) — fall back to the module object itself.
  const AsyncStorage = asyncStorageMod.default ?? asyncStorageMod;
  await AsyncStorage.clear();
  if (seed !== undefined) {
    await AsyncStorage.setItem(LANGUAGE_KEY, seed);
  }
  const useLanguageStore = require('../../store/languageStore').default;
  const i18n = require('../../i18n').default;
  // i18n re-init (fresh require above) performs one detector read —
  // clear counts so assertions only see store behaviour.
  jest.clearAllMocks();
  return { useLanguageStore, AsyncStorage, i18n };
};

beforeEach(() => {
  jest.clearAllMocks();
  jest.resetModules();
});

describe('languageStore — first-run gate', () => {
  it('starts uninitialized with hasSelectedLanguage = false', async () => {
    const { useLanguageStore } = await setup();
    const state = useLanguageStore.getState();
    expect(state.isInitialized).toBe(false);
    expect(state.hasSelectedLanguage).toBe(false);
    expect(state.isLoading).toBe(false);
  });

  it('initializeLanguage with empty storage shows the gate (no selection, initialized)', async () => {
    const { useLanguageStore } = await setup();
    await useLanguageStore.getState().initializeLanguage();

    const state = useLanguageStore.getState();
    expect(state.isInitialized).toBe(true);
    expect(state.hasSelectedLanguage).toBe(false);
    expect(['en', 'ru', 'uz']).toContain(state.currentLanguage);
  });

  it('REGRESSION: first-run init must NOT persist (preview must not skip the gate on relaunch)', async () => {
    const { useLanguageStore, AsyncStorage } = await setup();
    const setItemSpy = jest.spyOn(AsyncStorage, 'setItem');

    await useLanguageStore.getState().initializeLanguage();

    expect(setItemSpy).not.toHaveBeenCalledWith(
      LANGUAGE_KEY,
      expect.anything(),
    );
    expect(await AsyncStorage.getItem(LANGUAGE_KEY)).toBeNull();
  });

  it('REGRESSION: raw i18n.changeLanguage (preview taps) must NOT persist — store is the single writer', async () => {
    const { AsyncStorage, i18n } = await setup();
    const setItemSpy = jest.spyOn(AsyncStorage, 'setItem');

    await i18n.changeLanguage('ru');

    expect(setItemSpy).not.toHaveBeenCalledWith(
      LANGUAGE_KEY,
      expect.anything(),
    );
    expect(await AsyncStorage.getItem(LANGUAGE_KEY)).toBeNull();
  });

  it('setLanguage persists, applies, and completes the gate', async () => {
    const { useLanguageStore, AsyncStorage } = await setup();
    await useLanguageStore.getState().initializeLanguage();

    await useLanguageStore.getState().setLanguage('ru');

    const state = useLanguageStore.getState();
    expect(state.currentLanguage).toBe('ru');
    expect(state.hasSelectedLanguage).toBe(true);
    expect(state.isInitialized).toBe(true);
    expect(await AsyncStorage.getItem(LANGUAGE_KEY)).toBe('ru');
  });

  it('initializeLanguage restores a previously saved choice and skips the gate', async () => {
    const { useLanguageStore } = await setup('uz');
    await useLanguageStore.getState().initializeLanguage();

    const state = useLanguageStore.getState();
    expect(state.currentLanguage).toBe('uz');
    expect(state.hasSelectedLanguage).toBe(true);
    expect(state.isInitialized).toBe(true);
  });

  it('setLanguage ignores unsupported languages without persisting', async () => {
    const { useLanguageStore, AsyncStorage } = await setup();
    await useLanguageStore.getState().initializeLanguage();

    await useLanguageStore.getState().setLanguage('de' as any);

    const state = useLanguageStore.getState();
    expect(state.hasSelectedLanguage).toBe(false);
    expect(await AsyncStorage.getItem(LANGUAGE_KEY)).toBeNull();
  });

  it('concurrent initializeLanguage calls hydrate only once', async () => {
    const { useLanguageStore, AsyncStorage } = await setup();
    const getItemSpy = jest.spyOn(AsyncStorage, 'getItem');

    await Promise.all([
      useLanguageStore.getState().initializeLanguage(),
      useLanguageStore.getState().initializeLanguage(),
    ]);

    const languageReads = getItemSpy.mock.calls.filter(
      ([key]) => key === LANGUAGE_KEY,
    );
    expect(languageReads).toHaveLength(1);
    expect(useLanguageStore.getState().isInitialized).toBe(true);
  });

  it('second initializeLanguage after completion is a no-op', async () => {
    const { useLanguageStore, AsyncStorage } = await setup();
    await useLanguageStore.getState().setLanguage('ru');
    const getItemSpy = jest.spyOn(AsyncStorage, 'getItem');
    getItemSpy.mockClear();

    await useLanguageStore.getState().initializeLanguage();

    expect(getItemSpy).not.toHaveBeenCalledWith(LANGUAGE_KEY);
    expect(useLanguageStore.getState().currentLanguage).toBe('ru');
    expect(useLanguageStore.getState().hasSelectedLanguage).toBe(true);
  });
});
