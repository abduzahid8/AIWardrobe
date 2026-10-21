import AsyncStorage from '@react-native-async-storage/async-storage';
import { beginSignOutIntent, completeSignOutIntent, hasPendingSignOutIntent, SIGN_OUT_INTENT_KEY } from '../../src/lib/signOutIntent';

beforeEach(async () => { await completeSignOutIntent(); await AsyncStorage.clear(); });

it('records intent without tokens and a fresh module reads it', async () => {
    await beginSignOutIntent();
    expect(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY)).toBe('1');
    let fresh!: typeof import('../../src/lib/signOutIntent');
    jest.isolateModules(() => { fresh = require('../../src/lib/signOutIntent'); });
    await expect(fresh.hasPendingSignOutIntent()).resolves.toBe(true);
    await completeSignOutIntent();
    await expect(hasPendingSignOutIntent()).resolves.toBe(false);
});

it('keeps an in-process intent if its marker write fails', async () => {
    jest.mocked(AsyncStorage.setItem).mockRejectedValueOnce(new Error('synthetic write failure'));
    await expect(beginSignOutIntent()).rejects.toThrow('synthetic write failure');
    await expect(hasPendingSignOutIntent()).resolves.toBe(true);
});

it('does not interpret a failed read as absence', async () => {
    jest.mocked(AsyncStorage.getItem).mockRejectedValueOnce(new Error('synthetic read failure'));
    await expect(hasPendingSignOutIntent()).rejects.toThrow('synthetic read failure');
});

it('retains intent when removing its marker fails', async () => {
    await beginSignOutIntent();
    jest.mocked(AsyncStorage.removeItem).mockRejectedValueOnce(new Error('synthetic remove failure'));
    await expect(completeSignOutIntent()).rejects.toThrow('synthetic remove failure');
    await expect(hasPendingSignOutIntent()).resolves.toBe(true);
});
