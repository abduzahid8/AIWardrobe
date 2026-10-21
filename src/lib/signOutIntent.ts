import AsyncStorage from '@react-native-async-storage/async-storage';

/** Non-sensitive intent; deliberately excluded from ordinary account cleanup. */
export const SIGN_OUT_INTENT_KEY = 'auth_signout_pending_v1';
let pendingInProcess = false;

export async function beginSignOutIntent(): Promise<void> {
    pendingInProcess = true;
    await AsyncStorage.setItem(SIGN_OUT_INTENT_KEY, '1');
}

export async function hasPendingSignOutIntent(): Promise<boolean> {
    const marker = await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY);
    return pendingInProcess || marker !== null;
}

/** Call only after the SDK session and local account cleanup are confirmed. */
export async function completeSignOutIntent(): Promise<void> {
    await AsyncStorage.removeItem(SIGN_OUT_INTENT_KEY);
    pendingInProcess = false;
}
