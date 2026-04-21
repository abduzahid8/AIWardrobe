/**
 * Shared navigation ref for imperative navigation outside of screens.
 * Pass this ref to <NavigationContainer ref={navigationRef}>.
 */
import { createNavigationContainerRef } from '@react-navigation/native';
import { RootStackParamList } from './types';

export const navigationRef = createNavigationContainerRef<RootStackParamList>();

export function navigateTo(name: keyof RootStackParamList, params?: object): void {
    if (navigationRef.isReady()) {
        (navigationRef.navigate as any)(name, params);
    }
}
