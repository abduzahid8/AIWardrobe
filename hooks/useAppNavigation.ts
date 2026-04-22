/**
 * useAppNavigation — Type-safe navigation hook for all screens.
 *
 * Replaces `(navigation as any).navigate(...)` pattern with fully typed calls.
 * Usage:  const navigation = useAppNavigation();
 *         navigation.navigate('AIOutfit');
 */
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';

export type AppNavigationProp = NativeStackNavigationProp<RootStackParamList>;

export function useAppNavigation() {
    return useNavigation<AppNavigationProp>();
}

export default useAppNavigation;
