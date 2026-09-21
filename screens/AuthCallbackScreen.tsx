import React, { useEffect } from 'react';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { LinearGradient } from 'expo-linear-gradient';
import { useTranslation } from 'react-i18next';
import useAuthStore from '../store/auth';
import { useAppNavigation } from '../hooks/useAppNavigation';

/**
 * AuthCallbackScreen — deep-link target for OAuth (Google) PKCE redirects.
 *
 * The actual token exchange happens in `useAuthStore.handleDeepLink`
 * (subscribed globally in App.tsx). This screen exists so React Navigation
 * has a route for `auth/callback` instead of warning about an unhandled
 * URL, and to show progress while the exchange completes.
 *
 * Once `isAuthenticated` flips, RootNavigator unmounts this screen
 * automatically and routes to Main.
 */
const AuthCallbackScreen = () => {
  const { t } = useTranslation();
  const navigation = useAppNavigation();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authError = useAuthStore((s) => s.error);
  const loading = useAuthStore((s) => s.loading);

  useEffect(() => {
    if (isAuthenticated) return;
    // If the exchange failed, bounce back to SignIn after a beat so the
    // user sees the error alert there instead of a stuck spinner.
    if (!loading && authError) {
      const timer = setTimeout(() => {
        navigation.reset({ index: 0, routes: [{ name: 'SignIn' }] });
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [isAuthenticated, loading, authError, navigation]);

  return (
    <LinearGradient colors={['#0A0A0A', '#1A1C29', '#16213E']} style={styles.container}>
      <View style={styles.inner}>
        <ActivityIndicator size="large" color="#FFD700" />
        <ScaledText style={styles.title}>
          {authError ?? t('signIn.signingIn', { defaultValue: 'Signing you in…' })}
        </ScaledText>
      </View>
    </LinearGradient>
  );
};

export default AuthCallbackScreen;

const styles = StyleSheet.create({
  container: { flex: 1 },
  inner: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 32, gap: 16 },
  title: { fontSize: 16, fontWeight: '600', color: '#FFF', textAlign: 'center' },
});
