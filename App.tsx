import { NavigationContainer } from "@react-navigation/native";
import { StatusBar } from "expo-status-bar";
import * as SplashScreen from "expo-splash-screen";
import { useEffect, useRef } from "react";
import { AppState, StyleSheet, View, Text } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { QueryClientProvider } from "@tanstack/react-query";
import * as Linking from 'expo-linking';
import "./global.css";
import "./i18n";
import RootNavigator from "./navigation/RootNavigator";
import { navigationRef } from "./navigation/navigationRef";
import { ThemeProvider, useTheme } from "./src/theme/ThemeContext";
import { ErrorBoundary } from "./src/components/ErrorBoundary";
import SplashTransition from "./src/components/SplashTransition";
import crashReporting from "./src/services/crashReporting";
import { flushWardrobePersistence } from "./store/wardrobeStore";
import { bootstrapStores } from "./store/bootstrap";
import { queryClient } from "./src/lib/queryClient";
import { validateConfig } from "./src/config/env";
import useAuthStore from "./store/auth";
import { createAuthLinking } from "./src/lib/authLinking";
import { useSubscriptionClock } from './src/hooks/useSubscriptionClock';
import useTrackingPermission from "./src/hooks/useTrackingPermission";

// Deep linking configuration
const linking = {
  ...createAuthLinking(Linking, url => useAuthStore.getState().handleDeepLink(url)),
  prefixes: [Linking.createURL('/')],
  config: {
    screens: {
      ResetPassword: 'reset-password',
      SignUp: 'sign-up',
      ForgotPassword: 'forgot-password',
      AuthCallback: 'auth/callback',
      // Legal screens are deep-linkable (support flows, App Store review)
      // and it makes them directly e2e-coverable without scrolling.
      PrivacyPolicy: 'privacy',
      TermsOfService: 'terms',
    },
  },
};

// Keep splash visible until the app is actually ready to render.
// Must run before the first React render.
SplashScreen.preventAutoHideAsync().catch(() => {
  // Already prevented or not supported (e.g. web) — safe to ignore.
});

try {
  crashReporting.initialize();
} catch (_e) {
  // Must not crash before React renders
}
// Native screens: use platform UIViewController/Activity views instead of
// JS-composited views for every route (less GPU memory, native transitions).
// NOTE: enableFreeze is intentionally NOT called — freezing suspends blurred
// screens and under rapid tab switches a tab could stay stuck blank white.
// enableScreens alone is safe and keeps the native-view benefits.
try {
  const { enableScreens } = require('react-native-screens');
  enableScreens(true);
} catch (_e) {
  // Web / test env — safe to ignore.
}
bootstrapStores();


const ThemedStatusBar = () => {
  const { isDark } = useTheme();
  return <StatusBar style={isDark ? "light" : "dark"} />;
};

const MissingConfigScreen = ({ missing }: { missing: string[] }) => (
  <View style={styles.configError}>
    <Text style={styles.configErrorTitle}>Configuration Error</Text>
    <Text style={styles.configErrorBody}>
      Missing environment variables:{"\n"}
      {missing.join("\n")}
    </Text>
    <Text style={styles.configErrorHint}>
      Ensure your .env file is present or EAS Secrets are configured.
    </Text>
  </View>
);

const AppContent = () => {
  const { colors } = useTheme();
  const missingVars = validateConfig();

  // Initialize App Tracking Transparency
  useTrackingPermission();
  useSubscriptionClock();

  // Hide the splash once the first frame renders. We hide it even on the
  // MissingConfigScreen path so the user can see the actionable error
  // instead of an indefinite splash.
  useEffect(() => {
    if (missingVars.length > 0) {
      SplashScreen.hideAsync().catch(() => undefined);
    }
  }, [missingVars.length]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', state => {
      if (state !== 'active') void flushWardrobePersistence().catch(() => undefined);
    });
    return () => subscription.remove();
  }, []);

  const updatesChecked = useRef(false);

  useEffect(() => {
    if (updatesChecked.current) return;
    updatesChecked.current = true;
  }, []);

  if (missingVars.length > 0) {
    return <MissingConfigScreen missing={missingVars} />;
  }

  return (
    <GestureHandlerRootView style={[styles.container, { backgroundColor: colors.background }]}>
      <SafeAreaProvider>
        <NavigationContainer ref={navigationRef} linking={linking}>
          <ThemedStatusBar />
          <RootNavigator />
        </NavigationContainer>
        {/* Branded open/close handoff — sits above the navigator, never
            intercepts touches, unmounts itself once invisible. */}
        <SplashTransition />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
};

export default function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <AppContent />
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  configError: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
  },
  configErrorTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: "#B91C1C",
    marginBottom: 16,
  },
  configErrorBody: {
    fontSize: 14,
    color: "#4D4D4D",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 16,
    fontFamily: "monospace",
  },
  configErrorHint: {
    fontSize: 13,
    color: "#808080",
    textAlign: "center",
    lineHeight: 20,
  },
});
