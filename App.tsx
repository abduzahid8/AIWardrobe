import { NavigationContainer } from "@react-navigation/native";
import { StatusBar } from "expo-status-bar";
import * as SplashScreen from "expo-splash-screen";
import { useEffect } from "react";
import { StyleSheet, View, Text } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import * as Linking from 'expo-linking';
import "./global.css";
import "./i18n";
import RootNavigator from "./navigation/RootNavigator";
import { navigationRef } from "./navigation/navigationRef";
import { ThemeProvider, useTheme } from "./src/theme/ThemeContext";
import { ErrorBoundary } from "./src/components/ErrorBoundary";
import crashReporting from "./src/services/crashReporting";
import { bootstrapStores } from "./store/bootstrap";
import { validateConfig } from "./src/config/env";
import useAuthStore from "./store/auth";
import useTrackingPermission from "./src/hooks/useTrackingPermission";

// Deep linking configuration
const linking = {
  prefixes: [Linking.createURL('/')],
  config: {
    screens: {
      ResetPassword: 'reset-password',
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
bootstrapStores();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 2,
    },
  },
});

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

  // Hide the splash once the first frame renders. We hide it even on the
  // MissingConfigScreen path so the user can see the actionable error
  // instead of an indefinite splash.
  useEffect(() => {
    SplashScreen.hideAsync().catch(() => undefined);
  }, []);

  useEffect(() => {
    const handleDeepLink = async (event: { url: string }) => {
      const { handleDeepLink: processLink } = useAuthStore.getState();
      await processLink(event.url);
    };

    const subscription = Linking.addEventListener('url', handleDeepLink);

    // Check if app was opened from a link
    Linking.getInitialURL().then((url) => {
      if (url) handleDeepLink({ url });
    });

    return () => {
      subscription.remove();
    };
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
