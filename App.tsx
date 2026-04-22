import { NavigationContainer } from "@react-navigation/native";
import { StatusBar } from "expo-status-bar";
import { StyleSheet, View, Text } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./global.css";
import "./i18n";
import RootNavigator from "./navigation/RootNavigator";
import { navigationRef } from "./navigation/navigationRef";
import { ThemeProvider, useTheme } from "./src/theme/ThemeContext";
import { ErrorBoundary } from "./src/components/ErrorBoundary";
import crashReporting from "./src/services/crashReporting";
import { bootstrapStores } from "./store/bootstrap";
import { validateConfig } from "./src/config/env";

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

  if (missingVars.length > 0) {
    return <MissingConfigScreen missing={missingVars} />;
  }

  return (
    <GestureHandlerRootView style={[styles.container, { backgroundColor: colors.background }]}>
      <SafeAreaProvider>
        <NavigationContainer ref={navigationRef}>
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
