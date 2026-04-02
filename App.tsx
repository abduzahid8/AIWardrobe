import { NavigationContainer } from "@react-navigation/native";
import { StatusBar } from "expo-status-bar";
import { StyleSheet } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./global.css";
import "./i18n";
import RootNavigator from "./navigation/RootNavigator";
import { ThemeProvider, useTheme } from "./src/theme/ThemeContext";
import { ErrorBoundary } from "./src/components/ErrorBoundary";
import crashReporting from "./src/services/crashReporting";
import NetworkBanner from "./components/NetworkBanner";

// Initialize crash reporting as early as possible
console.log('[App] Initializing crash reporting');
crashReporting.initialize();

// React Query client — shared across the app
console.log('[App] Creating React Query client');
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 2,
    },
  },
});

// Status bar component that responds to theme
const ThemedStatusBar = () => {
  const { isDark } = useTheme();
  console.log('[App] ThemedStatusBar rendering - isDark:', isDark);
  return <StatusBar style={isDark ? "light" : "dark"} />;
};

// Main app content wrapped in theme
const AppContent = () => {
  const { colors } = useTheme();
  console.log('[App] AppContent rendering - background color:', colors.background);

  return (
    <GestureHandlerRootView style={[styles.container, { backgroundColor: colors.background }]}>
      <SafeAreaProvider>
        <NavigationContainer>
          <ThemedStatusBar />
          <RootNavigator />
        </NavigationContainer>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
};

export default function App() {
  console.log('[App] App component rendering');
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
});
