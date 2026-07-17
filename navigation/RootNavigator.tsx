import React, { useEffect, useState, useMemo } from "react";
import * as SplashScreen from "expo-splash-screen";
import { createStackNavigator } from "@react-navigation/stack";

// Imports screens...
import AIStylistScreen from "../screens/AIStylistScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
import ScanWardrobeScreen from "../screens/ScreenWardrobe";
import SignInScreen from "../screens/SignInScreen";
import SignUpScreen from "../screens/SignUpScreen";
import AIOutfitmaker from "../screens/AIOutfitmaker";
import TabNavigator from "../navigation/TabNavigator";
import WardrobeVideoScreen from "../screens/WardrobeVideoScreen";
import CameraScreen from "../screens/CameraScreen";
import ChatScreen from "../screens/ChatScreen";

import useAuthStore, { cleanupAuthSubscription } from "../store/auth";
import useSubscriptionStore from "../store/subscriptionStore";
import useDailyUsageStore from "../store/dailyUsageStore";
import useWardrobeStore from "../store/wardrobeStore";
import ReviewScreen from "../screens/ReviewScreen";
import OutfitCalendarScreen from "../screens/OutfitCalendarScreen";
import PaywallScreen from "../screens/PaywallScreen";
import ForgotPasswordScreen from "../screens/ForgotPasswordScreen";
import ResetPasswordScreen from "../screens/ResetPasswordScreen";
import OutfitAIScreen from "../screens/OutfitAIScreen";
import OutfitInspoScreen from "../screens/OutfitInspoScreen";
import CreateAvatarScreen from "../screens/CreateAvatarScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import WardrobeAnalyticsScreen from "../screens/WardrobeAnalyticsScreen";
import WeeklyInsightsScreen from "../screens/WeeklyInsightsScreen";
import PrivacyPolicyScreen from "../screens/PrivacyPolicyScreen";
import TermsOfServiceScreen from "../screens/TermsOfServiceScreen";
import ClothingDetailEditor from "../components/ClothingDetailEditor";
import ClothingDetailScreen from "../screens/ClothingDetailScreen";
import TrialExpiredScreen from "../screens/TrialExpiredScreen";
import PromoCodeScreen from "../screens/PromoCodeScreen";
import AdminPanelScreen from "../screens/AdminPanelScreen";
import GuideScreen from "../screens/GuideScreen";
import StyleQuizScreen from "../screens/StyleQuizScreen";
import BodyProfileScreen from "../screens/BodyProfileScreen";
import { addNotificationListeners } from "../src/services/notificationService";
import { notificationService } from "../src/services/notificationService";
import { RootStackParamList } from "./types";
import { navigationRef, navigateTo } from "./navigationRef";
import { useSessionGuard } from "../src/hooks/useSessionGuard";
import analyticsService from "../src/services/analyticsService";
import { iapService } from "../src/services/iapService";
import { colors } from "../src/theme";
import { LiquidPresets } from "./liquidTransitions";
import usePromoCodeStore from "../store/promoCodeStore";
import { useStylePreferenceStore } from "../store/stylePreferenceStore";
import { perfMark, perfMeasure } from "../src/utils/perf";


const Stack = createStackNavigator<RootStackParamList>();

const RootNavigator = () => {
  useSessionGuard();

  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const isInitialized = useAuthStore((s) => s.isInitialized);
  const [isAppReady, setIsAppReady] = useState(false);
  const hasActiveSubscription = useSubscriptionStore((s) => s.hasActiveSubscription);
  const isTrialExpired = useSubscriptionStore((s) => s.isTrialExpired);
  const isTrialPending = useSubscriptionStore((s) => s.isTrialPending);
  const isSubscriptionLoading = useSubscriptionStore((s) => s.isLoading);
  const initializeSubscription = useSubscriptionStore((s) => s.initializeSubscription);
  const verifySubscriptionFromServer = useSubscriptionStore((s) => s.verifySubscriptionFromServer);
  const hydratePromoCodeStore = usePromoCodeStore((s) => s.hydrate);
  const hasCompletedOnboarding = useStylePreferenceStore((s) => s.hasCompletedOnboarding);
  const onboardingStep = useStylePreferenceStore((s) => s.onboardingStep);

  // Memoize gate booleans so RootNavigator only re-renders when the values
  // that actually affect routing change. Without useMemo, every subscription
  // store update (e.g. isLoading flipping true→false during initializeSubscription)
  // caused the entire conditional Stack.Navigator children tree to be re-evaluated,
  // which React Navigation interpreted as a structural change and unmounted
  // TabNavigator — destroying HomeScreen and all other tab screens.
  const showTrialGate = useMemo(
    () =>
      isAuthenticated &&
      isTrialExpired &&
      !hasActiveSubscription &&
      !isTrialPending &&
      !isSubscriptionLoading,
    [isAuthenticated, isTrialExpired, hasActiveSubscription, isTrialPending, isSubscriptionLoading],
  );

  // NOTE: showPromoGate is intentionally disabled (always false).
  // Apple Guideline 2.1(a) requires that users can access the free tier
  // without being forced through a promo code or paywall. The PromoCode
  // screen remains accessible voluntarily (e.g. from Profile screen).
  const showPromoGate = false;
  const showOnboardingGate = useMemo(
    () => isAuthenticated && !hasCompletedOnboarding && onboardingStep < 6,
    [isAuthenticated, hasCompletedOnboarding, onboardingStep],
  );
  const showOnboardingPaywall = useMemo(
    () => isAuthenticated && !hasCompletedOnboarding && onboardingStep >= 6,
    [isAuthenticated, hasCompletedOnboarding, onboardingStep],
  );

  useEffect(() => {
    const initialize = async () => {
      try {
      perfMark('app:init');
      // 1. Auth must happen first — every other service wants to know who
      // the user is before initializing (for user-scoped analytics, IAP, etc.)
      // Hard 5-second timeout so the app never stays frozen on the splash if
      // Supabase's getSession() hangs (e.g. first cold-start on slow networks).
      const { initializeAuth } = useAuthStore.getState();
      await Promise.race([
        initializeAuth(),
        new Promise<void>((_, reject) =>
          setTimeout(() => reject(new Error('initializeAuth timed out after 5000ms')), 5000)
        ),
      ]).catch((err) => {
        console.warn('[RootNavigator] initializeAuth timeout or error — forcing isInitialized', err);
        // Ensure isInitialized is always set so the app can render
        useAuthStore.setState({ isInitialized: true, loading: false });
      });
      perfMeasure('app:init');
      console.log('[PERF] 🔐 Auth initialized');

      // 2. Analytics is synchronous — start it immediately so early events
      // (like "app_opened") land in the queue.
      analyticsService.initialize();

      // 3. Everything else is independent of each other and of the render
      // path. Run them in parallel and never block the UI on them.
      // Individual failures are swallowed so one slow service can't
      // block the others.
      //
      // rehydrateFromCloud is included here so wardrobe data fetching
      // starts immediately after auth resolves, in parallel with the other
      // services, instead of waiting for all five to finish first
      // (fixes Defect 1.3 — was previously called sequentially after this block).
      const { isAuthenticated: authStatus, user: currentUser } = useAuthStore.getState();

      const rehydrateStart = Date.now();
      // Start background services immediately in parallel — do NOT block the UI or splash screen on them.
      // Push notifications setup (triggers native dialogs), RevenueCat configuration, and wardrobe cloud sync
      // will run and resolve in the background while the UI loads.
      notificationService.initialize().catch((err) =>
        console.warn('[RootNavigator] notificationService failed', err),
      );
      iapService.initialize().catch((err) =>
        console.warn('[RootNavigator] iapService failed', err),
      );
      if (authStatus && currentUser?.id) {
        useWardrobeStore.getState().rehydrateFromCloud(false)
          .then(() => {
            const elapsed = Date.now() - rehydrateStart;
            console.log(`[PERF] 🟢 rehydrateFromCloud completed in ${elapsed}ms`);
          })
          .catch((err) => {
            const elapsed = Date.now() - rehydrateStart;
            console.warn(`[PERF] 🔴 rehydrateFromCloud FAILED after ${elapsed}ms`, err);
          });
      }

      // We only await fast local AsyncStorage hydrations (daily usage, subscription status, promo codes).
      // This is extremely fast (typically < 50ms) and prevents any native app launch hangs.
      const INIT_TIMEOUT_MS = 1500;
      const timeoutPromise = new Promise<void>((_, reject) =>
        setTimeout(() => reject(new Error(`App initialization timed out after ${INIT_TIMEOUT_MS}ms`)), INIT_TIMEOUT_MS)
      );

      await Promise.race([
        Promise.all([
          initializeSubscription().catch((err) =>
            console.warn('[RootNavigator] initializeSubscription failed', err),
          ),
          useDailyUsageStore.getState().hydrate().catch((err) =>
            console.warn('[RootNavigator] dailyUsage hydrate failed', err),
          ),
          hydratePromoCodeStore().catch((err) =>
            console.warn('[RootNavigator] promoCode hydrate failed', err),
          ),
        ]),
        timeoutPromise,
      ]).catch((err) => {
        console.error('[RootNavigator] Initialization timeout or error — forcing app ready', err);
      });

      // 4. Identify the user once services are ready. Verification against
      // the server is fire-and-forget — the UI doesn't need to block on it.
      if (authStatus && currentUser?.id) {
        analyticsService.setUserId(currentUser.id);
        iapService.identify(currentUser.id).catch(() => {});
        verifySubscriptionFromServer().catch((err) =>
          console.warn('[RootNavigator] verifySubscriptionFromServer failed', err),
        );
      }

      } catch (fatalErr) {
        // If auth or any unguarded step throws, we still need to let the
        // app render so the user sees something instead of a permanent
        // splash screen (which iOS eventually kills → "crash on launch").
        console.error('[RootNavigator] Fatal init error — forcing app ready', fatalErr);
      }

      // 5. Mark app as fully ready — splash screen can now be hidden safely.
      // This MUST always run, even if init failed, so the user isn't stuck
      // on a blank splash that iOS terminates.
      setIsAppReady(true);
      console.log('[PERF] ✅ App ready — all services initialized');
    };

    initialize();

    let removeListeners = () => {};
    try {
      removeListeners = addNotificationListeners(
        undefined,
        (response) => {
          const screen = response.notification.request.content.data?.screen;
          if (screen) {
            navigateTo(screen as keyof RootStackParamList);
          }
        }
      );
    } catch (e) {
      console.warn('[RootNavigator] Failed to add notification listeners', e);
    }

    return () => {
      try {
        removeListeners();
      } catch (e) {
        console.warn('[RootNavigator] Failed to remove notification listeners', e);
      }
      cleanupAuthSubscription();
    };
  }, []);

  // Gates are now rendered as the initial route (see stack below) instead
  // of being pushed via navigateTo side-effects. Declarative routing is
  // race-free: when `showPromoGate` / `showTrialGate` flip back to false,
  // React Navigation unmounts the gate automatically.

  useEffect(() => {
    if (isAppReady && isInitialized) {
      // Hide the splash screen only after ALL services are initialized and the app is ready to render.
      // Defer slightly to ensure the Navigator has mounted and rendered its first frame.
      const timer = setTimeout(() => {
        SplashScreen.hideAsync().catch(() => undefined);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isAppReady, isInitialized]);

  if (!isInitialized || !isAppReady) {
    // Keep the splash screen visible and render nothing while initializing.
    return null;
  }

  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
        // Liquid transition as default
        ...LiquidPresets.slide,
        cardStyle: {
          // Always use the light background — all screens in this app use the
          // light design system. Using `colors.background` caused a navy blue
          // (#0A1931) flash between tab switches when the device is in dark mode
          // because `colors` is resolved at module load time from the system
          // color scheme.
          backgroundColor: '#FFFFFF',
        },
      }}
    >
      {isAuthenticated ? (
        showTrialGate ? (
          <>
            <Stack.Screen
              name="TrialExpired"
              component={TrialExpiredScreen}
              options={{ ...LiquidPresets.fade }}
            />
            <Stack.Screen
              name="Paywall"
              component={PaywallScreen}
              options={{ ...LiquidPresets.fade }}
            />
          </>
        ) : showOnboardingGate ? (
          <Stack.Screen
            name="StyleQuiz"
            component={StyleQuizScreen}
            options={{ ...LiquidPresets.fade, gestureEnabled: false }}
          />
        ) : showOnboardingPaywall ? (
          <Stack.Screen
            name="Paywall"
            component={PaywallScreen}
            options={{
              ...LiquidPresets.fade,
              gestureEnabled: false,
            }}
          />
        ) : (
          <>
            <Stack.Screen name="Main" component={TabNavigator} />

            {/* Main tab navigation with Home, Add, and Profile */}
            {/* <Stack.Screen name="Home" component={TabNavigator} /> */}

            {/* Important: name should match ParamList */}
            <Stack.Screen
              name="ReviewScan"          // Route name (for navigation.navigate)
              component={ReviewScreen}   // Component itself (from file)
              options={{
                headerShown: false,
                ...LiquidPresets.rise,
              }}
            />

            <Stack.Screen
              name="AIChat"
              component={AIStylistScreen}
              options={{ ...LiquidPresets.rise }}
              initialParams={{ initialTab: 'chat' }}
            />
            <Stack.Screen
              name="AIOutfit"
              component={AIOutfitmaker}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="AITryOn"
              component={AITryOnScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="OutfitInspo"
              component={OutfitInspoScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="CreateAvatar"
              component={CreateAvatarScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="ScanWardrobe"
              component={ScanWardrobeScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="WardrobeVideo"
              component={WardrobeVideoScreen}
              options={{
                ...LiquidPresets.rise,
              }}
            />
            <Stack.Screen
              name="Camera"
              component={CameraScreen}
              options={{
                ...LiquidPresets.rise,
                headerShown: false,
              }}
            />
            <Stack.Screen
              name="Calendar"
              component={OutfitCalendarScreen}
              options={{ ...LiquidPresets.slide }}
            />

            <Stack.Screen
              name="OutfitAI"
              component={AIStylistScreen}
              options={{
                ...LiquidPresets.rise,
              }}
              initialParams={{ initialTab: 'outfit' }}
            />
            <Stack.Screen
              name="MyCloset"
              component={MyClosetScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="WardrobeAnalytics"
              component={WardrobeAnalyticsScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="WeeklyInsights"
              component={WeeklyInsightsScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="StylistChat"
              component={ChatScreen}
              options={{
                ...LiquidPresets.rise,
              }}
            />
            <Stack.Screen
              name="PrivacyPolicy"
              component={PrivacyPolicyScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="TermsOfService"
              component={TermsOfServiceScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="ClothingDetailEditor"
              component={ClothingDetailEditor}
              options={{
                ...LiquidPresets.rise,
              }}
            />
            <Stack.Screen
              name="ClothingDetail"
              component={ClothingDetailScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />

            {/* Global Paywall */}
            <Stack.Screen
              name="Paywall"
              component={PaywallScreen}
              options={{
                ...LiquidPresets.fade,
                presentation: 'card',
                gestureEnabled: false,
              }}
            />

            {/* Trial Expired (always available in stack for deep links / manual nav) */}
            <Stack.Screen
              name="TrialExpired"
              component={TrialExpiredScreen}
              options={{ ...LiquidPresets.fade }}
            />
            <Stack.Screen
              name="PromoCode"
              component={PromoCodeScreen}
              options={{ ...LiquidPresets.fade, gestureEnabled: false }}
            />

            {/* Admin Panel — shop catalog management for admin users */}
            <Stack.Screen
              name="AdminPanel"
              component={AdminPanelScreen}
              options={{ ...LiquidPresets.rise }}
            />

            {/* Guide — editable onboarding/guide page */}
            <Stack.Screen
              name="Guide"
              component={GuideScreen}
              options={{ ...LiquidPresets.slide }}
            />
          </>
        )
        ) : (
        <>
          <Stack.Screen
            name="SignIn"
            component={SignInScreen}
            options={{ ...LiquidPresets.fade }}
          />
          <Stack.Screen
            name="SignUp"
            component={SignUpScreen}
            options={{ ...LiquidPresets.slide }}
          />
          <Stack.Screen
            name="ForgotPassword"
            component={ForgotPasswordScreen}
            options={{ ...LiquidPresets.slide }}
          />
        </>
      )}

      {/*
        ResetPassword must be registered outside the auth ternary so it survives
        the isAuthenticated flip triggered by the deep-link session handshake.
        Without this, the screen unmounts before the user can set a new password.
      */}
      <Stack.Screen
        name="ResetPassword"
        component={ResetPasswordScreen}
        options={{ ...LiquidPresets.slide }}
      />
    </Stack.Navigator>
  );
};

export default RootNavigator;
