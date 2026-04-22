import React, { useEffect, useState } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Platform } from "react-native";

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

import useAuthStore from "../store/auth";
import useSubscriptionStore from "../store/subscriptionStore";
import useDailyUsageStore from "../store/dailyUsageStore";
import ReviewScreen from "../screens/ReviewScreen";
import OutfitCalendarScreen from "../screens/OutfitCalendarScreen";
import PaywallScreen from "../screens/PaywallScreen";
import ForgotPasswordScreen from "../screens/ForgotPasswordScreen";
import ResetPasswordScreen from "../screens/ResetPasswordScreen";
import OutfitAIScreen from "../screens/OutfitAIScreen";
import CreateAvatarScreen from "../screens/CreateAvatarScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import WardrobeAnalyticsScreen from "../screens/WardrobeAnalyticsScreen";
import PrivacyPolicyScreen from "../screens/PrivacyPolicyScreen";
import TermsOfServiceScreen from "../screens/TermsOfServiceScreen";
import ClothingDetailEditor from "../components/ClothingDetailEditor";
import ClothingDetailScreen from "../screens/ClothingDetailScreen";
import TrialExpiredScreen from "../screens/TrialExpiredScreen";
import { addNotificationListeners } from "../src/services/notificationService";
import { notificationService } from "../src/services/notificationService";
import { RootStackParamList } from "./types";
import { navigationRef, navigateTo } from "./navigationRef";
import { useSessionGuard } from "../src/hooks/useSessionGuard";
import analyticsService from "../src/services/analyticsService";
import { iapService } from "../src/services/iapService";
import { colors } from "../src/theme";


const Stack = createNativeStackNavigator<RootStackParamList>();

// iOS 26-style smooth transition config
const smoothTransitionConfig = {
  animation: 'spring' as const,
  config: {
    stiffness: 1000,
    damping: 500,
    mass: 3,
    overshootClamping: true,
    restDisplacementThreshold: 0.01,
    restSpeedThreshold: 0.01,
  },
};

const RootNavigator = () => {
  useSessionGuard();

  const { isAuthenticated } = useAuthStore();
  const {
    tier,
    hasActiveSubscription,
    isTrialExpired,
    isTrialPending,
    initializeSubscription,
    verifySubscriptionFromServer,
  } = useSubscriptionStore();

  // Show the non-dismissable TrialExpiredScreen when the 7-day trial ends
  // and the user has no paid subscription.
  // IMPORTANT: isTrialPending or isLoading must be false — we never show the gate
  // while the trial date is still being resolved from storage or the server.
  const showTrialGate =
    isAuthenticated &&
    isTrialExpired &&
    !hasActiveSubscription &&
    !isTrialPending &&
    !useSubscriptionStore.getState().isLoading;

  useEffect(() => {
    const initialize = async () => {
      // 1. Auth must happen first — every other service wants to know who
      // the user is before initializing (for user-scoped analytics, IAP, etc.)
      const { initializeAuth } = useAuthStore.getState();
      await initializeAuth();

      // 2. Analytics is synchronous — start it immediately so early events
      // (like "app_opened") land in the queue.
      analyticsService.initialize();

      // 3. Everything else is independent of each other and of the render
      // path. Run them in parallel and never block the UI on them.
      // Individual failures are swallowed so one slow service can't
      // block the others.
      await Promise.all([
        initializeSubscription().catch((err) =>
          console.warn('[RootNavigator] initializeSubscription failed', err),
        ),
        useDailyUsageStore.getState().hydrate().catch((err) =>
          console.warn('[RootNavigator] dailyUsage hydrate failed', err),
        ),
        notificationService.initialize().catch((err) =>
          console.warn('[RootNavigator] notificationService failed', err),
        ),
        iapService.initialize().catch((err) =>
          console.warn('[RootNavigator] iapService failed', err),
        ),
      ]);

      // 4. Identify the user once services are ready. Verification against
      // the server is fire-and-forget — the UI doesn't need to block on it.
      const { isAuthenticated: authStatus, user: currentUser } = useAuthStore.getState();
      if (authStatus && currentUser?.id) {
        analyticsService.setUserId(currentUser.id);
        iapService.identify(currentUser.id);
        verifySubscriptionFromServer().catch((err) =>
          console.warn('[RootNavigator] verifySubscriptionFromServer failed', err),
        );
      }
    };

    initialize();

    const removeListeners = addNotificationListeners(
      undefined,
      (response) => {
        const screen = response.notification.request.content.data?.screen;
        if (screen) {
          navigateTo(screen as keyof RootStackParamList);
        }
      }
    );

    return removeListeners;
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      verifySubscriptionFromServer();
    }
  }, [isAuthenticated]);

  // Auto-navigate to the trial-expired gate when the user has no active
  // subscription and the 7-day trial has ended. isTrialPending guards
  // against flashing the gate before initialization finishes.
  useEffect(() => {
    if (showTrialGate) {
      navigateTo('TrialExpired');
    }
  }, [showTrialGate]);

  return (
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          // iOS 26-style smooth animations
          animation: 'slide_from_right',
          animationDuration: 350,
          gestureEnabled: true,
          gestureDirection: 'horizontal',
          // Smooth spring-based transitions
          ...(Platform.OS === 'ios' && {
            animation: 'default',
            animationTypeForReplace: 'push',
          }),
          // Custom animation
          contentStyle: {
            backgroundColor: colors.background,
          },
        }}
      >
        {isAuthenticated ? (
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
                animation: 'fade_from_bottom',
              }}
            />

            <Stack.Screen
              name="AIChat"
              component={AIStylistScreen}
              options={{ animation: 'slide_from_right' }}
              initialParams={{ initialTab: 'chat' }}
            />
            <Stack.Screen
              name="AIOutfit"
              component={AIOutfitmaker}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="AITryOn"
              component={AITryOnScreen}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="CreateAvatar"
              component={CreateAvatarScreen}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="ScanWardrobe"
              component={ScanWardrobeScreen}
              options={{ animation: 'slide_from_bottom' }}
            />
            <Stack.Screen
              name="WardrobeVideo"
              component={WardrobeVideoScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="Camera"
              component={CameraScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'fullScreenModal',
                headerShown: false,
              }}
            />
            <Stack.Screen
              name="Calendar"
              component={OutfitCalendarScreen}
              options={{ animation: 'slide_from_right' }}
            />

            <Stack.Screen
              name="OutfitAI"
              component={AIStylistScreen}
              options={{
                animation: 'slide_from_right',
              }}
              initialParams={{ initialTab: 'outfit' }}
            />
            <Stack.Screen
              name="MyCloset"
              component={MyClosetScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="WardrobeAnalytics"
              component={WardrobeAnalyticsScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="StylistChat"
              component={ChatScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="PrivacyPolicy"
              component={PrivacyPolicyScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="TermsOfService"
              component={TermsOfServiceScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="ClothingDetailEditor"
              component={ClothingDetailEditor}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'fullScreenModal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="ClothingDetail"
              component={ClothingDetailScreen}
              options={{
                animation: 'slide_from_right',
                gestureEnabled: true,
                gestureDirection: 'horizontal',
              }}
            />

            {/* Global Paywall */}
            <Stack.Screen
              name="Paywall"
              component={PaywallScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />

            {/* Trial Expired (always available in stack for deep links / manual nav) */}
            <Stack.Screen
              name="TrialExpired"
              component={TrialExpiredScreen}
              options={{ animation: 'fade', gestureEnabled: false }}
            />
          </>
        ) : (
          <>
            <Stack.Screen
              name="SignIn"
              component={SignInScreen}
              options={{ animation: 'fade' }}
            />
            <Stack.Screen
              name="SignUp"
              component={SignUpScreen}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="ForgotPassword"
              component={ForgotPasswordScreen}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="ResetPassword"
              component={ResetPasswordScreen}
              options={{ animation: 'slide_from_right' }}
            />
          </>
        )}
      </Stack.Navigator>
  );
};

export default RootNavigator;