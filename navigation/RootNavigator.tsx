import React, { useEffect, useState, useRef } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Platform } from "react-native";
import { CommonActions, useNavigation } from "@react-navigation/native";

// Imports screens...
import StyleQuizScreen from "../screens/StyleQuizScreen";
import { useStylePreferenceStore } from "../store/stylePreferenceStore";
import HomeScreen from "../screens/HomeScreen";
import AIAssistant from "../screens/AIAssistant";
import AIStylistScreen from "../screens/AIStylistScreen";
import AddOutfitScreen from "../screens/AddOutfitScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
import ScanWardrobeScreen from "../screens/ScreenWardrobe";
import SignInScreen from "../screens/SignInScreen";
import SignUpScreen from "../screens/SignUpScreen";
import AIOutfitmaker from "../screens/AIOutfitmaker";
import DesignRoomScreen from "../screens/DesignRoomScreen";
import NewOutfitScreen from "../screens/NewOutfitScreen";
import TabNavigator from "../navigation/TabNavigator";
// import TabNavigator from "../navigation/TabNavigator"; // Ensure correct path
import WardrobeVideoScreen from "../screens/WardrobeVideoScreen";
import CameraScreen from "../screens/CameraScreen";

import useAuthStore from "../store/auth";
import useTrialStore from "../store/trialStore";
import TrialLimitModal from "../components/TrialLimitModal";
import ReviewScreen from "../screens/ReviewScreen";
import OutfitCalendarScreen from "../screens/OutfitCalendarScreen";
import AIHubScreen from "../screens/AIHubScreen";
import EmailOnboardingScreen from "../screens/EmailOnboardingScreen";
import TripPlannerScreen from "../screens/TripPlannerScreen";
import OutfitDetailScreen from "../screens/OutfitDetailScreen";
import PaywallScreen from "../screens/PaywallScreen";
import ForgotPasswordScreen from "../screens/ForgotPasswordScreen";
import ResetPasswordScreen from "../screens/ResetPasswordScreen";
import OutfitAIScreen from "../screens/OutfitAIScreen";
import CreateAvatarScreen from "../screens/CreateAvatarScreen";
import MeetingOutfitScreen from "../screens/MeetingOutfitScreen";
import PriceTrackerScreen from "../screens/PriceTrackerScreen";
import FlashSalesScreen from "../screens/FlashSalesScreen";
import FlashSaleEventScreen from "../screens/FlashSaleEventScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import StyleGoalsScreen from "../screens/StyleGoalsScreen";
import DailySuggestionScreen from "../screens/DailySuggestionScreen";
import WearLogScreen from "../screens/WearLogScreen";
import WeeklyInsightsScreen from "../screens/WeeklyInsightsScreen";
import { addNotificationListeners } from "../src/services/notificationService";
import { notificationService } from "../src/services/notificationService";
import { RootStackParamList } from "./types";
import { useSessionGuard } from "../src/hooks/useSessionGuard";
import analyticsService from "../src/services/analyticsService";
import { iapService } from "../src/services/iapService";
import { colors } from "../src/theme";


// 2. Передаем этот список в Stack
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
  // Session expiry guard — checks token on app foreground
  useSessionGuard();

  const { isAuthenticated, isTrialMode, startTrial } = useAuthStore();
  const { hasCompletedOnboarding, onboardingStep } = useStylePreferenceStore();
  const {
    trialCount,
    isTrialExpired,
    initializeTrial,
    incrementTrialCount
  } = useTrialStore();
  const [showTrialModal, setShowTrialModal] = useState(false);
  const [hasIncrementedThisSession, setHasIncrementedThisSession] = useState(false);

  // Navigation ref for notification-driven navigation
  const navigationRef = React.useRef<any>(null);

  useEffect(() => {
    const initialize = async () => {
      const { initializeAuth } = useAuthStore.getState();
      await initializeAuth();
      await initializeTrial();

      // Initialize services
      await notificationService.initialize();
      analyticsService.initialize();
      await iapService.initialize();

      // Set analytics user if already authenticated
      const { isAuthenticated: authStatus, user: currentUser } = useAuthStore.getState();
      if (authStatus && currentUser?.id) {
        analyticsService.setUserId(currentUser.id);
        iapService.identify(currentUser.id);
      }

      const { isTrialExpired: trialExpired } = useTrialStore.getState();

      if (!authStatus && !trialExpired) {
        // Auto-start trial mode - no need to show auth screens
        startTrial();
      }
    };

    initialize();

    // Listen for notification taps → navigate to correct screen
    const removeListeners = addNotificationListeners(
      undefined,
      (response) => {
        const screen = response.notification.request.content.data?.screen;
        if (screen && navigationRef.current) {
          navigationRef.current.navigate(screen);
        }
      }
    );

    return removeListeners;
  }, []);

  // Increment trial counter on app launch (only once per session, and ONLY for non-authenticated users)
  useEffect(() => {
    if (!isAuthenticated && isTrialMode && !isTrialExpired && !hasIncrementedThisSession) {
      incrementTrialCount();
      setHasIncrementedThisSession(true);
    }
  }, [isAuthenticated, isTrialMode, isTrialExpired, hasIncrementedThisSession, incrementTrialCount]);

  // Show modal when trial expires
  useEffect(() => {
    if (isTrialExpired && !isAuthenticated) {
      setShowTrialModal(true);
    }
  }, [isTrialExpired, isAuthenticated]);

  const handleNavigateToSignUp = () => {
    setShowTrialModal(false);
    const { endTrial } = useAuthStore.getState();
    endTrial();
  };

  const handleNavigateToSignIn = () => {
    setShowTrialModal(false);
    const { endTrial } = useAuthStore.getState();
    endTrial();
  };


  // Determine what to show based on authentication and trial state
  // Modified flow: Auth -> Onboarding (StyleQuiz) -> Paywall -> Main
  const shouldShowApp = isAuthenticated || (isTrialMode && !isTrialExpired);

  return (
    <>
      <TrialLimitModal
        visible={showTrialModal}
        onSignUp={handleNavigateToSignUp}
        onSignIn={handleNavigateToSignIn}
      />
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
        {shouldShowApp ? (
          <>
            {!hasCompletedOnboarding ? (
              <Stack.Screen name="StyleQuiz" component={StyleQuizScreen} />
            ) : (
              <Stack.Screen name="Main" component={TabNavigator} />
            )}

            {/* Main tab navigation with Home, Add, and Profile */}
            {/* <Stack.Screen name="Home" component={TabNavigator} /> */}

            <Stack.Screen
              name="AddOutfit"
              component={AddOutfitScreen}
              options={{
                presentation: 'modal',
                animation: 'slide_from_bottom',
                title: "Add New Item",
              }}
            />

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
              name="DesignRoom"
              component={DesignRoomScreen}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="Calendar"
              component={OutfitCalendarScreen}
              options={{ animation: 'slide_from_right' }}
            />
            <Stack.Screen
              name="NewOutfit"
              component={NewOutfitScreen}
              options={{
                animation: 'fade_from_bottom',
                presentation: 'transparentModal',
              }}
            />
            <Stack.Screen
              name="AIHub"
              component={AIHubScreen}
              options={{ animation: 'slide_from_bottom' }}
            />
            <Stack.Screen
              name="EmailOnboarding"
              component={EmailOnboardingScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="TripPlanner"
              component={TripPlannerScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="OutfitDetail"
              component={OutfitDetailScreen}
              options={{
                animation: 'fade',
                presentation: 'fullScreenModal',
                gestureEnabled: true,
              }}
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
              name="MeetingOutfit"
              component={MeetingOutfitScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="PriceTracker"
              component={PriceTrackerScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="FlashSales"
              component={FlashSalesScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="FlashSaleEvent"
              component={FlashSaleEventScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="MyCloset"
              component={MyClosetScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />
            <Stack.Screen
              name="StyleGoals"
              component={StyleGoalsScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />

            {/* ── Core Behavioral Loop (MVP) ── */}
            <Stack.Screen
              name="DailySuggestion"
              component={DailySuggestionScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="WearLog"
              component={WearLogScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="WeeklyInsights"
              component={WeeklyInsightsScreen}
              options={{
                animation: 'slide_from_right',
              }}
            />

            {/* Global Paywall explicitly for when onboarding is done */}
            {hasCompletedOnboarding && (
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
            )}

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
    </>
  );
};

export default RootNavigator;