import React, { useEffect, useState, useRef } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Platform } from "react-native";
import { CommonActions, useNavigation } from "@react-navigation/native";

// Imports screens...
import HomeScreen from "../screens/HomeScreen";
import AIAssistant from "../screens/AIAssistant";
import AIStylistScreen from "../screens/AIStylistScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
import ScanWardrobeScreen from "../screens/ScreenWardrobe";
import SignInScreen from "../screens/SignInScreen";
import SignUpScreen from "../screens/SignUpScreen";
import AIOutfitmaker from "../screens/AIOutfitmaker";
import TabNavigator from "../navigation/TabNavigator";
// import TabNavigator from "../navigation/TabNavigator"; // Ensure correct path
import WardrobeVideoScreen from "../screens/WardrobeVideoScreen";
import CameraScreen from "../screens/CameraScreen";
import ChatScreen from "../screens/ChatScreen";

import useAuthStore from "../store/auth";
import useTrialStore from "../store/trialStore";
import TrialLimitModal from "../components/TrialLimitModal";
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
  console.log('[RootNavigator] Component rendering');
  // Session expiry guard — checks token on app foreground
  useSessionGuard();

  const { isAuthenticated, isTrialMode, startTrial } = useAuthStore();
  const {
    trialCount,
    isTrialExpired,
    initializeTrial,
    incrementTrialCount
  } = useTrialStore();
  const [showTrialModal, setShowTrialModal] = useState(false);
  const [hasIncrementedThisSession, setHasIncrementedThisSession] = useState(false);
  
  console.log('[RootNavigator] Auth state:', { isAuthenticated, isTrialMode, isTrialExpired });

  // Navigation ref for notification-driven navigation
  const navigationRef = React.useRef<any>(null);

  useEffect(() => {
    console.log('[RootNavigator] Initializing app services');
    const initialize = async () => {
      console.log('[RootNavigator] Starting initialization');
      const { initializeAuth } = useAuthStore.getState();
      await initializeAuth();
      console.log('[RootNavigator] Auth initialized');
      await initializeTrial();
      console.log('[RootNavigator] Trial initialized');

      // Initialize services
      await notificationService.initialize();
      console.log('[RootNavigator] Notification service initialized');
      analyticsService.initialize();
      console.log('[RootNavigator] Analytics service initialized');
      await iapService.initialize();
      console.log('[RootNavigator] IAP service initialized');

      // Set analytics user if already authenticated
      const { isAuthenticated: authStatus, user: currentUser } = useAuthStore.getState();
      if (authStatus && currentUser?.id) {
        analyticsService.setUserId(currentUser.id);
        iapService.identify(currentUser.id);
        console.log('[RootNavigator] User identified for analytics and IAP:', currentUser.id);
      }

      const { isTrialExpired: trialExpired } = useTrialStore.getState();
      console.log('[RootNavigator] Trial expired status:', trialExpired);

      if (!authStatus && !trialExpired) {
        // Auto-start trial mode - no need to show auth screens
        console.log('[RootNavigator] Auto-starting trial mode');
        startTrial();
      }
    };

    initialize();

    // Listen for notification taps → navigate to correct screen
    const removeListeners = addNotificationListeners(
      undefined,
      (response) => {
        console.log('[RootNavigator] Notification tapped:', response.notification.request.content.data);
        const screen = response.notification.request.content.data?.screen;
        if (screen && navigationRef.current) {
          console.log('[RootNavigator] Navigating to screen from notification:', screen);
          navigationRef.current.navigate(screen);
        }
      }
    );

    return removeListeners;
  }, []);

  // Increment trial counter on app launch (only once per session, and ONLY for non-authenticated users)
  useEffect(() => {
    console.log('[RootNavigator] Checking trial increment - conditions:', { isAuthenticated, isTrialMode, isTrialExpired, hasIncrementedThisSession });
    if (!isAuthenticated && isTrialMode && !isTrialExpired && !hasIncrementedThisSession) {
      console.log('[RootNavigator] Incrementing trial count');
      incrementTrialCount();
      setHasIncrementedThisSession(true);
    }
  }, [isAuthenticated, isTrialMode, isTrialExpired, hasIncrementedThisSession, incrementTrialCount]);

  // Show modal when trial expires
  useEffect(() => {
    console.log('[RootNavigator] Trial expiry check - isTrialExpired:', isTrialExpired, 'isAuthenticated:', isAuthenticated);
    if (isTrialExpired && !isAuthenticated) {
      console.log('[RootNavigator] Showing trial modal');
      setShowTrialModal(true);
    }
  }, [isTrialExpired, isAuthenticated]);

  const handleNavigateToSignUp = () => {
    console.log('[RootNavigator] Navigate to SignUp pressed');
    setShowTrialModal(false);
    const { endTrial } = useAuthStore.getState();
    endTrial();
  };

  const handleNavigateToSignIn = () => {
    console.log('[RootNavigator] Navigate to SignIn pressed');
    setShowTrialModal(false);
    const { endTrial } = useAuthStore.getState();
    endTrial();
  };


  // Determine what to show based on authentication and trial state
  // Modified flow: Auth -> Main -> (optional) Paywall
  const shouldShowApp = isAuthenticated || (isTrialMode && !isTrialExpired);
  console.log('[RootNavigator] shouldShowApp:', shouldShowApp);

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