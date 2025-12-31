import React, { useEffect, useState } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Platform } from "react-native";

// Imports screens...
import HomeScreen from "../screens/HomeScreen";
import AIAssistant from "../screens/AIAssistant";
import AddOutfitScreen from "../screens/AddOutfitScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
import ScanWardrobeScreen from "../screens/ScreenWardrobe";
import SignInScreen from "../screens/SignInScreen";
import SignUpScreen from "../screens/SignUpScreen";
import AIOutfitmaker from "../screens/AIOutfitmaker";
import DesignRoomScreen from "../screens/DesignRoomScreen";
import NewOutfitScreen from "../screens/NewOutfitScreen";
import TabNavigator from "../navigation/TabNavigator";
import WardrobeVideoScreen from "../screens/WardrobeVideoScreen";

import useAuthStore from "../store/auth";
import useTrialStore from "../store/trialStore";
import TrialLimitModal from "../components/TrialLimitModal";
import ReviewScreen from "../screens/ReviewScreen";
import OutfitCalendarScreen from "../screens/OutfitCalendarScreen";
import AIHubScreen from "../screens/AIHubScreen";
import EmailOnboardingScreen from "../screens/EmailOnboardingScreen";
import TripPlannerScreen from "../screens/TripPlannerScreen";
import WardrobeAnalyticsScreen from "../screens/WardrobeAnalyticsScreen";
import OutfitDetailScreen from "../screens/OutfitDetailScreen";
import PaywallScreen from "../screens/PaywallScreen";
import OutfitAIScreen from "../screens/OutfitAIScreen";

export type RootStackParamList = {
  Home: undefined;
  SignIn: undefined;
  SignUp: undefined;
  AddOutfit: undefined;
  AIChat: undefined;
  AIOutfit: undefined;
  AITryOn: undefined;
  ScanWardrobe: undefined;

  // Самое важное: мы указываем, что этот экран ждет массив items!
  ReviewScan: { items: any[] };

  DesignRoom: undefined;
  NewOutfit: undefined;
  WardrobeVideo: undefined;
  Calendar: undefined;
  AIHub: undefined;
  EmailOnboarding: undefined;
  TripPlanner: undefined;
  WardrobeAnalytics: undefined;
  OutfitDetail: { image?: string; outfit?: any };
  Paywall: undefined;
  OutfitAI: undefined;  // 🧠 Multimodal AI Outfit Recommendations
};

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
  // @ts-ignore
  const { isAuthenticated, isTrialMode, startTrial } = useAuthStore();
  const {
    trialCount,
    isTrialExpired,
    initializeTrial,
    incrementTrialCount
  } = useTrialStore();
  const [showTrialModal, setShowTrialModal] = useState(false);
  const [hasIncrementedThisSession, setHasIncrementedThisSession] = useState(false);

  useEffect(() => {
    const initialize = async () => {
      const { initializeAuth } = useAuthStore.getState();
      await initializeAuth();
      await initializeTrial();

      // Automatically start trial mode for unauthenticated users
      const { isAuthenticated: authStatus } = useAuthStore.getState();
      const { isTrialExpired: trialExpired } = useTrialStore.getState();

      if (!authStatus && !trialExpired) {
        // Auto-start trial mode - no need to show auth screens
        startTrial();
      }
    };

    initialize();
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
            backgroundColor: '#FDFCF8',
          },
        }}
      >
        {shouldShowApp ? (
          <>
            {/* Main tab navigation with Home, Add, and Profile */}
            <Stack.Screen name="Home" component={TabNavigator} />

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
              component={AIAssistant}
              options={{ animation: 'slide_from_right' }}
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
              name="WardrobeAnalytics"
              component={WardrobeAnalyticsScreen}
              options={{
                animation: 'slide_from_right',
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
              name="Paywall"
              component={PaywallScreen}
              options={{
                animation: 'slide_from_bottom',
                presentation: 'modal',
                gestureEnabled: true,
                gestureDirection: 'vertical',
              }}
            />
            <Stack.Screen
              name="OutfitAI"
              component={OutfitAIScreen}
              options={{
                animation: 'slide_from_right',
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
          </>
        )}
      </Stack.Navigator>
    </>
  );
};

export default RootNavigator;