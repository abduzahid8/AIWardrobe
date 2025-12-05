import React, { useEffect } from "react";
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
import ReviewScreen from "../screens/ReviewScreen";

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
  const { isAuthenticated, initializeAuth } = useAuthStore();

  useEffect(() => {
    initializeAuth();
  }, []);

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
          backgroundColor: '#FDFCF8',
        },
      }}
    >
      {isAuthenticated ? (
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
            name="NewOutfit"
            component={NewOutfitScreen}
            options={{
              animation: 'fade_from_bottom',
              presentation: 'transparentModal',
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
  );
};

export default RootNavigator;