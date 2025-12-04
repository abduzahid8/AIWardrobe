import React, { useEffect } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

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

const RootNavigator = () => {
  // @ts-ignore
  const { isAuthenticated, initializeAuth } = useAuthStore();

  useEffect(() => {
    initializeAuth();
  }, []);

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {isAuthenticated ? (
        <>
          {/* Main tab navigation with Home, Add, and Profile */}
          <Stack.Screen name="Home" component={TabNavigator} />

          <Stack.Screen
            name="AddOutfit"
            component={AddOutfitScreen}
            options={{ presentation: 'modal', title: "Add New Item" }}
          />

          {/* Important: name should match ParamList */}
          <Stack.Screen
            name="ReviewScan"          // Route name (for navigation.navigate)
            component={ReviewScreen}   // Component itself (from file)
            options={{ headerShown: false }}
          />

          <Stack.Screen name="AIChat" component={AIAssistant} />
          <Stack.Screen name="AIOutfit" component={AIOutfitmaker} />
          <Stack.Screen name="AITryOn" component={AITryOnScreen} />
          <Stack.Screen name="ScanWardrobe" component={ScanWardrobeScreen} />
          <Stack.Screen name="WardrobeVideo" component={WardrobeVideoScreen} />

          <Stack.Screen name="DesignRoom" component={DesignRoomScreen} />
          <Stack.Screen name="NewOutfit" component={NewOutfitScreen} />
        </>
      ) : (
        <>
          <Stack.Screen name="SignIn" component={SignInScreen} />
          <Stack.Screen name="SignUp" component={SignUpScreen} />
        </>
      )}
    </Stack.Navigator>
  );
};

export default RootNavigator;