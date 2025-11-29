import React, { useEffect } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

// Импорт существующих экранов
import HomeScreen from "../screens/HomeScreen";
import AIAssistant from "../screens/AIAssistant";
import AddOutfitScreen from "../screens/AddOutfitScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
// 👇 ДОБАВЛЕН ИМПОРТ
import ScanWardrobeScreen from "../screens/ScreenWardrobe";
import SignInScreen from "../screens/SignInScreen";
import SignUpScreen from "../screens/SignUpScreen";
import AIOutfitmaker from "../screens/AIOutfitmaker";
import DesignRoomScreen from "../screens/DesignRoomScreen";
import NewOutfitScreen from "../screens/NewOutfitScreen";
import useAuthStore from "../store/auth";
import { RootStackParamList } from "./types";
import ReviewScreen from "../screens/ReviewScreen";


const Stack = createNativeStackNavigator<any>();

const RootNavigator = () => {
  // Получаем состояние авторизации и функцию инициализации
  // @ts-ignore - игнорируем возможные ошибки типизации Zustand
  const { isAuthenticated, initializeAuth } = useAuthStore();

  useEffect(() => {
    initializeAuth();
  }, [initializeAuth]);

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {isAuthenticated ? (
        // 🔓 Если пользователь вошел: Показываем главные экраны
        <>
          {/* Вместо Tabs пока используем Home, так как TabNavigator еще не создан */}
          <Stack.Screen name="Home" component={HomeScreen} />

          <Stack.Screen
            name="AddOutfit"
            component={AddOutfitScreen}
            options={{ presentation: 'modal', title: "Add New Item" }}
          />

          <Stack.Screen
            name="ReviewScan"
            component={ReviewScreen}
            options={{ headerShown: false }}
          />

          {/* AI Экраны */}
          <Stack.Screen name="AIChat" component={AIAssistant} />
          <Stack.Screen name="AIOutfit" component={AIOutfitmaker} />
          <Stack.Screen name="AITryOn" component={AITryOnScreen} />
          {/* 👇 ДОБАВЛЕН ЭКРАН СКАНИРОВАНИЯ */}
          <Stack.Screen name="ScanWardrobe" component={ScanWardrobeScreen} />

          {/* Design and save screens */}
          <Stack.Screen name="DesignRoom" component={DesignRoomScreen} />
          <Stack.Screen name="NewOutfit" component={NewOutfitScreen} />
        </>
      ) : (
        // 🔒 Если не вошел: Показываем экраны входа
        <>
          <Stack.Screen name="SignIn" component={SignInScreen} />
          <Stack.Screen name="SignUp" component={SignUpScreen} />
        </>
      )}
    </Stack.Navigator>
  );
};

export default RootNavigator;