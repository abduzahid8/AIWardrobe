import { StyleSheet, Text, TouchableOpacity, View, Platform } from "react-native";
import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import HomeScreen from "../screens/HomeScreen";
import { Ionicons } from "@expo/vector-icons";
import ProfileScreen from "../screens/ProfileScreen";
import AddOutfitScreen from "../screens/AddOutfitScreen";
import { TabParamList } from "./types";

export const TabNavigator = () => {
  const Tab = createBottomTabNavigator<TabParamList>();

  // ✅ FIX: Define the config HERE (Before the return)
  // Note: Standard Bottom Tabs usually switch instantly and might ignore this config,
  // but this is the correct syntax if you want to define it.
  const transitionConfig = {
    animation: Platform.OS === 'ios' ? 'shift' : 'fade',
    transitionSpec: {
      open: {
        animation: 'spring',
        config: {
          stiffness: 300,
          damping: 30,
          mass: 0.8,
          overshootClamping: false,
        },
      },
      close: {
        animation: 'spring',
        config: {
          stiffness: 300,
          damping: 30,
          mass: 0.8,
          overshootClamping: false,
        },
      },
    },
  };

  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
        tabBarActiveTintColor: "#000",
        tabBarInactiveTintColor: "#D3D3D3",
        tabBarStyle: {
          height: 70,
          paddingBottom: 10,
          paddingTop: 10,
          backgroundColor: "#fff",
          borderTopWidth: 1,
          borderTopColor: "#f0f0f0",
        },
        // To use the config, you would reference the variable here
        // (However, note that BottomTabNavigator does not natively support transitionSpec like StackNavigator does)
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="home" color={color} size={size} />
          ),
        }}
      />

      <Tab.Screen
        name="Add"
        component={AddOutfitScreen}
        options={{
          tabBarIcon: ({ }) => (
            <View className="w-12 h-12 rounded-full bg-black items-center justify-center">
              <Text className="text-white text-[28px] leading-[28px]">+</Text>
            </View>
          ),
          tabBarButton: (props) => {
            const { delayLongPress, ...rest } = props as any;
            return (
              <TouchableOpacity
                {...rest}
                delayLongPress={delayLongPress !== null ? delayLongPress : undefined}
                style={{ alignItems: 'center', justifyContent: 'center' }}
              />
            );
          },
        }}
      />

      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="person" color={color} size={size} />
          ),
        }}
      />
    </Tab.Navigator >
  );
};

const styles = StyleSheet.create({});