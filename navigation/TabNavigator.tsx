import { StyleSheet, Text, TouchableOpacity, View, Platform } from "react-native";
import React, { useMemo, useCallback } from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";
import { useTranslation } from "react-i18next";

// New Imports
import DailyBriefScreen from "../src/features/home/DailyBriefScreen";
import CuratedClosetScreen from "../src/features/closet/CuratedClosetScreen";
import AddOutfitScreen from "../screens/AddOutfitScreen"; // Keeping for now
import DesignRoomScreen from "../screens/DesignRoomScreen"; // Keeping for now
import ProfileScreen from "../screens/ProfileScreen"; // Keeping for now

import { colors } from "../src/theme";

const Tab = createBottomTabNavigator();

const TabNavigator = () => {
  const { t } = useTranslation();

  const tabBarStyle = useMemo(() => ({
    backgroundColor: colors.background,
    borderTopColor: colors.border,
    height: Platform.OS === "ios" ? 85 : 60,
    paddingTop: 10,
  }), []);

  const getTabBarIcon = useCallback(({ route, focused, color, size }: any) => {
    let iconName: any;

    if (route.name === "Home") {
      iconName = focused ? "newspaper" : "newspaper-outline";
    } else if (route.name === "Discover") {
      iconName = focused ? "shirt" : "shirt-outline";
    } else if (route.name === "AddOutfit") {
      iconName = focused ? "add-circle" : "add-circle-outline";
    } else if (route.name === "DesignRoom") {
      iconName = focused ? "easel" : "easel-outline";
    } else if (route.name === "Profile") {
      iconName = focused ? "person" : "person-outline";
    }

    return <Ionicons name={iconName} size={size} color={color} />;
  }, []);

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarShowLabel: false,
        tabBarStyle,
        tabBarActiveTintColor: colors.text.primary,
        tabBarInactiveTintColor: colors.text.secondary,
        tabBarIcon: (props) => getTabBarIcon({ route, ...props }),
      })}
    >
      <Tab.Screen name="Home" component={DailyBriefScreen} />
      <Tab.Screen name="Discover" component={CuratedClosetScreen} />
      <Tab.Screen name="AddOutfit" component={AddOutfitScreen} />
      <Tab.Screen name="DesignRoom" component={DesignRoomScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
};

export default TabNavigator;