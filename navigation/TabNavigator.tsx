import { StyleSheet, Text, TouchableOpacity, View, Platform } from "react-native";
import React, { useMemo, useCallback } from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";
import { useTranslation } from "react-i18next";
import Animated, {
  FadeIn,
  FadeOut,
  SlideInRight,
  SlideOutLeft,
  SlideInLeft,
  SlideOutRight,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";

// New Imports
import DailyBriefScreen from "../src/features/home/DailyBriefScreen";
import CuratedClosetScreen from "../src/features/closet/CuratedClosetScreen";
import AddOutfitScreen from "../screens/AddOutfitScreen"; // Keeping for now
import DesignRoomScreen from "../screens/DesignRoomScreen"; // Keeping for now
import ProfileScreen from "../screens/ProfileScreen"; // Keeping for now

import { colors } from "../src/theme";

const Tab = createBottomTabNavigator();

// Animated tab icon with scale effect
const AnimatedTabIcon = ({ focused, iconName, color, size }: any) => {
  return (
    <Animated.View
      entering={FadeIn.duration(200)}
      style={{
        transform: [{ scale: focused ? 1.15 : 1 }],
      }}
    >
      <Ionicons name={iconName} size={size} color={color} />
    </Animated.View>
  );
};

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

    return <AnimatedTabIcon focused={focused} iconName={iconName} color={color} size={size} />;
  }, []);

  // Custom tab button with haptic feedback
  const TabButton = useCallback(({ children, onPress, accessibilityState }: any) => {
    const handlePress = () => {
      // Haptic feedback on tab press
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      onPress();
    };

    return (
      <TouchableOpacity
        onPress={handlePress}
        style={styles.tabButton}
        activeOpacity={0.7}
      >
        {children}
      </TouchableOpacity>
    );
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
        tabBarButton: (props) => <TabButton {...props} />,
        // Smooth animations for tab content
        animation: 'fade',
        animationDuration: 250,
        lazy: true,
        // iOS-style smooth tab switching
        ...(Platform.OS === 'ios' && {
          animation: 'shift',
        }),
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

const styles = StyleSheet.create({
  tabButton: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default TabNavigator;