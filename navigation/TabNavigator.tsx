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
import HomeScreen from "../screens/HomeScreen";
import CuratedClosetScreen from "../src/features/closet/CuratedClosetScreen";
import AIHubScreen from "../screens/AIHubScreen";
import DesignRoomScreen from "../screens/DesignRoomScreen";
import ProfileScreen from "../screens/ProfileScreen";

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
      iconName = focused ? "today" : "today-outline";
    } else if (route.name === "Closet") {
      iconName = focused ? "shirt" : "shirt-outline";
    } else if (route.name === "AI") {
      iconName = focused ? "sparkles" : "sparkles-outline";
    } else if (route.name === "Style") {
      iconName = focused ? "color-palette" : "color-palette-outline";
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
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Closet" component={CuratedClosetScreen} />
      <Tab.Screen name="AI" component={AIHubScreen} />
      <Tab.Screen name="Style" component={DesignRoomScreen} />
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