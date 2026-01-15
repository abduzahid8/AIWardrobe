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

// Original Screens
import HomeScreen from "../screens/HomeScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import AIHubScreen from "../screens/AIHubScreen";
import InspoScreen from "../screens/InspoScreen";
import ProfileScreen from "../screens/ProfileScreen";

import { colors } from "../src/theme";

const Tab = createBottomTabNavigator();

// Type for animated tab icon props
interface AnimatedTabIconProps {
  focused: boolean;
  iconName: string;
  color: string;
  size: number;
}

// Type for tab bar icon props
interface TabBarIconProps {
  route: { name: string };
  focused: boolean;
  color: string;
  size: number;
}

// Type for tab button props
interface TabButtonProps {
  children: React.ReactNode;
  onPress: () => void;
  accessibilityState?: { selected?: boolean };
}

// Animated tab icon with scale effect
const AnimatedTabIcon = ({ focused, iconName, color, size }: AnimatedTabIconProps) => {
  return (
    <Animated.View
      entering={FadeIn.duration(200)}
      style={{
        transform: [{ scale: focused ? 1.15 : 1 }],
      }}
    >
      <Ionicons name={iconName as any} size={size} color={color} />
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

  const getTabBarIcon = useCallback(({ route, focused, color, size }: TabBarIconProps) => {
    let iconName: string;

    if (route.name === "Home") {
      iconName = focused ? "today" : "today-outline";
    } else if (route.name === "Closet") {
      iconName = focused ? "shirt" : "shirt-outline";
    } else if (route.name === "AI") {
      iconName = focused ? "sparkles" : "sparkles-outline";
    } else if (route.name === "Inspo") {
      iconName = focused ? "compass" : "compass-outline";
    } else if (route.name === "Profile") {
      iconName = focused ? "person" : "person-outline";
    } else {
      iconName = "help-outline";
    }

    return <AnimatedTabIcon focused={focused} iconName={iconName} color={color} size={size} />;
  }, []);

  // Custom tab button with haptic feedback
  const TabButton = useCallback(({ children, onPress, accessibilityState }: TabButtonProps) => {
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
        tabBarButton: (props) => <TabButton {...props as any} />,
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
      <Tab.Screen name="Closet" component={MyClosetScreen} />
      <Tab.Screen name="AI" component={AIHubScreen} />
      <Tab.Screen name="Inspo" component={InspoScreen} />
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