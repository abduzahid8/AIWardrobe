import { LayoutChangeEvent, Platform, StyleSheet, TouchableOpacity, View, useWindowDimensions } from "react-native";
import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";
import { useTranslation } from "react-i18next";
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withSequence,
  interpolate,
  useDerivedValue,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";

// Original Screens
import HomeScreen from "../screens/HomeScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
import InspoScreen from "../screens/InspoScreen";
import ProfileScreen from "../screens/ProfileScreen";

// Screen-level ErrorBoundary — isolates crashes to individual tabs
import { ErrorBoundary } from "../src/components/ErrorBoundary";
import { createLogger } from "../src/utils/logger";

const logger = createLogger('TabNavigator');

/** Wraps a screen component in its own ErrorBoundary */
const withErrorBoundary = (Screen: React.ComponentType<any>, name: string) => {
  const Wrapped = (props: any) => (
    <ErrorBoundary>
      <Screen {...props} />
    </ErrorBoundary>
  );
  Wrapped.displayName = `ErrorBoundary(${name})`;
  return Wrapped;
};

const SafeHomeScreen = withErrorBoundary(HomeScreen, 'Home');
const SafeClosetScreen = withErrorBoundary(MyClosetScreen, 'Closet');
const SafeAITryOnScreen = withErrorBoundary(AITryOnScreen, 'AI');
const SafeInspoScreen = withErrorBoundary(InspoScreen, 'Inspo');
const SafeProfileScreen = withErrorBoundary(ProfileScreen, 'Profile');

// 2026 Design System
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { colors, spacing } = LiquidGlass2026Theme;
const TAB_BAR_HORIZONTAL_MARGIN = 20;
const TAB_BAR_HORIZONTAL_PADDING = spacing.xs;

const Tab = createBottomTabNavigator();

// Type for animated tab icon props
interface AnimatedTabIconProps {
  focused: boolean;
  iconName: string;
  color: string;
  size: number;
  label: string;
}

// Helper for icon names
const getIconName = (routeName: string, focused: boolean) => {
  if (routeName === "Home") return focused ? "today" : "today-outline";
  if (routeName === "Closet") return focused ? "shirt" : "shirt-outline";
  if (routeName === "AI") return focused ? "sparkles" : "sparkles";
  if (routeName === "Inspo") return focused ? "compass" : "compass-outline";
  if (routeName === "Profile") return focused ? "person" : "person-outline";
  return "help-outline";
};

// Animated tab item with parallax icon/text movement
const AnimatedTabItem = ({ focused, iconName, color, size, label }: AnimatedTabIconProps) => {
  // Drive local animations based on focused state
  const progress = useDerivedValue(() => {
    return withTiming(focused ? 1 : 0, { duration: 250 });
  });

  const rotation = useSharedValue(0);

  React.useEffect(() => {
    if (focused) {
      // Trigger a playful "wiggle" or "shake" when clicked
      rotation.value = withSequence(
        withTiming(-15, { duration: 50 }),
        withTiming(15, { duration: 50 }),
        withTiming(-10, { duration: 50 }),
        withTiming(10, { duration: 50 }),
        withTiming(0, { duration: 50 })
      );
    }
  }, [focused]);

  const iconStyle = useAnimatedStyle(() => ({
    // Icon stays in place but wobbles when clicked
    transform: [
      { rotate: `${rotation.value}deg` },
    ],
  }));

  const labelStyle = useAnimatedStyle(() => ({
    opacity: progress.value,
    transform: [
      { translateY: interpolate(progress.value, [0, 1], [8, 0]) },
      { scale: progress.value }
    ],
  }));

  return (
    <View style={styles.tabItemContainer}>
      <Animated.View style={iconStyle}>
        <Ionicons name={iconName as any} size={28} color={color} />
      </Animated.View>
      {/* Optional: Add label if desired, currently hidden to match "minimalist" request, but code supports it */}
      {/* <Animated.Text style={[styles.tabLabel, { color }, labelStyle]}>
        {label}
      </Animated.Text> */}
    </View>
  );
};

// Liquid Parallax Tab Bar
const LiquidParallaxTabBar = ({ state, descriptors, navigation }: any) => {
  logger.debug('LiquidParallaxTabBar rendering', { tabIndex: state.index });
  const { width } = useWindowDimensions();
  const fallbackTabBarWidth = Math.max(width - (TAB_BAR_HORIZONTAL_MARGIN * 2), 0);
  const [tabBarWidth, setTabBarWidth] = React.useState(fallbackTabBarWidth);
  const tabWidth =
    Math.max(tabBarWidth - (TAB_BAR_HORIZONTAL_PADDING * 2), 0) / Math.max(state.routes.length, 1);

  // Shared value for the sliding indicator
  const translateX = useSharedValue(state.index * tabWidth);

  React.useEffect(() => {
    setTabBarWidth(fallbackTabBarWidth);
  }, [fallbackTabBarWidth]);

  const handleTabBarLayout = React.useCallback((event: LayoutChangeEvent) => {
    const nextWidth = event.nativeEvent.layout.width;
    if (nextWidth > 0) {
      setTabBarWidth(nextWidth);
    }
  }, []);

  // Update position when index changes
  React.useEffect(() => {
    translateX.value = withSpring(state.index * tabWidth, {
      damping: 12, // Lower damping for more bounce (15 -> 12)
      stiffness: 150, // Higher stiffness for faster snap (120 -> 150)
      mass: 1, // Heavier mass for momentum (0.8 -> 1)
    });
  }, [state.index, tabWidth, translateX]);

  const animatedIndicatorStyle = useAnimatedStyle(() => ({
    // Indicator needs to start from 0 relative to the container
    transform: [{ translateX: translateX.value }],
    width: tabWidth,
  }));

  return (
    <BlurView
      intensity={Platform.OS === 'ios' ? 80 : 100}
      tint="light"
      style={styles.tabBarContainer}
      onLayout={handleTabBarLayout}
    >
      <LinearGradient
        colors={['rgba(255,255,255,0.94)', 'rgba(240,246,255,0.88)']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.tabBarGradient}
      />
      {/* Glass overlay for extra depth */}
      <View style={styles.glassOverlay} />

      {/* Liquid Blob Indicator - Background Layer */}
      <Animated.View style={[styles.indicatorContainer, animatedIndicatorStyle]}>
        <View style={styles.liquidBlob} />
      </Animated.View>

      {/* Tabs - Foreground Layer */}
      <View style={styles.tabBarContent}>
        {state.routes.map((route: any, index: number) => {
          const { options } = descriptors[route.key];
          const isFocused = state.index === index;

          // Use 2026 theme colors
          const activeColor = colors.text.primary;
          const inactiveColor = colors.text.tertiary;
          const iconColor = isFocused ? activeColor : inactiveColor;

          const onPress = () => {
            logger.debug('Tab pressed', { name: route.name, isFocused });
            const event = navigation.emit({
              type: 'tabPress',
              target: route.key,
              canPreventDefault: true,
            });

            if (!isFocused && !event.defaultPrevented) {
              logger.debug('Navigating to tab', route.name);
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              navigation.navigate(route.name);
            } else {
              logger.debug('Tab already focused or event prevented');
            }
          };

          return (
            <TouchableOpacity
              key={route.key}
              accessibilityRole="button"
              accessibilityState={isFocused ? { selected: true } : {}}
              accessibilityLabel={options.tabBarAccessibilityLabel}
              testID={options.tabBarTestID}
              onPress={onPress}
              style={styles.tabButton}
              activeOpacity={1}
            >
              <AnimatedTabItem
                focused={isFocused}
                iconName={getIconName(route.name, isFocused)}
                color={iconColor}
                size={24}
                label={route.name}
              />
            </TouchableOpacity>
          );
        })}
      </View>
    </BlurView>
  );
};

const TabNavigator = () => {
  logger.debug('TabNavigator component rendering');
  const { t } = useTranslation();

  return (
    <Tab.Navigator
      tabBar={(props) => <LiquidParallaxTabBar {...props} />}
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarShowLabel: false,
        // Using 'shift' animation for that "Parallax" feel on iOS page transitions
        animation: Platform.OS === 'ios' ? 'shift' : 'fade',
        lazy: true,
      })}
    >
      <Tab.Screen
        name="Home"
        component={SafeHomeScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.home', 'Home') }}
      />
      <Tab.Screen
        name="Closet"
        component={SafeClosetScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.closet', 'My Closet') }}
      />
      {/* AI tab: Try On — full-length photo + AI outfit preview */}
      <Tab.Screen
        name="AI"
        component={SafeAITryOnScreen}
        initialParams={{ asTab: true }}
        options={{ tabBarAccessibilityLabel: t('tabs.ai', 'AI Try On') }}
      />
      <Tab.Screen
        name="Inspo"
        component={SafeInspoScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.inspo', 'Inspiration') }}
      />
      <Tab.Screen
        name="Profile"
        component={SafeProfileScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.profile', 'Profile') }}
      />
    </Tab.Navigator>
  );
};

const styles = StyleSheet.create({
  tabBarContainer: {
    position: 'absolute',
    bottom: 30,
    left: TAB_BAR_HORIZONTAL_MARGIN,
    right: TAB_BAR_HORIZONTAL_MARGIN,
    height: 72,
    borderRadius: 36,
    overflow: 'hidden',
    shadowColor: "#173A65",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.12,
    shadowRadius: 22,
    elevation: 12,
  },
  tabBarGradient: {
    ...StyleSheet.absoluteFillObject,
  },
  glassOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255,255,255,0.42)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    borderRadius: 36,
  },
  tabBarContent: {
    flexDirection: 'row',
    height: '100%',
    zIndex: 2,
    alignItems: 'center',
    paddingHorizontal: TAB_BAR_HORIZONTAL_PADDING,
  },
  tabButton: {
    flex: 1,
    height: 54,
    justifyContent: 'center',
    alignItems: 'center',
    // Ensure touch target meets WCAG 3.0 minimum
    minWidth: spacing.touchTarget.minimum,
  },
  tabItemContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabLabel: {
    fontSize: 10,
    fontWeight: '600',
    marginTop: 4,
  },
  indicatorContainer: {
    position: 'absolute',
    left: TAB_BAR_HORIZONTAL_PADDING,
    height: '100%',
    justifyContent: 'flex-end',
    alignItems: 'center',
    zIndex: 1,
    paddingBottom: 7,
  },
  liquidBlob: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    shadowColor: '#173A65',
    shadowOpacity: 0.14,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
});

export default TabNavigator;