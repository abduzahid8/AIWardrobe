import { LayoutChangeEvent, Platform, Pressable, StyleSheet, View, useWindowDimensions, Alert } from "react-native";
import React from "react";
import { BottomTabNavigationOptions, createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";
import { useTranslation } from "react-i18next";
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { TabTransitionContext } from "../components/CrossfadeTabView";

// Original Screens
import HomeScreen from "../screens/HomeScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import AIHubScreen from "../screens/AIHubScreen";
import InspoScreen from "../screens/InspoScreen";
import ProfileScreen from "../screens/ProfileScreen";

// Screen-level ErrorBoundary — isolates crashes to individual tabs
import { ErrorBoundary } from "../src/components/ErrorBoundary";
import { createLogger } from "../src/utils/logger";
import { useAdminGuard } from "../hooks/useAdminGuard";

const logger = createLogger('TabNavigator');

// 2026 Design System
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { colors, spacing } = LiquidGlass2026Theme;
const TAB_BAR_HORIZONTAL_MARGIN = 16;
const TAB_BAR_HORIZONTAL_PADDING = spacing.sm;

const Tab = createBottomTabNavigator();

// Helper for icon names
const getIconName = (routeName: string, focused: boolean) => {
  if (routeName === "Home") return focused ? "today" : "today-outline";
  if (routeName === "Closet") return focused ? "shirt" : "shirt-outline";
  if (routeName === "AI") return focused ? "sparkles" : "sparkles";
  if (routeName === "Inspo") return focused ? "compass" : "compass-outline";
  if (routeName === "Profile") return focused ? "person" : "person-outline";
  return "help-outline";
};

// ── Tab Icon — static, no 3D animation ────────────────────────────────
interface TabIconProps {
  focused: boolean;
  iconName: string;
  color: string;
  size: number;
  label: string;
}

const TabIcon = ({ focused, iconName, color, size, label }: TabIconProps) => {
  return (
    <View style={styles.tabItemContainer}>
      <Ionicons name={iconName as any} size={26} color={color} />
    </View>
  );
};

// ── Liquid Glass Tab Bar — smooth sliding indicator ───────────────────
const LiquidParallaxTabBar = ({ state, descriptors, navigation, isAdmin }: any) => {
  logger.debug('LiquidParallaxTabBar rendering', { tabIndex: state.index });
  const { width } = useWindowDimensions();
  const fallbackTabBarWidth = Math.max(width - (TAB_BAR_HORIZONTAL_MARGIN * 2), 0);
  const [tabBarWidth, setTabBarWidth] = React.useState(fallbackTabBarWidth);
  const tabWidth =
    Math.max(tabBarWidth - (TAB_BAR_HORIZONTAL_PADDING * 2), 0) / Math.max(state.routes.length, 1);

  // Blob indicator — smooth slide to center of active icon
  const BLOB_SIZE = 48;
  const blobCenterOffset = (tabWidth - BLOB_SIZE) / 2;
  const blobTranslateX = useSharedValue(state.index * tabWidth + blobCenterOffset);

  React.useEffect(() => {
    setTabBarWidth(fallbackTabBarWidth);
  }, [fallbackTabBarWidth]);

  const handleTabBarLayout = React.useCallback((event: LayoutChangeEvent) => {
    const nextWidth = event.nativeEvent.layout.width;
    if (nextWidth > 0) {
      setTabBarWidth(nextWidth);
    }
  }, []);

  // Smooth slide to center of active tab
  React.useEffect(() => {
    const offset = (tabWidth - BLOB_SIZE) / 2;
    blobTranslateX.value = withTiming(state.index * tabWidth + offset, {
      duration: 250,
      easing: Easing.bezier(0.25, 0.1, 0.25, 1),
    });
  }, [state.index, tabWidth, blobTranslateX]);

  const animatedBlobStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: blobTranslateX.value },
    ],
  }));

  return (
    <View
      style={styles.tabBarContainer}
      onLayout={handleTabBarLayout}
    >
      <BlurView
        intensity={Platform.OS === 'ios' ? 80 : 100}
        tint="light"
        style={StyleSheet.absoluteFill}
        pointerEvents="none"
      />
      <LinearGradient
        colors={['rgba(255,255,255,0.94)', 'rgba(240,246,255,0.88)']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.tabBarGradient}
        pointerEvents="none"
      />
      <View style={styles.glassOverlay} pointerEvents="none" />

      {/* Morphing Blob Indicator */}
      <Animated.View style={[styles.indicatorContainer, animatedBlobStyle]} pointerEvents="none">
        <View style={styles.liquidBlob} />
      </Animated.View>

      {/* Tab Buttons */}
      <View style={styles.tabBarContent}>
        {state.routes.map((route: any, index: number) => {
          const { options } = descriptors[route.key];
          const isFocused = state.index === index;
          const activeColor = colors.text.primary;
          const inactiveColor = colors.text.tertiary;
          const iconColor = isFocused ? activeColor : inactiveColor;

          const onPress = () => {
            console.log('[IPAD-DEBUG] Tab pressed:', route.name);
            const event = navigation.emit({
              type: 'tabPress',
              target: route.key,
              canPreventDefault: true,
            });
            if (!isFocused && !event.defaultPrevented) {
              console.log('[IPAD-DEBUG] Navigating to tab:', route.name);
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              navigation.navigate(route.name);
            }
          };

          return (
            <Pressable
              key={route.key}
              accessibilityRole="button"
              accessibilityState={isFocused ? { selected: true } : {}}
              accessibilityLabel={options.tabBarAccessibilityLabel}
              testID={options.tabBarTestID}
              onPress={onPress}
              style={({ pressed }) => [
                styles.tabButton,
                pressed && { opacity: 0.7 }
              ]}
            >
              <TabIcon
                focused={isFocused}
                iconName={getIconName(route.name, isFocused)}
                color={iconColor}
                size={24}
                label={route.name}
              />
            </Pressable>
          );
        })}
      </View>
    </View>
  );
};

// ── Stable wrapper factory — creates components ONCE ─────────────────
const createAnimatedTabScreen = (Screen: React.ComponentType<any>, tabIndex: number) => {
  const Wrapped = (props: any) => {
    // DISABLED: CrossfadeTabView causes touch issues on iPad
    // Using direct screen rendering instead
    return (
      <ErrorBoundary>
        <Screen {...props} />
      </ErrorBoundary>
    );
  };
  Wrapped.displayName = `AnimatedTab(${tabIndex})`;
  return Wrapped;
};

const AnimatedHomeScreen = createAnimatedTabScreen(HomeScreen, 0);
const AnimatedClosetScreen = createAnimatedTabScreen(MyClosetScreen, 1);
const AnimatedAIScreen = createAnimatedTabScreen(AIHubScreen, 2);
const AnimatedInspoScreen = createAnimatedTabScreen(InspoScreen, 3);
const AnimatedProfileScreen = createAnimatedTabScreen(ProfileScreen, 4);

// ── TabNavigator ─────────────────────────────────────────────────────
const TabNavigator = () => {
  logger.debug('TabNavigator component rendering');
  const { t } = useTranslation();
  const { isAdmin } = useAdminGuard();
  // Shared values for tab transition direction — updated via ref + deferred setState
  const currentTab = useSharedValue(0);
  const previousTab = useSharedValue(0);
  const trackedIndex = React.useRef(0);

  // Deferred state update: shared values are set in useEffect (after commit),
  // NOT during render — this prevents the "Cannot update a component while
  // rendering a different component" crash.
  const [pendingIndex, setPendingIndex] = React.useState<number | null>(null);

  React.useEffect(() => {
    if (pendingIndex !== null && pendingIndex !== trackedIndex.current) {
      previousTab.value = trackedIndex.current;
      currentTab.value = pendingIndex;
      trackedIndex.current = pendingIndex;
    }
  }, [pendingIndex, currentTab, previousTab]);

  // Memoize tabBar to prevent excessive re-renders
  const renderTabBar = React.useCallback((props: any) => {
    const idx = props.state.index;
    if (idx !== trackedIndex.current) {
      queueMicrotask(() => setPendingIndex(idx));
    }
    return <LiquidParallaxTabBar {...props} isAdmin={isAdmin} />;
  }, [isAdmin]);

  const screenOptions = React.useCallback(({ route }: any): BottomTabNavigationOptions => ({
    headerShown: false,
    tabBarShowLabel: false,
    animation: 'fade',
    lazy: false,
  }), []);

  return (
    <TabTransitionContext.Provider value={{ currentTab, previousTab }}>
      <Tab.Navigator
        tabBar={renderTabBar}
        screenOptions={screenOptions}
    >
      <Tab.Screen
        name="Home"
        component={AnimatedHomeScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.home') }}
      />
      <Tab.Screen
        name="Closet"
        component={AnimatedClosetScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.closet') }}
      />
      <Tab.Screen
        name="AI"
        component={AnimatedAIScreen}
        initialParams={{ asTab: true }}
        options={{ tabBarAccessibilityLabel: t('tabs.ai') }}
      />
      <Tab.Screen
        name="Inspo"
        component={AnimatedInspoScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.inspo') }}
      />
      <Tab.Screen
        name="Profile"
        component={AnimatedProfileScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.profile') }}
      />
    </Tab.Navigator>
    </TabTransitionContext.Provider>
  );
};

const styles = StyleSheet.create({
  tabBarContainer: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 34 : 20,
    left: TAB_BAR_HORIZONTAL_MARGIN,
    right: TAB_BAR_HORIZONTAL_MARGIN,
    height: 68,
    borderRadius: 34,
    overflow: 'hidden',
    shadowColor: "#173A65",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    elevation: 8,
    backgroundColor: 'transparent',
    alignSelf: 'center',
  },
  tabBarGradient: {
    ...StyleSheet.absoluteFillObject,
  },
  glassOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255,255,255,0.42)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    borderRadius: 34,
  },
  tabBarContent: {
    flexDirection: 'row',
    height: '100%',
    zIndex: 2,
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingHorizontal: TAB_BAR_HORIZONTAL_PADDING,
    pointerEvents: 'auto' as const,
  },
  tabButton: {
    flex: 1,
    height: 56,
    justifyContent: 'center',
    alignItems: 'center',
    minWidth: 44,
    pointerEvents: 'auto' as const,
  },
  tabItemContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  indicatorContainer: {
    position: 'absolute',
    left: TAB_BAR_HORIZONTAL_PADDING,
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  liquidBlob: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    shadowColor: '#173A65',
    shadowOpacity: 0.14,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
});

export default TabNavigator;