import { LayoutChangeEvent, Platform, Pressable, StyleSheet, View, useWindowDimensions, Alert, InteractionManager } from "react-native";
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
import { perfTabSwitch, perfTabSwitchComplete } from "../src/utils/perf";

// Original Screens
import HomeScreen from "../screens/HomeScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
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
const TAB_BAR_HORIZONTAL_MARGIN = 64;
const TAB_BAR_HORIZONTAL_PADDING = spacing.sm;

const Tab = createBottomTabNavigator();

// Helper for icon names
const getIconName = (routeName: string, focused: boolean) => {
  if (routeName === "Home") return focused ? "today" : "today-outline";
  if (routeName === "Closet") return focused ? "shirt" : "shirt-outline";
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

// ── Memoized tab bar background — BlurView + gradient + glass overlay ────────
// Extracted into a separate React.memo component so the expensive BlurView
// GPU pass is paid at most once per mount, not on every tab-press re-render
// of LiquidParallaxTabBar (fixes Defect 1.6).
const TabBarBackground = React.memo(() => (
  <>
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
  </>
));
TabBarBackground.displayName = 'TabBarBackground';

// ── Liquid Glass Tab Bar — smooth sliding indicator ───────────────────
const LiquidParallaxTabBar = ({ state, descriptors, navigation, isAdmin, onActiveIndexChange }: any) => {
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

  React.useEffect(() => {
    onActiveIndexChange?.(state.index);
  }, [state.index, onActiveIndexChange]);

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
      duration: 150,
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
      <TabBarBackground />

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
            if (!event.defaultPrevented) {
              console.log('[IPAD-DEBUG] Navigating to tab:', route.name);
              // Record the current tab name for perf timing
              const currentRoute = state.routes[state.index]?.name ?? 'Unknown';
              // Only record perf + haptics when actually switching tabs
              if (!isFocused) {
                perfTabSwitch(currentRoute, route.name);
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              }
              // Always call navigate — React Navigation deduplicates same-screen
              // navigations, so this is safe even when already focused.
              // Using isFocused as a guard caused a double-tap bug: on rapid
              // taps the stale state.index made isFocused=true even when the
              // tab wasn't actually active yet, silently swallowing the press.
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
    const [isReady, setIsReady] = React.useState(false);

    React.useEffect(() => {
      const task = InteractionManager.runAfterInteractions(() => {
        setIsReady(true);
      });
      return () => task.cancel();
    }, []);

    // Wrap in a View with the app background color so the placeholder shown
    // during lazy first-mount is never a blank white void. Without this,
    // switching to an unvisited tab shows a white flash for one render cycle
    // before the screen's own background paints.
    return (
      <ErrorBoundary>
        <View style={{ flex: 1, backgroundColor: '#FFFFFF' }}>
          {isReady ? (
            <Screen {...props} />
          ) : (
            <View style={{ flex: 1, backgroundColor: '#FFFFFF' }} />
          )}
        </View>
      </ErrorBoundary>
    );
  };
  Wrapped.displayName = `AnimatedTab(${tabIndex})`;
  return Wrapped;
};

const AnimatedHomeScreen = createAnimatedTabScreen(HomeScreen, 0);
const AnimatedClosetScreen = createAnimatedTabScreen(MyClosetScreen, 1);
const AnimatedInspoScreen = createAnimatedTabScreen(InspoScreen, 2);
const AnimatedProfileScreen = createAnimatedTabScreen(ProfileScreen, 3);

// ── TabNavigator ─────────────────────────────────────────────────────
const TabNavigator = () => {
  const { t } = useTranslation();
  // Only subscribe to isAdmin (boolean) — the loading state change from
  // useAdminGuard would otherwise cause TabNavigator to re-render on every
  // app launch when the async admin check resolves.
  const { isAdmin } = useAdminGuard();
  // Shared values for tab transition direction — updated via ref + deferred setState
  const currentTab = useSharedValue(0);
  const previousTab = useSharedValue(0);
  const trackedIndex = React.useRef(0);

  const handleActiveIndexChange = React.useCallback((nextIndex: number) => {
    if (nextIndex !== trackedIndex.current) {
      previousTab.value = trackedIndex.current;
      currentTab.value = nextIndex;
      trackedIndex.current = nextIndex;
    }
  // currentTab and previousTab are Reanimated shared values (mutable refs) —
  // they do not trigger re-renders and do not need to be listed as deps.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Memoize tabBar to prevent excessive re-renders
  const renderTabBar = React.useCallback((props: any) => {
    return (
      <LiquidParallaxTabBar
        {...props}
        isAdmin={isAdmin}
        onActiveIndexChange={handleActiveIndexChange}
      />
    );
  }, [handleActiveIndexChange, isAdmin]);

  const screenOptions = React.useCallback(({ route }: any): BottomTabNavigationOptions => ({
    headerShown: false,
    tabBarShowLabel: false,
    // 'none' eliminates the brief animation window where the white
    // sceneContainerStyle background flashes through during fast tab switches.
    // The blob indicator animation already provides visual feedback.
    animation: 'none',
    // Keep lazy: true (default) so screens mount only when first visited —
    // this keeps startup fast and the JS thread unblocked.
    // The white flash on first visit is handled by sceneContainerStyle below
    // (matching the app background color) so there's nothing white to show.
  }), []);

  // Fire perfTabSwitchComplete on every tab focus — this is the moment the
  // screen is actually visible to the user (after the navigator animation).
  const screenListeners = React.useMemo(() => ({
    focus: (e: any) => {
      const routeName = e.target?.split('-')[0] ?? '';
      if (routeName) {
        perfTabSwitchComplete(routeName);
      }
    },
  }), []);

  return (
    <TabTransitionContext.Provider value={{ currentTab, previousTab }}>
      <Tab.Navigator
        tabBar={renderTabBar}
        screenOptions={screenOptions}
        screenListeners={screenListeners}
        // @ts-ignore - sceneContainerStyle is supported at runtime but fails typecheck in this React Navigation version
        sceneContainerStyle={{ backgroundColor: '#FFFFFF' }}
    >
      <Tab.Screen
        name="Home"
        component={AnimatedHomeScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.home'), tabBarTestID: 'homeTab' }}
      />
      <Tab.Screen
        name="Closet"
        component={AnimatedClosetScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.closet'), tabBarTestID: 'closetTab' }}
      />
      <Tab.Screen
        name="Inspo"
        component={AnimatedInspoScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.inspo'), tabBarTestID: 'inspoTab' }}
      />
      <Tab.Screen
        name="Profile"
        component={AnimatedProfileScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.profile'), tabBarTestID: 'profileTab' }}
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
