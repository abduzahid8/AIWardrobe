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
import { ScreenTransitionWrapper } from "../components/ScreenTransitionWrapper";
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

// Memoized — previously re-rendered all 4 icons on every tab press.
const TabIcon = React.memo(({ iconName, color }: TabIconProps) => {
  return (
    <View style={styles.tabItemContainer}>
      <Ionicons name={iconName as any} size={26} color={color} />
    </View>
  );
});
TabIcon.displayName = 'TabIcon';

// Memoized tab button — previously `onPress` + `style` closures were
// re-created for every route on every LiquidParallaxTabBar render,
// forcing all Pressables + icons to reconcile on each tab switch.
interface TabButtonProps {
  routeKey: string;
  routeName: string;
  isFocused: boolean;
  activeColor: string;
  inactiveColor: string;
  accessibilityLabel?: string;
  testID?: string;
  onPress: (routeName: string, routeKey: string, isFocused: boolean) => void;
}

const TabButton = React.memo(({
  routeKey,
  routeName,
  isFocused,
  activeColor,
  inactiveColor,
  accessibilityLabel,
  testID,
  onPress,
}: TabButtonProps) => {
  const handlePress = React.useCallback(() => {
    onPress(routeName, routeKey, isFocused);
  }, [onPress, routeName, routeKey, isFocused]);

  const styleCb = React.useCallback(
    ({ pressed }: { pressed: boolean }) => [
      styles.tabButton,
      pressed && { opacity: 0.7 },
    ],
    [],
  );

  const iconColor = isFocused ? activeColor : inactiveColor;

  return (
    <Pressable
      key={routeKey}
      accessibilityRole="button"
      accessibilityState={isFocused ? { selected: true } : {}}
      accessibilityLabel={accessibilityLabel}
      testID={testID}
      onPress={handlePress}
      style={styleCb}
    >
      <TabIcon
        focused={isFocused}
        iconName={getIconName(routeName, isFocused)}
        color={iconColor}
        size={24}
        label={routeName}
      />
    </Pressable>
  );
});
TabButton.displayName = 'TabButton';

// ── Memoized tab bar background — BlurView + gradient + glass overlay ────────
// Extracted into a separate React.memo component so the expensive BlurView
// GPU pass is paid at most once per mount, not on every tab-press re-render
// of LiquidParallaxTabBar (fixes Defect 1.6).
// NOTE: intensity kept low (30/40) — 95/100 forces a full-screen GPU blur
// re-composite on every tab press / scroll under the bar.
const TabBarBackground = React.memo(() => (
  <>
    <BlurView
      intensity={Platform.OS === 'ios' ? 30 : 40}
      tint="light"
      style={[StyleSheet.absoluteFill, { borderRadius: 34 }]}
      pointerEvents="none"
    />
    <LinearGradient
      colors={['rgba(255,255,255,0.97)', 'rgba(240,246,255,0.94)']}
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
const LiquidParallaxTabBar = ({ state, descriptors, navigation, isAdmin }: any) => {
  const { width } = useWindowDimensions();
  const fallbackTabBarWidth = Math.max(width - (TAB_BAR_HORIZONTAL_MARGIN * 2), 0);
  const [tabBarWidth, setTabBarWidth] = React.useState(fallbackTabBarWidth);
  const tabWidth =
    Math.max(tabBarWidth - (TAB_BAR_HORIZONTAL_PADDING * 2), 0) / Math.max(state.routes.length, 1);

  // Blob indicator — smooth slide to center of active icon
  const BLOB_SIZE = 70; // must match liquidBlob's width (styles.liquidBlob)
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
      duration: 150,
      easing: Easing.bezier(0.25, 0.1, 0.25, 1),
    });
  }, [state.index, tabWidth, blobTranslateX]);

  const animatedBlobStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: blobTranslateX.value },
    ],
  }));

  // Drop accidental multi-taps on the same tab: time-based debounce so it
  // can't swallow legit presses the way an isFocused guard did.
  const lastPressRef = React.useRef<Record<string, number>>({});
  const SAME_TAB_DEBOUNCE_MS = 600;
  const stateRef = React.useRef(state);
  stateRef.current = state;

  // Stable press handler shared by all memoized TabButtons — previously an
  // inline closure per route per render forced full tab-bar reconciliation.
  const handleTabPress = React.useCallback((routeName: string, routeKey: string, isFocused: boolean) => {
    const now = Date.now();
    const last = lastPressRef.current[routeName] ?? 0;
    if (now - last < SAME_TAB_DEBOUNCE_MS) return;
    lastPressRef.current[routeName] = now;
    const cur = stateRef.current;
    if (!isFocused) {
      import('../src/services/tabPrewarmService').then(({ prewarmTab }) => {
        prewarmTab(routeName);
      }).catch(() => {});
    }
    const event = navigation.emit({
      type: 'tabPress',
      target: routeKey,
      canPreventDefault: true,
    });
    if (!event.defaultPrevented) {
      const currentRoute = cur.routes[cur.index]?.name ?? 'Unknown';
      if (!isFocused) {
        perfTabSwitch(currentRoute, routeName);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      }
      navigation.navigate(routeName);
    }
  }, [navigation]);

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

      {/* Tab Buttons — memoized, stable handler */}
      <View style={styles.tabBarContent}>
        {state.routes.map((route: any, index: number) => {
          const { options } = descriptors[route.key];
          const isFocused = state.index === index;
          return (
            <TabButton
              key={route.key}
              routeKey={route.key}
              routeName={route.name}
              isFocused={isFocused}
              activeColor={colors.text.primary}
              inactiveColor={colors.text.tertiary}
              accessibilityLabel={options.tabBarAccessibilityLabel}
              testID={options.tabBarTestID}
              onPress={handleTabPress}
            />
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
      // Defer heavy tab-screen mount past the tab-switch animation so the
      // transition stays at 60fps. Plain setTimeout — InteractionManager is
      // deprecated (startup warning) and runAfterInteractions fires at
      // roughly the same point for this use case.
      const timer = setTimeout(() => {
        setIsReady(true);
      }, 50);
      return () => clearTimeout(timer);
    }, []);

    // Wrap in a View with the app background color so the placeholder shown
    // during lazy first-mount is never a blank white void. Without this,
    // switching to an unvisited tab shows a white flash for one render cycle
    // before the screen's own background paints.
    return (
      <ErrorBoundary>
        <ScreenTransitionWrapper preset="fade_slide_up">
          <View style={{ flex: 1, backgroundColor: '#FFFFFF' }}>
            {isReady ? (
              <Screen {...props} />
            ) : (
              <View style={{ flex: 1, backgroundColor: '#FFFFFF' }} />
            )}
          </View>
        </ScreenTransitionWrapper>
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

  // Memoize tabBar to prevent excessive re-renders
  const renderTabBar = React.useCallback((props: any) => {
    return (
      <LiquidParallaxTabBar
        {...props}
        isAdmin={isAdmin}
      />
    );
  }, [isAdmin]);

  const screenOptions = React.useCallback(({ route }: any): BottomTabNavigationOptions => ({
    headerShown: false,
    tabBarShowLabel: false,
    // Kill the default bottom-tab hairline (borderTopWidth) + Android
    // elevation so no stray line can render behind/above the custom
    // floating pill — e.g. during lazy-mount fallback frames.
    tabBarStyle: {
      borderTopWidth: 0,
      borderWidth: 0,
      elevation: 0,
      shadowOpacity: 0,
      backgroundColor: 'transparent',
    },
    // 'none' eliminates the brief animation window where the white
    // sceneContainerStyle background flashes through during fast tab switches.
    // The blob indicator animation already provides visual feedback.
    animation: 'none',
    // Keep lazy: true (default) so screens mount only when first visited —
    // this keeps startup fast and the JS thread unblocked.
    // NOTE: freezeOnBlur was removed — it suspends blurred screens via
    // react-freeze and under rapid tab switches a screen could stay stuck
    // in the suspended (blank white) state. Background-tab re-renders are
    // already contained by memoized cells + selective zustand subscriptions.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    lazy: true as any,
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
    <Tab.Navigator
      tabBar={renderTabBar}
      screenOptions={screenOptions}
      screenListeners={screenListeners}
      // @ts-ignore - sceneContainerStyle is supported at runtime but fails typecheck in this React Navigation version
      sceneContainerStyle={{ backgroundColor: '#FFFFFF' }}
      // @ts-ignore - supported at runtime in this version, detaches background tabs natively
      detachInactiveScreens
    >
      <Tab.Screen
        name="Home"
        component={AnimatedHomeScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.home'), tabBarTestID: 'homeTab' } as any}
      />
      <Tab.Screen
        name="Closet"
        component={AnimatedClosetScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.closet'), tabBarTestID: 'closetTab' } as any}
      />
      <Tab.Screen
        name="Inspo"
        component={AnimatedInspoScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.inspo'), tabBarTestID: 'inspoTab' } as any}
      />
      <Tab.Screen
        name="Profile"
        component={AnimatedProfileScreen}
        options={{ tabBarAccessibilityLabel: t('tabs.profile'), tabBarTestID: 'profileTab' } as any}
      />
    </Tab.Navigator>
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
    // No border — the 1px glassOverlay border used to render as a stray
    // hairline along the top edge of the pill (clipped by overflow:hidden).
    borderWidth: 0,
    borderTopWidth: 0,
    shadowColor: "#173A65",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    // elevation on a transparent + overflow:hidden rounded view draws a
    // rectangular shadow that leaks out as a straight line above the pill
    // on Android — keep it at 0 and rely on the iOS shadow props.
    elevation: 0,
    backgroundColor: 'transparent',
    alignSelf: 'center',
  },
  tabBarGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    borderRadius: 34,
  },
  glassOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    backgroundColor: 'rgba(255,255,255,0.6)',
    borderWidth: 0,
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
    // backgroundColor: '#000',
    padding:10,
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
    width: 70,
    height: 50,
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
