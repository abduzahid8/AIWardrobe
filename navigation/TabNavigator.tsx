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
  withTiming,
  Easing,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { CrossfadeTabView, TabTransitionContext } from "../components/CrossfadeTabView";
import { useIsFocused } from "@react-navigation/native";

// Original Screens
import HomeScreen from "../screens/HomeScreen";
import MyClosetScreen from "../screens/MyClosetScreen";
import AITryOnScreen from "../screens/AITryOnScreen";
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
const TAB_BAR_HORIZONTAL_MARGIN = 20;
const TAB_BAR_HORIZONTAL_PADDING = spacing.xs;

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
const LiquidParallaxTabBar = ({ state, descriptors, navigation }: any) => {
  logger.debug('LiquidParallaxTabBar rendering', { tabIndex: state.index });
  const { width } = useWindowDimensions();
  const fallbackTabBarWidth = Math.max(width - (TAB_BAR_HORIZONTAL_MARGIN * 2), 0);
  const [tabBarWidth, setTabBarWidth] = React.useState(fallbackTabBarWidth);
  const tabWidth =
    Math.max(tabBarWidth - (TAB_BAR_HORIZONTAL_PADDING * 2), 0) / Math.max(state.routes.length, 1);

  // Blob indicator — smooth slide to center of active icon
  const BLOB_SIZE = 56;
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
      <View style={styles.glassOverlay} />

      {/* Morphing Blob Indicator */}
      <Animated.View style={[styles.indicatorContainer, animatedBlobStyle]}>
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
              <TabIcon
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

// ── Stable wrapper factory — creates components ONCE ─────────────────
const createAnimatedTabScreen = (Screen: React.ComponentType<any>, tabIndex: number) => {
  const Wrapped = (props: any) => {
    // `useIsFocused()` is reactive — the wrapper re-renders on every
    // focus/blur so CrossfadeTabView's `isActive` prop always reflects
    // the current focus state. Using `navigation.isFocused()` directly
    // is NOT reactive and, with `lazy: true`, returns false on the
    // initial mount of a newly-focused tab → opacity stuck at 0 →
    // white screen.
    const isActive = useIsFocused();
    return (
      <ErrorBoundary>
        <CrossfadeTabView isActive={isActive} index={tabIndex}>
          <Screen {...props} />
        </CrossfadeTabView>
      </ErrorBoundary>
    );
  };
  Wrapped.displayName = `AnimatedTab(${tabIndex})`;
  return Wrapped;
};

const AnimatedHomeScreen = createAnimatedTabScreen(HomeScreen, 0);
const AnimatedClosetScreen = createAnimatedTabScreen(MyClosetScreen, 1);
const AnimatedAIScreen = createAnimatedTabScreen(AITryOnScreen, 2);
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

  return (
    <TabTransitionContext.Provider value={{ currentTab, previousTab }}>
    <Tab.Navigator
      tabBar={(props) => {
        const idx = props.state.index;
        if (idx !== trackedIndex.current) {
          // Defer setState via queueMicrotask — runs after render completes
          queueMicrotask(() => setPendingIndex(idx));
        }
        return <LiquidParallaxTabBar {...props} />;
      }}
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarShowLabel: false,
        animation: 'fade',
        lazy: true,
      })}
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
      {isAdmin && (
        <Tab.Screen
          name="Inspo"
          component={AnimatedInspoScreen}
          options={{ tabBarAccessibilityLabel: t('tabs.inspo') }}
        />
      )}
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
    minWidth: spacing.touchTarget.minimum,
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