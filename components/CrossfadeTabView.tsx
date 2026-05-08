import React, { useContext, useEffect, useRef } from 'react';
import { Dimensions, StyleSheet, View } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
  withSequence,
  Easing,
  interpolate,
  Extrapolation,
  type SharedValue,
} from 'react-native-reanimated';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ── "Aurora Morph" — Innovative Tab Transition ────────────────────────
// The new page morphs from a rounded card shape into full screen,
// like an iris/camera aperture opening. Combined with directional
// parallax, elastic scale bounce, and a sweeping light ray overlay.

// Morph: borderRadius animates from large (rounded card) → 0 (full screen)
const MORPH_RADIUS_START = 48;
const MORPH_RADIUS_END = 0;

// Parallax slide
const ENTER_PARALLAX = 0.30;
const EXIT_PARALLAX = 0.12;

// Scale
const ENTER_SCALE = 0.82;
const EXIT_SCALE = 0.90;
const OVERSHOOT = 1.05;

// Timing
const ENTER_MS = 500;
const EXIT_MS = 340;

// Spring configs
const ENTER_SPRING = { damping: 16, stiffness: 150, mass: 0.5 };
const EXIT_SPRING = { damping: 22, stiffness: 240, mass: 0.6 };
const SCALE_SPRING = { damping: 12, stiffness: 200, mass: 0.4 };

// ── Context: shared values for tab index tracking ────────────────────
export const TabTransitionContext = React.createContext<{
  currentTab: SharedValue<number>;
  previousTab: SharedValue<number>;
}>({
  currentTab: { value: 0 } as SharedValue<number>,
  previousTab: { value: 0 } as SharedValue<number>,
});

interface CrossfadeTabViewProps {
  children: React.ReactNode;
  isActive: boolean;
  index: number;
}

/**
 * "Aurora Morph" — Innovative multi-layered tab transition:
 *
 * 1. **Iris Morph** — New page starts as a rounded card (borderRadius: 48)
 *    and morphs into full-screen rectangle (borderRadius: 0), creating
 *    a camera-aperture "iris open" reveal effect
 *
 * 2. **Directional Parallax** — Page slides in from the navigation direction
 *    (left if tab index increases, right if decreases)
 *
 * 3. **Elastic Scale Bounce** — Enter: 0.82→1.05→1.0 (overshoot landing),
 *    Exit: 1.0→0.90 (shrink away)
 *
 * 4. **Light Ray Sweep** — A diagonal gradient overlay sweeps across
 *    the page during entrance, creating a "light passing through" effect
 *
 * 5. **Coordinated Timing** — Enter and exit run simultaneously with
 *    matched durations, so there's never a blank flash between tabs
 */
export const CrossfadeTabView: React.FC<CrossfadeTabViewProps> = ({
  children,
  isActive,
  index,
}) => {
  const { currentTab, previousTab } = useContext(TabTransitionContext);

  const progress = useSharedValue(isActive ? 1 : 0);
  const translateX = useSharedValue(0);
  const scale = useSharedValue(1);
  const borderRadius = useSharedValue(0);
  const lightSweep = useSharedValue(0);

  const wasActive = useRef(isActive);

  useEffect(() => {
    const prevIdx = previousTab.value;
    const direction = prevIdx < index ? 1 : -1;

    if (isActive && !wasActive.current) {
      // ── ENTERING ────────────────────────────────────────────
      // Start: rounded card, off-screen, scaled down
      progress.value = 0;
      translateX.value = direction * SCREEN_WIDTH * ENTER_PARALLAX;
      scale.value = ENTER_SCALE;
      borderRadius.value = MORPH_RADIUS_START;
      lightSweep.value = 0;

      // Coordinated entrance — all fire simultaneously
      progress.value = withTiming(1, {
        duration: ENTER_MS,
        easing: Easing.out(Easing.cubic),
      });
      translateX.value = withSpring(0, ENTER_SPRING);
      scale.value = withSequence(
        withSpring(OVERSHOOT, SCALE_SPRING),
        withSpring(1, { damping: 18, stiffness: 260, mass: 0.5 })
      );
      // Iris morph: borderRadius 48→0 with spring for organic feel
      borderRadius.value = withSpring(MORPH_RADIUS_END, {
        damping: 14, stiffness: 180, mass: 0.5,
      });
      // Light ray sweep: quick pass then fade
      lightSweep.value = withSequence(
        withTiming(1, { duration: ENTER_MS * 0.6, easing: Easing.inOut(Easing.quad) }),
        withTiming(0, { duration: ENTER_MS * 0.4, easing: Easing.out(Easing.quad) })
      );

    } else if (!isActive && wasActive.current) {
      // ── EXITING ──────────────────────────────────────────────
      progress.value = withTiming(0, {
        duration: EXIT_MS,
        easing: Easing.in(Easing.cubic),
      });
      translateX.value = withSpring(-direction * SCREEN_WIDTH * EXIT_PARALLAX, EXIT_SPRING);
      scale.value = withTiming(EXIT_SCALE, {
        duration: EXIT_MS,
        easing: Easing.in(Easing.cubic),
      });
      // Reverse morph: screen rounds off as it leaves
      borderRadius.value = withTiming(MORPH_RADIUS_START * 0.6, {
        duration: EXIT_MS,
        easing: Easing.in(Easing.cubic),
      });
      lightSweep.value = 0;
    }

    wasActive.current = isActive;
  }, [isActive, index, currentTab, previousTab]);

  const animatedStyle = useAnimatedStyle(() => {
    const opacityVal = interpolate(
      progress.value,
      [0, 0.15, 0.5, 1],
      [0, 0.3, 0.75, 1],
      Extrapolation.CLAMP
    );

    return {
      opacity: opacityVal,
      transform: [
        { translateX: translateX.value },
        { scale: scale.value },
      ],
      borderRadius: borderRadius.value,
    };
  });

  const lightSweepStyle = useAnimatedStyle(() => ({
    opacity: interpolate(lightSweep.value, [0, 0.4, 0.7, 1], [0, 0.18, 0.12, 0], Extrapolation.CLAMP),
    transform: [
      { translateX: interpolate(lightSweep.value, [0, 1], [-SCREEN_WIDTH, SCREEN_WIDTH * 0.5], Extrapolation.CLAMP) },
      { skewX: '-25deg' },
    ],
  }));

  return (
    <Animated.View style={[styles.container, animatedStyle]} pointerEvents="box-none">
      <View style={StyleSheet.absoluteFill} pointerEvents="auto">
        {children}
      </View>
      {/* Light ray sweep overlay */}
      <Animated.View style={[styles.lightRay, lightSweepStyle]} pointerEvents="none" />
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    overflow: 'hidden',
  },
  lightRay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 999,
    overflow: 'hidden',
    backgroundColor: 'rgba(255,255,255,0.7)',
    width: SCREEN_WIDTH * 0.6,
  },
});

export default CrossfadeTabView;
