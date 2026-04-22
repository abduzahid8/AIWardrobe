import React, { useContext, useEffect, useRef } from 'react';
import { Dimensions, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
  withSequence,
  withDelay,
  Easing,
  interpolate,
  Extrapolation,
  type SharedValue,
} from 'react-native-reanimated';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ── "Aurora Flow" Animation Constants ────────────────────────────────
const PERSPECTIVE = 900;
const ENTER_ROTATE = 12;
const EXIT_ROTATE = -8;
const PARALLAX_RATIO = 0.22;
const ENTER_MS = 480;
const EXIT_MS = 300;

const ELASTIC = { damping: 10, stiffness: 180, mass: 0.5 };
const SETTLE = { damping: 20, stiffness: 300, mass: 0.6 };

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
 * "Aurora Flow" — innovative multi-layered tab transition:
 * 1. 3D perspective card rotation (rotateY)
 * 2. Directional parallax slide (translateX)
 * 3. Elastic scale overshoot (0.88 → 1.06 → 1.0)
 * 4. Aurora shimmer sweep overlay
 * 5. Staggered opacity reveal
 *
 * Uses useEffect (JS thread) to trigger animations when isActive changes.
 * Direction is read from context shared values (set by TabNavigator).
 * No filter/blur (crashes on most RN versions), no ref mutations in worklets.
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
  const rotateY = useSharedValue(0);
  const shimmer = useSharedValue(0);

  const hasMounted = useRef(false);
  const wasActive = useRef(isActive);

  useEffect(() => {
    // Read direction from shared values (updated by TabNavigator before this runs)
    const prevIdx = previousTab.value;
    const direction = prevIdx < index ? 1 : -1;

    if (isActive && !wasActive.current) {
      // ── ENTERING ────────────────────────────────────────────
      if (!hasMounted.current) {
        // First mount: let screen's own layout animations play
        progress.value = 1;
        translateX.value = 0;
        scale.value = 1;
        rotateY.value = 0;
        shimmer.value = 0;
        hasMounted.current = true;
        wasActive.current = true;
        return;
      }

      // Subsequent focus: full Aurora Flow entrance
      progress.value = 0;
      translateX.value = direction * SCREEN_WIDTH * PARALLAX_RATIO;
      scale.value = 0.88;
      rotateY.value = direction * ENTER_ROTATE;
      shimmer.value = 0;

      // Staggered reveal: slight delay then fade in
      progress.value = withDelay(30, withTiming(1, {
        duration: ENTER_MS,
        easing: Easing.out(Easing.cubic),
      }));
      // Directional parallax with elastic spring
      translateX.value = withSpring(0, ELASTIC);
      // Scale: 0.88 → overshoot 1.06 → settle 1.0
      scale.value = withSequence(
        withSpring(1.06, ELASTIC),
        withSpring(1, SETTLE)
      );
      // 3D card rotation settles
      rotateY.value = withSpring(0, { damping: 16, stiffness: 240, mass: 0.6 });
      // Aurora shimmer sweeps across during entrance
      shimmer.value = withSequence(
        withTiming(1, { duration: ENTER_MS * 0.5, easing: Easing.inOut(Easing.quad) }),
        withTiming(0, { duration: ENTER_MS * 0.5, easing: Easing.out(Easing.quad) })
      );

    } else if (!isActive && wasActive.current) {
      // ── EXITING ──────────────────────────────────────────────
      progress.value = withTiming(0, {
        duration: EXIT_MS,
        easing: Easing.in(Easing.cubic),
      });
      translateX.value = withSpring(-direction * SCREEN_WIDTH * PARALLAX_RATIO * 0.5, {
        damping: 24, stiffness: 260, mass: 0.7,
      });
      scale.value = withTiming(0.94, {
        duration: EXIT_MS,
        easing: Easing.in(Easing.cubic),
      });
      rotateY.value = withTiming(-direction * EXIT_ROTATE, {
        duration: EXIT_MS,
        easing: Easing.in(Easing.cubic),
      });
      shimmer.value = 0;
    }

    wasActive.current = isActive;
  }, [isActive, index, currentTab, previousTab]);

  const animatedStyle = useAnimatedStyle(() => {
    const opacityVal = interpolate(progress.value, [0, 0.3, 1], [0, 0.6, 1], Extrapolation.CLAMP);
    const scaleVal = interpolate(scale.value, [0.88, 1.06, 1], [0.88, 1.06, 1], Extrapolation.CLAMP);

    return {
      opacity: opacityVal,
      transform: [
        { perspective: PERSPECTIVE },
        { translateX: translateX.value },
        { scale: scaleVal },
        { rotateY: `${rotateY.value}deg` },
      ],
    };
  });

  const shimmerStyle = useAnimatedStyle(() => ({
    opacity: interpolate(shimmer.value, [0, 0.5, 1], [0, 0.12, 0], Extrapolation.CLAMP),
    transform: [
      { translateX: interpolate(shimmer.value, [0, 1], [-SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.3], Extrapolation.CLAMP) },
    ],
  }));

  return (
    <Animated.View style={[styles.container, animatedStyle]}>
      {children}
      <Animated.View style={[styles.shimmerOverlay, shimmerStyle]} pointerEvents="none">
        <Animated.View style={styles.shimmerGradient} />
      </Animated.View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    overflow: 'hidden',
  },
  shimmerOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 999,
    overflow: 'hidden',
  },
  shimmerGradient: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.6)',
    transform: [{ skewX: '-20deg' }],
  },
});

export default CrossfadeTabView;
