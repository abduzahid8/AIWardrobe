/**
 * navigation/liquidTransitions.ts
 * =================================
 * Custom "Liquid" page transition presets for @react-navigation/stack.
 *
 * Three presets:
 *  - LiquidSlide  — horizontal push with elastic scale + fade (default)
 *  - LiquidRise   — vertical rise from bottom with spring overshoot
 *  - LiquidFade   — gentle fade with subtle scale pulse
 */

import type { StackCardStyleInterpolator } from '@react-navigation/stack';
import { Dimensions } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ── Spring configs for liquid feel ──────────────────────────────────
// Higher stiffness = faster start. Higher damping = less oscillation tail.
// These values cut perceived transition time ~50% vs the original settings
// for snappier page switches.
const LIQUID_SPRING = {
  damping: 32,
  stiffness: 400,
  mass: 0.4,
};

const ELASTIC_SPRING = {
  damping: 30,
  stiffness: 450,
  mass: 0.3,
};

const FADE_SPRING = {
  damping: 34,
  stiffness: 420,
  mass: 0.35,
};

// ── LiquidSlide: horizontal push with elastic scale + fade ──────────
// Reduced translateX distance and scale range for snappier feel
export const LiquidSlideInterpolator: StackCardStyleInterpolator = ({
  current,
  next,
  layouts,
  closing,
}) => {
  'worklet';

  const progress = current.progress;

  const translateX = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [SCREEN_WIDTH * 0.2, 0],
    extrapolate: 'clamp',
  });

  const scale = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [0.96, 1],
    extrapolate: 'clamp',
  });

  const opacity = progress.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0.5, 0.85, 1],
    extrapolate: 'clamp',
  });

  const overlayOpacity = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [0.2, 0],
    extrapolate: 'clamp',
  });

  return {
    cardStyle: {
      transform: [
        { translateX },
        { scale },
      ],
      opacity,
    },
    overlayStyle: {
      opacity: overlayOpacity,
    },
  };
};

// ── LiquidRise: vertical rise from bottom with spring overshoot ──────
// Reduced translateY distance and scale range for snappier feel
export const LiquidRiseInterpolator: StackCardStyleInterpolator = ({
  current,
  next,
  layouts,
  closing,
}) => {
  'worklet';

  const progress = current.progress;

  const translateY = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [SCREEN_HEIGHT * 0.3, 0],
    extrapolate: 'clamp',
  });

  const scale = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [0.94, 1],
    extrapolate: 'clamp',
  });

  const opacity = progress.interpolate({
    inputRange: [0, 0.4, 1],
    outputRange: [0, 0.85, 1],
    extrapolate: 'clamp',
  });

  const overlayOpacity = progress.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 0],
    extrapolate: 'clamp',
  });

  return {
    cardStyle: {
      transform: [
        { translateY },
        { scale },
      ],
      opacity,
    },
    overlayStyle: {
      opacity: overlayOpacity,
    },
  };
};

// ── LiquidFade: gentle fade with subtle scale pulse ─────────────────
export const LiquidFadeInterpolator: StackCardStyleInterpolator = ({
  current,
  next,
  closing,
}) => {
  'worklet';

  const progress = current.progress;

  const scale = progress.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0.96, 1.02, 1],
    extrapolate: 'clamp',
  });

  const opacity = progress.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0, 0.8, 1],
    extrapolate: 'clamp',
  });

  return {
    cardStyle: {
      transform: [{ scale }],
      opacity,
    },
  };
};

// ── Shared spring spec builder ───────────────────────────────────────
const springSpec = (config: typeof LIQUID_SPRING) => ({
  open: {
    animation: 'spring' as const,
    config,
  },
  close: {
    animation: 'spring' as const,
    config: {
      damping: config.damping + 4,
      stiffness: config.stiffness + 40,
      mass: config.mass,
    },
  },
});

// ── Convenience screen option presets ───────────────────────────────
export const LiquidPresets = {
  /** Default horizontal liquid slide */
  slide: {
    cardStyleInterpolator: LiquidSlideInterpolator,
    transitionSpec: springSpec(LIQUID_SPRING),
    gestureEnabled: true,
    gestureDirection: 'horizontal' as const,
  },
  /** Vertical rise (for modals, AI screens, camera) */
  rise: {
    cardStyleInterpolator: LiquidRiseInterpolator,
    transitionSpec: springSpec(ELASTIC_SPRING),
    gestureEnabled: true,
    gestureDirection: 'vertical' as const,
  },
  /** Gentle fade (for auth, trial expired) */
  fade: {
    cardStyleInterpolator: LiquidFadeInterpolator,
    transitionSpec: springSpec(FADE_SPRING),
    gestureEnabled: false,
  },
};
