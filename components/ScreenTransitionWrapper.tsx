import React, { useEffect, useRef } from 'react';
import { Dimensions, Platform, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
  withSequence,
  Easing,
  runOnJS,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import { useFocusEffect } from '@react-navigation/native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ============================================
// TRANSITION PRESETS
// ============================================

export type TransitionPreset =
  | 'fade_slide_up'      // Content fades in + slides up (default for push screens)
  | 'fade_scale'         // Content fades in + scales from 0.96 (for AI/magic screens)
  | 'slide_from_side'    // Content slides in from right with fade
  | 'fade_only'          // Simple crossfade
  | 'bounce_in'          // Playful bounce entrance (for fun/interactive screens)
  | 'slide_from_bottom'; // Content slides up from bottom (for modals/sheets)

interface ScreenTransitionWrapperProps {
  children: React.ReactNode;
  preset?: TransitionPreset;
  delay?: number;
  style?: ViewStyle;
}

/**
 * Wraps screen content with smooth enter/exit animations.
 * Uses useFocusEffect to re-trigger animation when the screen gains focus.
 *
 * Usage: Wrap your screen's return value:
 *   return <ScreenTransitionWrapper preset="fade_scale">{...}</ScreenTransitionWrapper>
 */
export const ScreenTransitionWrapper: React.FC<ScreenTransitionWrapperProps> = ({
  children,
  preset = 'fade_slide_up',
  delay = 0,
  style,
}) => {
  const progress = useSharedValue(0);
  const hasAnimated = useRef(false);

  const animateIn = () => {
    'worklet';
    if (delay > 0) {
      progress.value = withSequence(
        withTiming(0, { duration: delay }),
        withTiming(1, {
          duration: 450,
          easing: Easing.out(Easing.cubic),
        })
      );
    } else {
      progress.value = withTiming(1, {
        duration: 450,
        easing: Easing.out(Easing.cubic),
      });
    }
  };

  useFocusEffect(
    React.useCallback(() => {
      // Reset and animate in on each focus
      progress.value = 0;
      animateIn();
      return () => {
        // Animate out when losing focus
        progress.value = withTiming(0, {
          duration: 200,
          easing: Easing.in(Easing.cubic),
        });
      };
    }, [preset, delay])
  );

  const animatedStyle = useAnimatedStyle(() => {
    const p = progress.value;

    switch (preset) {
      case 'fade_slide_up':
        return {
          opacity: interpolate(p, [0, 1], [0, 1], Extrapolation.CLAMP),
          transform: [
            { translateY: interpolate(p, [0, 1], [40, 0], Extrapolation.CLAMP) },
          ],
        };

      case 'fade_scale':
        return {
          opacity: interpolate(p, [0, 1], [0, 1], Extrapolation.CLAMP),
          transform: [
            { scale: interpolate(p, [0, 1], [0.92, 1], Extrapolation.CLAMP) },
          ],
        };

      case 'slide_from_side':
        return {
          opacity: interpolate(p, [0, 0.3, 1], [0, 0.5, 1], Extrapolation.CLAMP),
          transform: [
            { translateX: interpolate(p, [0, 1], [SCREEN_WIDTH * 0.15, 0], Extrapolation.CLAMP) },
          ],
        };

      case 'fade_only':
        return {
          opacity: interpolate(p, [0, 1], [0, 1], Extrapolation.CLAMP),
        };

      case 'bounce_in': {
        return {
          opacity: interpolate(p, [0, 0.5, 1], [0, 0.8, 1], Extrapolation.CLAMP),
          transform: [
            { scale: interpolate(p, [0, 0.6, 0.8, 1], [0.8, 1.04, 0.98, 1], Extrapolation.CLAMP) },
          ],
        };
      }

      case 'slide_from_bottom':
        return {
          opacity: interpolate(p, [0, 1], [0, 1], Extrapolation.CLAMP),
          transform: [
            { translateY: interpolate(p, [0, 1], [SCREEN_HEIGHT * 0.15, 0], Extrapolation.CLAMP) },
          ],
        };

      default:
        return { opacity: p };
    }
  });

  return (
    <Animated.View style={[{ flex: 1 }, animatedStyle, style]}>
      {children}
    </Animated.View>
  );
};

export default ScreenTransitionWrapper;
