/**
 * navigation/liquidTransitions.ts
 * =================================
 * Native-stack transition presets. The RootNavigator uses
 * @react-navigation/native-stack, so pushes/pops/gestures run as true
 * platform transitions (UIViewController / Fragment) at 60fps — not JS
 * interpolators. Direction is implied by `animation`; `gestureDirection`
 * keeps the swipe axis matching the animation axis.
 */
import type { NativeStackNavigationOptions } from '@react-navigation/native-stack';

// ── Convenience screen option presets ───────────────────────────────
export const LiquidPresets: Record<
  'slide' | 'rise' | 'fade',
  Pick<
    NativeStackNavigationOptions,
    'animation' | 'gestureEnabled' | 'gestureDirection' | 'presentation'
  >
> = {
  /** Default horizontal push (iOS edge-swipe / Android back included) */
  slide: {
    animation: 'slide_from_right',
    gestureEnabled: true,
    gestureDirection: 'horizontal',
  },
  /** Vertical rise (AI screens, camera, sheets) */
  rise: {
    animation: 'slide_from_bottom',
    gestureEnabled: true,
    gestureDirection: 'vertical',
  },
  /** Gentle fade (auth, paywall, trial screens) */
  fade: {
    animation: 'fade',
    gestureEnabled: false,
  },
};
