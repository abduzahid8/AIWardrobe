// AIWardrobe Animation Presets
// Inspired by: Alta Daily (floating, 3D), 33 Club (smooth transitions), DiWander (interactive)

import {
    withSpring,
    withTiming,
    withSequence,
    withRepeat,
    Easing,
    WithSpringConfig,
    WithTimingConfig,
} from 'react-native-reanimated';

// ============================================
// SPRING CONFIGURATIONS
// ============================================

// Smooth, natural spring (default)
export const springConfig: WithSpringConfig = {
    damping: 20,
    stiffness: 200,
    mass: 1,
};

// Quick, responsive spring
export const springFastConfig: WithSpringConfig = {
    damping: 25,
    stiffness: 350,
    mass: 0.8,
};

// Bouncy, playful spring
export const springBouncyConfig: WithSpringConfig = {
    damping: 12,
    stiffness: 120,
    mass: 1,
};

// Gentle, slow spring
export const springGentleConfig: WithSpringConfig = {
    damping: 30,
    stiffness: 150,
    mass: 1.2,
};

// ============================================
// TIMING CONFIGURATIONS
// ============================================

export const timingFastConfig: WithTimingConfig = {
    duration: 150,
    easing: Easing.out(Easing.cubic),
};

export const timingNormalConfig: WithTimingConfig = {
    duration: 250,
    easing: Easing.out(Easing.cubic),
};

export const timingSlowConfig: WithTimingConfig = {
    duration: 400,
    easing: Easing.out(Easing.cubic),
};

export const timingLinearConfig: WithTimingConfig = {
    duration: 250,
    easing: Easing.linear,
};

export const timingBounceConfig: WithTimingConfig = {
    duration: 500,
    easing: Easing.bounce,
};

// ============================================
// PRESET ANIMATIONS
// ============================================

// Button press animation (scale down)
export const buttonPressAnimation = (pressed: boolean, scale = 0.97) => {
    'worklet';
    return withSpring(pressed ? scale : 1, springFastConfig);
};

// Fade in animation
export const fadeInAnimation = (delay = 0) => {
    'worklet';
    return withTiming(1, {
        duration: 600,
        easing: Easing.out(Easing.cubic),
    });
};

// Fade out animation
export const fadeOutAnimation = () => {
    'worklet';
    return withTiming(0, {
        duration: 300,
        easing: Easing.in(Easing.cubic),
    });
};

// Slide in from bottom (bottom sheet, modals)
export const slideInFromBottomAnimation = () => {
    'worklet';
    return withSpring(0, springConfig);
};

// Slide out to bottom
export const slideOutToBottomAnimation = (height: number) => {
    'worklet';
    return withTiming(height, {
        duration: 250,
        easing: Easing.in(Easing.cubic),
    });
};

// Scale in animation (cards appearing)
export const scaleInAnimation = (delay = 0) => {
    'worklet';
    return withSequence(
        withTiming(0, { duration: 0 }),
        withTiming(delay, { duration: delay }),
        withSpring(1, springBouncyConfig)
    );
};

// Scale out animation
export const scaleOutAnimation = () => {
    'worklet';
    return withTiming(0, {
        duration: 200,
        easing: Easing.in(Easing.cubic),
    });
};

// Alta-style floating animation (continuous)
export const floatingAnimation = (amplitude = 3, duration = 3000) => {
    'worklet';
    return withRepeat(
        withSequence(
            withTiming(-amplitude, {
                duration: duration / 2,
                easing: Easing.inOut(Easing.sin),
            }),
            withTiming(amplitude, {
                duration: duration / 2,
                easing: Easing.inOut(Easing.sin),
            })
        ),
        -1, // infinite
        true // reverse
    );
};

// Glow/pulse animation (continuous)
export const glowAnimation = (minOpacity = 0.2, maxOpacity = 0.5, duration = 2000) => {
    'worklet';
    return withRepeat(
        withSequence(
            withTiming(minOpacity, {
                duration: duration / 2,
                easing: Easing.inOut(Easing.sin),
            }),
            withTiming(maxOpacity, {
                duration: duration / 2,
                easing: Easing.inOut(Easing.sin),
            })
        ),
        -1,
        true
    );
};

// Rotate animation (for loading spinners)
export const rotateAnimation = (duration = 1000) => {
    'worklet';
    return withRepeat(
        withTiming(360, {
            duration,
            easing: Easing.linear,
        }),
        -1,
        false
    );
};

// Shake animation (error feedback)
export const shakeAnimation = () => {
    'worklet';
    return withSequence(
        withTiming(-10, { duration: 50 }),
        withTiming(10, { duration: 50 }),
        withTiming(-10, { duration: 50 }),
        withTiming(10, { duration: 50 }),
        withTiming(0, { duration: 50 })
    );
};

// Heart beat animation (favorites)
export const heartBeatAnimation = () => {
    'worklet';
    return withSequence(
        withSpring(1.2, { damping: 10, stiffness: 300 }),
        withSpring(1, springConfig)
    );
};

// Swipe gesture animation
export const swipeAnimation = (translateX: number, threshold: number) => {
    'worklet';
    const shouldDismiss = Math.abs(translateX) > threshold;

    if (shouldDismiss) {
        return withTiming(translateX > 0 ? 500 : -500, timingFastConfig);
    } else {
        return withSpring(0, springConfig);
    }
};

// Stagger animation (list items appearing)
export const staggerAnimation = (index: number, delayPerItem = 50) => {
    'worklet';
    return withSequence(
        withTiming(0, { duration: 0 }),
        withTiming(0, { duration: index * delayPerItem }),
        withSpring(1, springConfig)
    );
};

// Page transition slide in
export const pageTransitionSlideIn = (fromRight = true) => {
    'worklet';
    return withTiming(0, {
        duration: 300,
        easing: Easing.out(Easing.cubic),
    });
};

// 3D tilt animation (on scroll/interaction)
export const tiltAnimation = (value: number, maxAngle = 8) => {
    'worklet';
    // Convert value to angle within range
    const angle = (value / 100) * maxAngle;
    return withSpring(angle, springGentleConfig);
};

// ============================================
// GESTURE ANIMATIONS
// ============================================

// Pull to refresh
export const pullToRefreshAnimation = (progress: number) => {
    'worklet';
    if (progress >= 1) {
        return withSpring(1, springBouncyConfig);
    } else {
        return progress;
    }
};

// Drag and drop
export const dragAnimation = (position: { x: number; y: number }) => {
    'worklet';
    return {
        x: withSpring(position.x, springFastConfig),
        y: withSpring(position.y, springFastConfig),
    };
};

// ============================================
// LAYOUT ANIMATIONS
// ============================================

// Smooth height change (for expandable content)
export const heightAnimation = (height: number) => {
    'worklet';
    return withTiming(height, {
        duration: 250,
        easing: Easing.out(Easing.cubic),
    });
};

// Width animation (progress bars, etc.)
export const widthAnimation = (width: number) => {
    'worklet';
    return withTiming(width, {
        duration: 300,
        easing: Easing.out(Easing.cubic),
    });
};

// Opacity + Scale combo (common for modals)
export const modalAnimation = (show: boolean) => {
    'worklet';
    return {
        opacity: show ? fadeInAnimation() : fadeOutAnimation(),
        scale: show ? withSpring(1, springConfig) : withTiming(0.9, timingFastConfig),
    };
};

// ============================================
// HELPER FUNCTIONS
// ============================================

// Interpolate for smooth transitions
export const interpolate = (
    value: number,
    inputRange: number[],
    outputRange: number[]
) => {
    'worklet';
    // Simple linear interpolation
    const [inMin, inMax] = inputRange;
    const [outMin, outMax] = outputRange;

    if (value <= inMin) return outMin;
    if (value >= inMax) return outMax;

    const ratio = (value - inMin) / (inMax - inMin);
    return outMin + ratio * (outMax - outMin);
};

// Ease value in/out
export const easeInOut = (value: number) => {
    'worklet';
    return Easing.inOut(Easing.cubic)(value);
};
