/**
 * useAccessibility Hook
 * WCAG 3.0 compliance utilities for React Native
 * Respects system preferences for reduced motion, high contrast, and dynamic type
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { AccessibilityInfo, useColorScheme, Platform, Dimensions } from 'react-native';
import { LiquidGlass2026Theme, AccessibilityOverrides } from '../constants/LiquidGlass2026Theme';

// Types for accessibility state
interface AccessibilityState {
    // System preferences
    isReducedMotionEnabled: boolean;
    isHighContrastEnabled: boolean;
    isScreenReaderEnabled: boolean;
    isBoldTextEnabled: boolean;

    // Dynamic type scale (1.0 = default)
    dynamicTypeScale: number;

    // Derived preferences
    prefersReducedTransparency: boolean;
    colorScheme: 'light' | 'dark';
}

interface AccessibilityHelpers {
    // Get motion-safe animation config
    getAnimationConfig: <T extends object>(config: T) => T;

    // Get accessible color for glass effects
    getGlassColor: (glassKey: keyof typeof LiquidGlass2026Theme.colors.glass) => string;

    // Get shadow or border based on contrast preference
    getElevationStyle: (elevation: number) => object;

    // Scale font size with dynamic type
    scaleFontSize: (baseFontSize: number) => number;

    // Get minimum touch target size
    getMinTouchTarget: () => number;

    // Announce to screen reader
    announceForAccessibility: (message: string) => void;
}

export interface UseAccessibilityReturn extends AccessibilityState, AccessibilityHelpers { }

/**
 * Hook for managing accessibility preferences and providing helper utilities
 */
export const useAccessibility = (): UseAccessibilityReturn => {
    // State for accessibility preferences
    const [isReducedMotionEnabled, setReducedMotion] = useState(false);
    const [isScreenReaderEnabled, setScreenReader] = useState(false);
    const [isBoldTextEnabled, setBoldText] = useState(false);
    const [isHighContrastEnabled, setHighContrast] = useState(false);
    const [dynamicTypeScale, setDynamicTypeScale] = useState(1.0);

    // Color scheme from system
    const colorScheme = useColorScheme() ?? 'light';

    // Derived: Reduced transparency (maps from high contrast on iOS)
    const prefersReducedTransparency = isHighContrastEnabled;

    // Subscribe to accessibility changes
    useEffect(() => {
        // Check initial states
        const checkInitialStates = async () => {
            try {
                const [reduceMotion, screenReader, boldText] = await Promise.all([
                    AccessibilityInfo.isReduceMotionEnabled(),
                    AccessibilityInfo.isScreenReaderEnabled(),
                    Platform.OS === 'ios' ? AccessibilityInfo.isBoldTextEnabled() : Promise.resolve(false),
                ]);

                setReducedMotion(reduceMotion);
                setScreenReader(screenReader);
                setBoldText(boldText);

                // High contrast is approximated by bold text on iOS
                // On Android, we'd check system settings differently
                if (Platform.OS === 'ios') {
                    setHighContrast(boldText);
                }
            } catch (error) {
                console.warn('Failed to check accessibility settings:', error);
            }
        };

        checkInitialStates();

        // Subscribe to changes
        const reduceMotionSubscription = AccessibilityInfo.addEventListener(
            'reduceMotionChanged',
            setReducedMotion
        );

        const screenReaderSubscription = AccessibilityInfo.addEventListener(
            'screenReaderChanged',
            setScreenReader
        );

        const boldTextSubscription = Platform.OS === 'ios'
            ? AccessibilityInfo.addEventListener('boldTextChanged', (enabled) => {
                setBoldText(enabled);
                setHighContrast(enabled);
            })
            : null;

        return () => {
            reduceMotionSubscription.remove();
            screenReaderSubscription.remove();
            boldTextSubscription?.remove();
        };
    }, []);

    // Listen for font scale changes (Dynamic Type)
    useEffect(() => {
        const updateFontScale = () => {
            const { fontScale } = Dimensions.get('window');
            setDynamicTypeScale(
                Math.max(
                    AccessibilityOverrides.dynamicType.minimumScale,
                    Math.min(fontScale, AccessibilityOverrides.dynamicType.maximumScale)
                )
            );
        };

        updateFontScale();

        const subscription = Dimensions.addEventListener('change', updateFontScale);
        return () => subscription.remove();
    }, []);

    // Helper: Get motion-safe animation config
    const getAnimationConfig = useCallback(<T extends object>(config: T): T => {
        if (isReducedMotionEnabled) {
            return {
                ...config,
                ...AccessibilityOverrides.reducedMotion,
            } as T;
        }
        return config;
    }, [isReducedMotionEnabled]);

    // Helper: Get accessible glass color
    const getGlassColor = useCallback((
        glassKey: keyof typeof LiquidGlass2026Theme.colors.glass
    ): string => {
        if (isHighContrastEnabled && glassKey in AccessibilityOverrides.highContrast.glass) {
            return (AccessibilityOverrides.highContrast.glass as Record<string, string>)[glassKey]
                ?? LiquidGlass2026Theme.colors.glass[glassKey];
        }
        return LiquidGlass2026Theme.colors.glass[glassKey];
    }, [isHighContrastEnabled]);

    // Helper: Get elevation style (shadow or border)
    const getElevationStyle = useCallback((elevation: number): object => {
        if (isHighContrastEnabled) {
            return AccessibilityOverrides.highContrast.shadows.getShadow();
        }
        return LiquidGlass2026Theme.elevation.getShadow(elevation, colorScheme === 'dark');
    }, [isHighContrastEnabled, colorScheme]);

    // Helper: Scale font size with dynamic type
    const scaleFontSize = useCallback((baseFontSize: number): number => {
        return Math.round(baseFontSize * dynamicTypeScale);
    }, [dynamicTypeScale]);

    // Helper: Get minimum touch target
    const getMinTouchTarget = useCallback((): number => {
        // Larger targets for motor impairments or when screen reader is active
        if (isScreenReaderEnabled) {
            return LiquidGlass2026Theme.spacing.touchTarget.large;
        }
        return LiquidGlass2026Theme.spacing.touchTarget.minimum;
    }, [isScreenReaderEnabled]);

    // Helper: Announce to screen reader
    const announceForAccessibility = useCallback((message: string): void => {
        AccessibilityInfo.announceForAccessibility(message);
    }, []);

    // Memoize return value
    return useMemo(() => ({
        // State
        isReducedMotionEnabled,
        isHighContrastEnabled,
        isScreenReaderEnabled,
        isBoldTextEnabled,
        dynamicTypeScale,
        prefersReducedTransparency,
        colorScheme,

        // Helpers
        getAnimationConfig,
        getGlassColor,
        getElevationStyle,
        scaleFontSize,
        getMinTouchTarget,
        announceForAccessibility,
    }), [
        isReducedMotionEnabled,
        isHighContrastEnabled,
        isScreenReaderEnabled,
        isBoldTextEnabled,
        dynamicTypeScale,
        prefersReducedTransparency,
        colorScheme,
        getAnimationConfig,
        getGlassColor,
        getElevationStyle,
        scaleFontSize,
        getMinTouchTarget,
        announceForAccessibility,
    ]);
};

/**
 * Simple hook just for reduced motion preference
 */
export const useReducedMotion = (): boolean => {
    const [isEnabled, setEnabled] = useState(false);

    useEffect(() => {
        AccessibilityInfo.isReduceMotionEnabled().then(setEnabled);

        const subscription = AccessibilityInfo.addEventListener(
            'reduceMotionChanged',
            setEnabled
        );

        return () => subscription.remove();
    }, []);

    return isEnabled;
};

/**
 * Simple hook for screen reader status
 */
export const useScreenReader = (): boolean => {
    const [isEnabled, setEnabled] = useState(false);

    useEffect(() => {
        AccessibilityInfo.isScreenReaderEnabled().then(setEnabled);

        const subscription = AccessibilityInfo.addEventListener(
            'screenReaderChanged',
            setEnabled
        );

        return () => subscription.remove();
    }, []);

    return isEnabled;
};

export default useAccessibility;
