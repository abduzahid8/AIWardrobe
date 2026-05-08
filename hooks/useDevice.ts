/**
 * useDevice hook - iPad and tablet detection utilities
 * Provides responsive breakpoints and device type detection for adaptive layouts
 */

import { useMemo } from 'react';
import { useWindowDimensions, Platform, ScaledSize, Dimensions } from 'react-native';

// Breakpoints based on Apple device sizes and common tablet dimensions
export const Breakpoints = {
    // Phone sizes
    phoneSmall: 320,   // iPhone SE
    phoneMedium: 375,  // iPhone 14/15
    phoneLarge: 428,   // iPhone 14/15 Pro Max

    // Tablet sizes  
    tabletSmall: 744,  // iPad mini
    tabletMedium: 810, // iPad Air (portrait)
    tabletLarge: 1024, // iPad Pro 11"
    tabletXLarge: 1366, // iPad Pro 12.9"

    // Common breakpoints
    sm: 640,
    md: 768,
    lg: 1024,
    xl: 1280,
} as const;

export type DeviceType = 'phone' | 'tablet' | 'desktop';
export type Orientation = 'portrait' | 'landscape';

export interface DeviceInfo {
    // Device type detection
    isPhone: boolean;
    isTablet: boolean;
    isPad: boolean; // iOS specific
    isAndroidTablet: boolean;
    deviceType: DeviceType;

    // Orientation
    orientation: Orientation;
    isPortrait: boolean;
    isLandscape: boolean;

    // Screen dimensions
    width: number;
    height: number;

    // Breakpoint checks
    isSmallScreen: boolean;  // < 375
    isMediumScreen: boolean; // 375-428
    isLargeScreen: boolean;  // > 428
    isCompactWidth: boolean; // < 768 (phone or split-view iPad)
    isRegularWidth: boolean; // >= 768 (full iPad or large screen)

    // iPad-specific
    isIpadMultitasking: boolean; // iPad in split view or slide over

    // Size class approximations (similar to iOS size classes)
    horizontalSizeClass: 'compact' | 'regular';
    verticalSizeClass: 'compact' | 'regular';
}

/**
 * Detects if the device is an iPad based on screen dimensions and platform
 * iPads have a minimum width of 744pt (iPad mini) in portrait
 */
function detectTablet(width: number, height: number): { isTablet: boolean; isPad: boolean; isAndroidTablet: boolean } {
    const isPad = Platform.OS === 'ios' && Math.min(width, height) >= 744;
    const isAndroidTablet = Platform.OS === 'android' && Math.min(width, height) >= 600;
    const isTablet = isPad || isAndroidTablet;

    return { isTablet, isPad, isAndroidTablet };
}

/**
 * Hook for detecting device type and responsive breakpoints
 * Automatically updates when screen size changes (rotation, multitasking)
 */
export function useDevice(): DeviceInfo {
    const { width, height } = useWindowDimensions();

    return useMemo((): DeviceInfo => {
        const { isTablet, isPad, isAndroidTablet } = detectTablet(width, height);
        const isPhone = !isTablet;

        const isPortrait = height >= width;
        const isLandscape = width > height;
        const orientation: Orientation = isPortrait ? 'portrait' : 'landscape';

        // Size classes (similar to iOS)
        const horizontalSizeClass: 'compact' | 'regular' = width >= Breakpoints.md ? 'regular' : 'compact';
        const verticalSizeClass: 'compact' | 'regular' = height >= Breakpoints.md ? 'regular' : 'compact';

        // iPad multitasking detection (split view or slide over)
        // In multitasking, iPad width can be as small as 320pt (iPhone SE size)
        const isIpadMultitasking = isPad && width < Breakpoints.tabletSmall;

        let deviceType: DeviceType;
        if (isTablet) {
            deviceType = 'tablet';
        } else if (width >= Breakpoints.lg) {
            deviceType = 'desktop';
        } else {
            deviceType = 'phone';
        }

        return {
            isPhone,
            isTablet,
            isPad,
            isAndroidTablet,
            deviceType,

            orientation,
            isPortrait,
            isLandscape,

            width,
            height,

            isSmallScreen: width < Breakpoints.phoneMedium,
            isMediumScreen: width >= Breakpoints.phoneMedium && width < Breakpoints.phoneLarge,
            isLargeScreen: width >= Breakpoints.phoneLarge,
            isCompactWidth: width < Breakpoints.md,
            isRegularWidth: width >= Breakpoints.md,

            isIpadMultitasking,

            horizontalSizeClass,
            verticalSizeClass,
        };
    }, [width, height]);
}

/**
 * Standalone function for detecting device type outside of React components
 * Uses Dimensions API instead of useWindowDimensions
 */
export function getDeviceInfo(): DeviceInfo {
    const { width, height } = Dimensions.get('window');
    const { isTablet, isPad, isAndroidTablet } = detectTablet(width, height);
    const isPhone = !isTablet;

    const isPortrait = height >= width;
    const isLandscape = width > height;

    const horizontalSizeClass: 'compact' | 'regular' = width >= Breakpoints.md ? 'regular' : 'compact';
    const verticalSizeClass: 'compact' | 'regular' = height >= Breakpoints.md ? 'regular' : 'compact';

    const isIpadMultitasking = isPad && width < Breakpoints.tabletSmall;

    let deviceType: DeviceType;
    if (isTablet) {
        deviceType = 'tablet';
    } else if (width >= Breakpoints.lg) {
        deviceType = 'desktop';
    } else {
        deviceType = 'phone';
    }

    return {
        isPhone,
        isTablet,
        isPad,
        isAndroidTablet,
        deviceType,

        orientation: isPortrait ? 'portrait' : 'landscape',
        isPortrait,
        isLandscape,

        width,
        height,

        isSmallScreen: width < Breakpoints.phoneMedium,
        isMediumScreen: width >= Breakpoints.phoneMedium && width < Breakpoints.phoneLarge,
        isLargeScreen: width >= Breakpoints.phoneLarge,
        isCompactWidth: width < Breakpoints.md,
        isRegularWidth: width >= Breakpoints.md,

        isIpadMultitasking,

        horizontalSizeClass,
        verticalSizeClass,
    };
}

/**
 * Hook that returns responsive value based on screen width
 * Usage: const columns = useResponsive({ phone: 2, tablet: 3, desktop: 4 });
 */
export function useResponsive<T>(values: { phone?: T; tablet?: T; desktop?: T; default: T }): T {
    const { deviceType } = useDevice();

    return useMemo(() => {
        if (deviceType === 'tablet' && values.tablet !== undefined) {
            return values.tablet;
        }
        if (deviceType === 'desktop' && values.desktop !== undefined) {
            return values.desktop;
        }
        if (deviceType === 'phone' && values.phone !== undefined) {
            return values.phone;
        }
        return values.default;
    }, [deviceType, values]);
}

/**
 * Hook for responsive column count (useful for grids)
 * Automatically adjusts columns based on screen width
 */
export function useResponsiveColumns(config?: {
    phone?: number;
    tabletPortrait?: number;
    tabletLandscape?: number;
    minColumnWidth?: number;
}): number {
    const { width, height, isTablet, isPortrait } = useDevice();

    const {
        phone = 2,
        tabletPortrait = 3,
        tabletLandscape = 4,
        minColumnWidth = 160,
    } = config || {};

    return useMemo(() => {
        if (!isTablet) {
            return phone;
        }

        if (isPortrait) {
            return tabletPortrait;
        }

        // For landscape, calculate based on min column width
        const calculatedColumns = Math.floor(width / minColumnWidth);
        return Math.max(tabletLandscape, calculatedColumns);
    }, [width, height, isTablet, isPortrait, phone, tabletPortrait, tabletLandscape, minColumnWidth]);
}

export default useDevice;
