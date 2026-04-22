/**
 * AIWardrobe — Unified Design Tokens
 * ====================================
 * SINGLE SOURCE OF TRUTH for all colors, typography, spacing, shadows, and animations.
 *
 * Previously spread across 4+ files:
 *   - constants/LiquidGlass2026Theme.ts  (primary design system)
 *   - constants/AppColors.ts             (flat color map)
 *   - constants/ClosetlyTheme.ts         (legacy theme)
 *   - src/theme/index.ts                 (light/dark export)
 *   - constants/AltaColors.ts            (unused)
 *
 * Now everything lives here. Old files re-export from this module for backward compatibility.
 */

import { Platform, Dimensions, PixelRatio, Appearance } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ============================================
// COLOR SYSTEM — Monochrome (Black & White)
// ============================================

/** LiquidGlass colors — structured tokens */
export const LiquidGlassColors = {
    glass: {
        clear: 'rgba(255, 255, 255, 0.12)',
        light: 'rgba(255, 255, 255, 0.25)',
        frosted: 'rgba(255, 255, 255, 0.45)',
        opaque: 'rgba(255, 255, 255, 0.85)',
        tinted: 'rgba(255, 255, 255, 0.08)',
        dark: 'rgba(0, 0, 0, 0.15)',
        darkFrosted: 'rgba(0, 0, 0, 0.35)',
    },

    background: {
        primary: '#FFFFFF',
        secondary: '#F5F5F5',
        tertiary: '#EEEEEE',
        elevated: '#FFFFFF',
    },

    text: {
        primary: '#0A1931',
        secondary: '#4D4D4D',
        tertiary: '#808080',
        disabled: '#B3B3B3',
        onGlass: '#0A1931',
        onDark: '#FFFFFF',
    },

    accent: {
        primary: '#0A1931', // Innovation Blue
        secondary: '#0A1931', // Old Primary
        tertiary: '#666666',
        success: '#1A7A4A',
        warning: '#8A5C00',
        error: '#B91C1C',
    },

    gradients: {
        liquidGlass: ['rgba(255,255,255,0.9)', 'rgba(255,255,255,0.6)'] as const,
        primaryAccent: ['#0A1931', '#333333'] as const,
        warmGlow: ['#333333', '#0A1931'] as const,
        coolWave: ['#4D4D4D', '#0A1931'] as const,
        premium: ['#0A1931', '#1A1A1A'] as const,
        dark: ['#0A1931', '#1A1A1A'] as const,
    },

    border: {
        glass: 'rgba(0, 0, 0, 0.1)',
        glassDark: 'rgba(0, 0, 0, 0.15)',
        subtle: 'rgba(0, 0, 0, 0.08)',
        strong: 'rgba(0, 0, 0, 0.2)',
    },
};

// ============================================
// LIGHT / DARK COLOR PALETTES
// (consumed by ThemeContext for runtime switching)
// ============================================

export const lightColors = {
    background: '#FFFFFF',
    surface: '#FFFFFF',
    surfaceHighlight: '#F5F5F5',
    surfaceSecondary: '#EEEEEE',

    accentCard: '#F5F5F5',
    accentCardDark: '#0A1931',

    primary: '#0A1931',
    primaryLight: '#F0F4FF',
    primaryDark: '#0A1931',

    secondary: '#4D4D4D',
    secondaryLight: '#F5F5F5',
    secondaryDark: '#333333',

    text: {
        primary: '#0A1931',
        secondary: '#4D4D4D',
        accent: '#0A1931',
        muted: '#808080',
        inverse: '#FFFFFF',
        disabled: '#B3B3B3',
    },

    border: '#E0E0E0',
    borderLight: '#F0F0F0',
    borderSubtle: 'rgba(0,0,0,0.08)',

    // Semantic colors — used across screens for status/actions
    accent: '#7B61FF',
    accentSoft: 'rgba(123,97,255,0.15)',
    success: '#30D158',
    error: '#FF453A',
    warning: '#FF9F0A',
    info: '#0A84FF',

    favorite: '#FF453A',
    delete: '#FF453A',
    edit: '#7B61FF',

    button: {
        primary: '#0A1931',
        primaryText: '#FFFFFF',
        secondary: '#F5F5F5',
        secondaryText: '#0A1931',
        ghost: 'transparent',
        ghostText: '#4D4D4D',
        cta: '#0A1931',
        ctaText: '#FFFFFF',
    },

    glass: {
        background: 'rgba(255, 255, 255, 0.9)',
        border: 'rgba(0, 0, 0, 0.1)',
        dark: 'rgba(0, 0, 0, 0.05)',
    },

    gradients: {
        primary: ['#0A1931', '#333333'],
        secondary: ['#FFFFFF', '#F5F5F5'],
        accent: ['#333333', '#0A1931'],
        warm: ['#F5F5F5', '#EEEEEE'],
        dark: ['#0A1931', '#1A1A1A'],
        hero: ['#FFFFFF', '#F5F5F5', '#EEEEEE'],
    },
};

export const darkColors = {
    background: '#0A1931',
    surface: '#0A0A0A',
    surfaceHighlight: '#1A1A1A',
    surfaceSecondary: '#0A0A0A',

    accentCard: '#1A1A1A',
    accentCardDark: '#FFFFFF',

    primary: '#FFFFFF',
    primaryLight: '#333333',
    primaryDark: '#FFFFFF',

    secondary: '#B3B3B3',
    secondaryLight: '#1A1A1A',
    secondaryDark: '#CCCCCC',

    glow: '#FFFFFF',

    text: {
        primary: '#FFFFFF',
        secondary: '#B3B3B3',
        accent: '#FFFFFF',
        muted: '#808080',
        inverse: '#0A1931',
        disabled: '#4D4D4D',
    },

    border: '#333333',
    borderLight: '#1A1A1A',
    borderSubtle: 'rgba(255,255,255,0.08)',

    // Semantic colors — used across screens for status/actions
    accent: '#9B7BFF',
    accentSoft: 'rgba(155,123,255,0.20)',
    success: '#32D74B',
    error: '#FF453A',
    warning: '#FFD60A',
    info: '#0A84FF',

    favorite: '#FF453A',
    delete: '#FF6961',
    edit: '#9B7BFF',

    button: {
        primary: '#FFFFFF',
        primaryText: '#0A1931',
        secondary: '#1A1A1A',
        secondaryText: '#FFFFFF',
        ghost: 'transparent',
        ghostText: '#B3B3B3',
        cta: '#FFFFFF',
        ctaText: '#0A1931',
    },

    glass: {
        background: 'rgba(0, 0, 0, 0.9)',
        border: 'rgba(255, 255, 255, 0.1)',
        dark: 'rgba(0, 0, 0, 0.4)',
    },

    gradients: {
        primary: ['#FFFFFF', '#CCCCCC'],
        secondary: ['#0A0A0A', '#0A1931'],
        accent: ['#CCCCCC', '#FFFFFF'],
        warm: ['#1A1A1A', '#0A0A0A'],
        purple: ['#FFFFFF', '#E0E0E0'],
        dark: ['#0A1931', '#0A0A0A'],
        hero: ['#0A1931', '#0A0A0A', '#1A1A1A'],
    },
};

// Runtime theme (defaults to system preference)
const colorScheme = Appearance.getColorScheme();
export const colors = colorScheme === 'dark' ? darkColors : lightColors;
export const getThemeColors = (isDark: boolean) => isDark ? darkColors : lightColors;

// ============================================
// FLAT APP COLORS (backward-compat for AppColors consumers)
// ============================================

export const AppColors = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    surfaceSecondary: '#FAFAFA',
    surfaceHighlight: '#EEEEEE',

    text: '#0A1931',
    textSecondary: '#4D4D4D',
    textMuted: '#808080',
    textLight: '#B3B3B3',

    primary: '#0A1931',
    accent: '#0A1931',

    success: '#0A1931',
    error: '#0A1931',
    warning: '#0A1931',
    favorite: '#0A1931',
    delete: '#0A1931',

    border: '#E0E0E0',
    borderLight: '#F0F0F0',

    glass: 'rgba(255, 255, 255, 0.9)',
    glassDark: 'rgba(0, 0, 0, 0.05)',

    premium: '#0A1931',
    premiumDark: '#333333',
    vip: '#0A1931',
    vipDark: '#1A1A1A',

    gradientPrimary: ['#0A1931', '#333333'] as const,
    gradientAccent: ['#333333', '#0A1931'] as const,
    gradientPremium: ['#0A1931', '#1A1A1A'] as const,
    gradientVIP: ['#1A1A1A', '#0A1931'] as const,
};

export const AppColorsDark = {
    background: '#0A1931',
    surface: '#1A1A1A',
    surfaceSecondary: '#0A0A0A',
    surfaceHighlight: '#2A2A2A',

    text: '#FFFFFF',
    textSecondary: '#B3B3B3',
    textMuted: '#808080',
    textLight: '#4D4D4D',

    primary: '#FFFFFF',
    accent: '#FFFFFF',

    success: '#FFFFFF',
    error: '#FFFFFF',
    warning: '#FFFFFF',
    favorite: '#FFFFFF',
    delete: '#FFFFFF',

    border: '#333333',
    borderLight: '#2A2A2A',

    glass: 'rgba(0, 0, 0, 0.9)',
    glassDark: 'rgba(255, 255, 255, 0.05)',

    premium: '#FFFFFF',
    premiumDark: '#CCCCCC',
    vip: '#FFFFFF',
    vipDark: '#E0E0E0',

    gradientPrimary: ['#FFFFFF', '#CCCCCC'] as const,
    gradientAccent: ['#CCCCCC', '#FFFFFF'] as const,
    gradientPremium: ['#FFFFFF', '#E0E0E0'] as const,
    gradientVIP: ['#E0E0E0', '#FFFFFF'] as const,
};

// ============================================
// SPATIAL ELEVATION (Z-axis depth)
// ============================================

export const SpatialElevation = {
    levels: {
        surface: 0,
        raised: 4,
        card: 8,
        floating: 16,
        modal: 24,
        popover: 32,
        overlay: 48,
        alert: 56,
    },

    getShadow: (elevation: number, isDark = false) => {
        const opacity = Math.min(0.05 + (elevation * 0.008), 0.25);
        const blur = Math.min(4 + (elevation * 1.5), 64);
        const offset = Math.min(2 + (elevation * 0.5), 24);

        return {
            shadowColor: isDark ? '#FFFFFF' : '#000000',
            shadowOffset: { width: 0, height: offset },
            shadowOpacity: opacity,
            shadowRadius: blur,
            elevation: Math.min(elevation, 24),
        };
    },
};

// ============================================
// TYPOGRAPHY
// ============================================

export const LiquidGlassTypography = {
    fontFamily: Platform.select({
        ios: 'System',
        android: 'Roboto',
        default: 'System',
    }),

    scale: {
        displayLarge: { fontSize: 44, lineHeight: 52, fontWeight: '700' as const, letterSpacing: -1.0 },
        displayMedium: { fontSize: 36, lineHeight: 44, fontWeight: '700' as const, letterSpacing: -0.5 },
        displaySmall: { fontSize: 32, lineHeight: 40, fontWeight: '600' as const, letterSpacing: -0.3 },

        headlineLarge: { fontSize: 28, lineHeight: 36, fontWeight: '600' as const, letterSpacing: -0.2 },
        headlineMedium: { fontSize: 24, lineHeight: 32, fontWeight: '600' as const, letterSpacing: -0.1 },
        headlineSmall: { fontSize: 20, lineHeight: 28, fontWeight: '600' as const, letterSpacing: 0 },

        titleLarge: { fontSize: 18, lineHeight: 26, fontWeight: '600' as const, letterSpacing: 0 },
        titleMedium: { fontSize: 16, lineHeight: 24, fontWeight: '600' as const, letterSpacing: 0.1 },
        titleSmall: { fontSize: 14, lineHeight: 20, fontWeight: '600' as const, letterSpacing: 0.1 },

        bodyLarge: { fontSize: 16, lineHeight: 24, fontWeight: '400' as const, letterSpacing: 0.15 },
        bodyMedium: { fontSize: 14, lineHeight: 20, fontWeight: '400' as const, letterSpacing: 0.2 },
        bodySmall: { fontSize: 12, lineHeight: 16, fontWeight: '400' as const, letterSpacing: 0.3 },

        labelLarge: { fontSize: 14, lineHeight: 20, fontWeight: '500' as const, letterSpacing: 0.5 },
        labelMedium: { fontSize: 12, lineHeight: 16, fontWeight: '500' as const, letterSpacing: 0.5 },
        labelSmall: { fontSize: 11, lineHeight: 14, fontWeight: '500' as const, letterSpacing: 0.6, textTransform: 'uppercase' as const },
    },

    withDynamicScale: (baseStyle: Record<string, unknown>, scale: number) => ({
        ...baseStyle,
        fontSize: (baseStyle.fontSize as number) * Math.max(0.8, Math.min(scale, 2.0)),
        lineHeight: (baseStyle.lineHeight as number) * Math.max(0.8, Math.min(scale, 2.0)),
    }),
};

/** Legacy typography shortcuts (from src/theme/index.ts) */
export const typography = {
    h1: { fontSize: 32, fontWeight: '800' as const, lineHeight: 38, letterSpacing: -0.5 },
    h2: { fontSize: 24, fontWeight: '700' as const, lineHeight: 30, letterSpacing: -0.3 },
    h3: { fontSize: 20, fontWeight: '700' as const, lineHeight: 26 },
    body: { fontSize: 16, fontWeight: '400' as const, lineHeight: 24 },
    bodySmall: { fontSize: 14, fontWeight: '400' as const, lineHeight: 20 },
    caption: { fontSize: 12, fontWeight: '500' as const, lineHeight: 16 },
    button: { fontSize: 16, fontWeight: '600' as const, letterSpacing: 0.3 },
};

// ============================================
// SPACING & LAYOUT
// ============================================

export const LiquidGlassSpacing = {
    base: 4,
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
    xxxl: 64,

    screenPadding: 20,
    cardGap: 16,
    sectionGap: 32,

    bento: { gap: 12, padding: 16, itemMinHeight: 100 },
    touchTarget: { minimum: 44, comfortable: 48, large: 56 },
};

/** Legacy spacing shortcuts (from src/theme/index.ts) */
export const spacing = {
    xs: 4,
    s: 8,
    m: 16,
    l: 24,
    xl: 32,
    xxl: 48,
    xxxl: 64,
};

// ============================================
// BORDER RADIUS
// ============================================

export const LiquidGlassRadius = {
    none: 0,
    xs: 6,
    sm: 12,
    md: 16,
    lg: 20,
    xl: 24,
    xxl: 32,
    xxxl: 44,

    card: 40,
    button: 28,
    chip: 24,
    pill: 99,
    input: 16,
    modal: 40,
    bottomSheet: 40,

    full: 9999,
};

/** Legacy border radius shortcuts */
export const borderRadius = {
    s: 12,
    m: 16,
    l: 24,
    xl: 32,
    xxl: 40,
    full: 9999,
};

// ============================================
// SHADOWS
// ============================================

export const shadows = {
    soft: { shadowColor: '#000000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.03, shadowRadius: 12, elevation: 2 },
    medium: { shadowColor: '#000000', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.05, shadowRadius: 24, elevation: 4 },
    strong: { shadowColor: '#000000', shadowOffset: { width: 0, height: 16 }, shadowOpacity: 0.08, shadowRadius: 36, elevation: 8 },
    card: { shadowColor: '#000000', shadowOffset: { width: 0, height: 12 }, shadowOpacity: 0.06, shadowRadius: 24, elevation: 6 },
    glow: { shadowColor: '#0A1931', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.15, shadowRadius: 24, elevation: 8 },
    /** ClosetlyTheme-compatible aliases */
    cardSmall: { shadowColor: '#000000', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.04, shadowRadius: 16, elevation: 4 },
    button: { shadowColor: '#0A1931', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.12, shadowRadius: 16, elevation: 6 },
    floating: { shadowColor: '#0A1931', shadowOffset: { width: 0, height: 16 }, shadowOpacity: 0.15, shadowRadius: 36, elevation: 12 },
};

// ============================================
// ANIMATION
// ============================================

export const LiquidGlassAnimation = {
    duration: { instant: 50, fast: 150, normal: 250, slow: 400, emphasis: 600 },

    spring: {
        snappy: { damping: 20, stiffness: 300, mass: 0.8 },
        bouncy: { damping: 12, stiffness: 150, mass: 1 },
        smooth: { damping: 25, stiffness: 120, mass: 1.2 },
        gentle: { damping: 30, stiffness: 80, mass: 1.5 },
    },

    easing: {
        standard: [0.4, 0.0, 0.2, 1],
        decelerate: [0.0, 0.0, 0.2, 1],
        accelerate: [0.4, 0.0, 1, 1],
        sharp: [0.4, 0.0, 0.6, 1],
    },

    reducedMotion: { duration: 0, spring: { damping: 100, stiffness: 1000, mass: 1 } },
};

/** Legacy animations (from src/theme/index.ts) */
export const animations = {
    spring: { damping: 20, stiffness: 200, mass: 1 },
    springFast: { damping: 25, stiffness: 350, mass: 0.8 },
    springBouncy: { damping: 12, stiffness: 120, mass: 1 },
    timing: { fast: 150, normal: 250, slow: 400 },
    scale: { pressed: 0.97, normal: 1 },
    float: { amplitude: 3, duration: 3000 },
    tilt: { maxAngle: 8, perspective: 1000 },
    glow: { minOpacity: 0.2, maxOpacity: 0.5, duration: 2000 },
    orbit: { radius: 60, duration: 4000 },
    fadeIn: { duration: 600, delay: 50 },
};

// ============================================
// BLUR
// ============================================

export const LiquidGlassBlur = {
    intensity: { subtle: 20, light: 40, medium: 60, heavy: 80, extreme: 100 },
    tint: { light: 'light' as const, dark: 'dark' as const, default: 'default' as const, chromatic: 'chromaBlur' as const },
    getDynamicBlur: (scrollOffset: number, maxScroll: number = 100) => {
        const progress = Math.min(scrollOffset / maxScroll, 1);
        return {
            intensity: 20 + (progress * 60),
            backgroundColor: `rgba(255, 255, 255, ${0.6 + (progress * 0.25)})`,
        };
    },
};

// ============================================
// ACCESSIBILITY
// ============================================

export const AccessibilityOverrides = {
    highContrast: {
        glass: { clear: 'rgba(255, 255, 255, 0.95)', frosted: '#FFFFFF', border: '#000000' },
        text: { primary: '#000000', secondary: '#333333' },
        shadows: {
            getShadow: () => ({ borderWidth: 2, borderColor: '#000000', shadowOpacity: 0 }),
        },
    },
    reducedMotion: { duration: 0, spring: { damping: 1000, stiffness: 1000, mass: 1 } },
    dynamicType: { minimumScale: 0.8, maximumScale: 2.5 },
};

// ============================================
// BENTO GRID
// ============================================

export const BentoGridConfig = {
    columns: { compact: 2, regular: 3, expanded: 4 },
    getColumns: (width: number = SCREEN_WIDTH): number => {
        if (width < 400) return 2;
        if (width < 600) return 3;
        return 4;
    },
    itemSizes: {
        small: { colSpan: 1, rowSpan: 1 },
        medium: { colSpan: 2, rowSpan: 1 },
        large: { colSpan: 2, rowSpan: 2 },
        wide: { colSpan: 3, rowSpan: 1 },
        tall: { colSpan: 1, rowSpan: 2 },
        hero: { colSpan: 4, rowSpan: 2 },
    },
    getItemDimensions: (
        columns: number,
        colSpan: number,
        gap: number = LiquidGlassSpacing.bento.gap,
        padding: number = LiquidGlassSpacing.screenPadding
    ) => {
        const availableWidth = SCREEN_WIDTH - (padding * 2);
        const totalGaps = (columns - 1) * gap;
        const cellWidth = (availableWidth - totalGaps) / columns;
        return {
            width: (cellWidth * colSpan) + ((colSpan - 1) * gap),
            minHeight: cellWidth,
        };
    },
};

// ============================================
// HAPTICS
// ============================================

export const haptics = {
    light: 'light',
    medium: 'medium',
    heavy: 'heavy',
    success: 'success',
    warning: 'warning',
    error: 'error',
};

// ============================================
// CLOSETLY COMPAT — legacy ClosetlyTheme shape
// ============================================

export const ClosetlyThemeCompat = {
    colors: {
        background: '#FFFFFF',
        card: '#F5F5F5',
        cardHover: '#EEEEEE',
        text: '#000000',
        textSecondary: '#4D4D4D',
        textMuted: '#808080',
        overlay: 'rgba(255, 255, 255, 0.95)',
        glassBg: 'rgba(255, 255, 255, 0.85)',
        success: '#000000',
        matchHighlight: '#000000',
    },
    spacing: { xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48, cardGap: 16, screenPadding: 20, sectionGap: 24 },
    borderRadius: { sm: 12, md: 20, lg: 24, xl: 32, card: 32, button: 24, pill: 50 },
    shadows,
    typography: {
        hero: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 36, fontWeight: '700' as const, letterSpacing: -0.5, color: '#000000' },
        header: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 32, fontWeight: '700' as const, letterSpacing: -0.3, color: '#000000' },
        title: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 24, fontWeight: '600' as const, letterSpacing: -0.2, color: '#000000' },
        subtitle: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 18, fontWeight: '600' as const, color: '#000000' },
        body: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 16, fontWeight: '400' as const, color: '#666666' },
        label: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 12, fontWeight: '500' as const, letterSpacing: 0.5, textTransform: 'uppercase' as const, color: '#999999' },
        caption: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 11, fontWeight: '500' as const, color: '#999999' },
        matchScore: { fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif', fontSize: 14, fontWeight: '700' as const, color: '#000000' },
    },
    animation: { fast: 150, normal: 250, slow: 400, spring: { damping: 15, stiffness: 150, mass: 1 } },
    cardDimensions: {
        carouselItem: { width: 140, height: 180 },
        carouselItemLarge: { width: 160, height: 200 },
        modelView: { width: 200, height: 280 },
    },
};

// ============================================
// COMBINED THEME EXPORT (primary API)
// ============================================

export const LiquidGlass2026Theme = {
    colors: LiquidGlassColors,
    elevation: SpatialElevation,
    typography: LiquidGlassTypography,
    spacing: LiquidGlassSpacing,
    radius: LiquidGlassRadius,
    animation: LiquidGlassAnimation,
    blur: LiquidGlassBlur,
    accessibility: AccessibilityOverrides,
    bento: BentoGridConfig,

    screen: {
        width: SCREEN_WIDTH,
        height: SCREEN_HEIGHT,
        pixelRatio: PixelRatio.get(),
    },

    platform: Platform.OS,
    isIOS: Platform.OS === 'ios',
    isAndroid: Platform.OS === 'android',
};

export default LiquidGlass2026Theme;
