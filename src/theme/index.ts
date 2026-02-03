// AIWardrobe Monochrome Theme - Pure Black and White Only
import { Appearance } from 'react-native';

// ============================================
// LIGHT THEME - Pure Black and White
// ============================================
export const lightColors = {
    // Backgrounds - Pure White
    background: "#FFFFFF",
    surface: "#FFFFFF",
    surfaceHighlight: "#F5F5F5",
    surfaceSecondary: "#EEEEEE",

    // Accent Color Card - Monochrome
    accentCard: "#F5F5F5",
    accentCardDark: "#000000",

    // Primary accent - Pure Black
    primary: "#000000",
    primaryLight: "#E0E0E0",
    primaryDark: "#000000",

    // Secondary accent - Gray
    secondary: "#4D4D4D",
    secondaryLight: "#F5F5F5",
    secondaryDark: "#333333",

    // Text - Pure Black/White
    text: {
        primary: "#000000",
        secondary: "#4D4D4D",
        accent: "#000000",
        muted: "#808080",
        inverse: "#FFFFFF",
        disabled: "#B3B3B3",
    },

    // Borders - Monochrome
    border: "#E0E0E0",
    borderLight: "#F0F0F0",

    // Status - All Black (Monochrome)
    success: "#000000",
    error: "#000000",
    warning: "#000000",
    info: "#000000",

    // Actions - Monochrome
    favorite: "#000000",
    delete: "#000000",
    edit: "#000000",

    // Buttons - Monochrome
    button: {
        primary: "#000000",
        primaryText: "#FFFFFF",
        secondary: "#F5F5F5",
        secondaryText: "#000000",
        ghost: "transparent",
        ghostText: "#4D4D4D",
        cta: "#000000",
        ctaText: "#FFFFFF",
    },

    // Glass - Monochrome
    glass: {
        background: "rgba(255, 255, 255, 0.9)",
        border: "rgba(0, 0, 0, 0.1)",
        dark: "rgba(0, 0, 0, 0.05)",
    },

    // Gradients - Black to Gray
    gradients: {
        primary: ["#000000", "#333333"],
        secondary: ["#FFFFFF", "#F5F5F5"],
        accent: ["#333333", "#000000"],
        warm: ["#F5F5F5", "#EEEEEE"],
        dark: ["#000000", "#1A1A1A"],
        hero: ["#FFFFFF", "#F5F5F5", "#EEEEEE"],
    },
};

// ============================================
// DARK THEME - Inverted Black and White
// ============================================
export const darkColors = {
    // Backgrounds - Pure Black
    background: "#000000",
    surface: "#0A0A0A",
    surfaceHighlight: "#1A1A1A",
    surfaceSecondary: "#0A0A0A",

    // Accent Color Card - Monochrome
    accentCard: "#1A1A1A",
    accentCardDark: "#FFFFFF",

    // Primary accent - Pure White
    primary: "#FFFFFF",
    primaryLight: "#333333",
    primaryDark: "#FFFFFF",

    // Secondary accent - Gray
    secondary: "#B3B3B3",
    secondaryLight: "#1A1A1A",
    secondaryDark: "#CCCCCC",

    // Glow accent - White
    glow: "#FFFFFF",

    // Text - Pure White/Black
    text: {
        primary: "#FFFFFF",
        secondary: "#B3B3B3",
        accent: "#FFFFFF",
        muted: "#808080",
        inverse: "#000000",
        disabled: "#4D4D4D",
    },

    // Borders - Monochrome
    border: "#333333",
    borderLight: "#1A1A1A",

    // Status - All White (Monochrome)
    success: "#FFFFFF",
    error: "#FFFFFF",
    warning: "#FFFFFF",
    info: "#FFFFFF",

    // Actions - Monochrome
    favorite: "#FFFFFF",
    delete: "#FFFFFF",
    edit: "#FFFFFF",

    // Buttons - Monochrome
    button: {
        primary: "#FFFFFF",
        primaryText: "#000000",
        secondary: "#1A1A1A",
        secondaryText: "#FFFFFF",
        ghost: "transparent",
        ghostText: "#B3B3B3",
        cta: "#FFFFFF",
        ctaText: "#000000",
    },

    // Glass - Monochrome
    glass: {
        background: "rgba(0, 0, 0, 0.9)",
        border: "rgba(255, 255, 255, 0.1)",
        dark: "rgba(0, 0, 0, 0.4)",
    },

    // Gradients - White to Gray
    gradients: {
        primary: ["#FFFFFF", "#CCCCCC"],
        secondary: ["#0A0A0A", "#000000"],
        accent: ["#CCCCCC", "#FFFFFF"],
        warm: ["#1A1A1A", "#0A0A0A"],
        purple: ["#FFFFFF", "#E0E0E0"],
        dark: ["#000000", "#0A0A0A"],
        hero: ["#000000", "#0A0A0A", "#1A1A1A"],
    },
};

// ============================================
// ACTIVE THEME (defaults to system preference)
// ============================================
const colorScheme = Appearance.getColorScheme();
export const colors = colorScheme === 'dark' ? darkColors : lightColors;

// Helper to get theme colors
export const getThemeColors = (isDark: boolean) => isDark ? darkColors : lightColors;

// ============================================
// TYPOGRAPHY
// ============================================
export const typography = {
    h1: {
        fontSize: 32,
        fontWeight: "800" as const,
        lineHeight: 38,
        letterSpacing: -0.5,
    },
    h2: {
        fontSize: 24,
        fontWeight: "700" as const,
        lineHeight: 30,
        letterSpacing: -0.3,
    },
    h3: {
        fontSize: 20,
        fontWeight: "700" as const,
        lineHeight: 26,
    },
    body: {
        fontSize: 16,
        fontWeight: "400" as const,
        lineHeight: 24,
    },
    bodySmall: {
        fontSize: 14,
        fontWeight: "400" as const,
        lineHeight: 20,
    },
    caption: {
        fontSize: 12,
        fontWeight: "500" as const,
        lineHeight: 16,
    },
    button: {
        fontSize: 16,
        fontWeight: "600" as const,
        letterSpacing: 0.3,
    },
};

// ============================================
// SPACING
// ============================================
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
export const borderRadius = {
    s: 8,
    m: 12,
    l: 16,
    xl: 20,
    xxl: 24,
    full: 9999,
};

// ============================================
// SHADOWS - Monochrome (Black only)
// ============================================
export const shadows = {
    soft: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.04,
        shadowRadius: 8,
        elevation: 2,
    },
    medium: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
    },
    strong: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.08,
        shadowRadius: 24,
        elevation: 8,
    },
    card: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.03,
        shadowRadius: 4,
        elevation: 1,
    },
    glow: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 12,
        elevation: 6,
    },
};

// ============================================
// ANIMATIONS
// ============================================
export const animations = {
    // Spring physics
    spring: {
        damping: 20,
        stiffness: 200,
        mass: 1,
    },
    springFast: {
        damping: 25,
        stiffness: 350,
        mass: 0.8,
    },
    springBouncy: {
        damping: 12,
        stiffness: 120,
        mass: 1,
    },
    // Timing
    timing: {
        fast: 150,
        normal: 250,
        slow: 400,
    },
    // Scale
    scale: {
        pressed: 0.97,
        normal: 1,
    },
    // Float effects
    float: {
        amplitude: 3,
        duration: 3000,
    },
    tilt: {
        maxAngle: 8,
        perspective: 1000,
    },
    glow: {
        minOpacity: 0.2,
        maxOpacity: 0.5,
        duration: 2000,
    },
    orbit: {
        radius: 60,
        duration: 4000,
    },
    fadeIn: {
        duration: 600,
        delay: 50,
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
