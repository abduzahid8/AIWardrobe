// AIWardrobe Premium Theme - Wardrobe 2.0 + Alta Daily Inspired
import { Appearance } from 'react-native';

// ============================================
// LIGHT THEME - Clean, airy Wardrobe 2.0 style
// ============================================
export const lightColors = {
    // Backgrounds - Light blue tint like Wardrobe 2.0
    background: "#F8FAFC",
    surface: "#FFFFFF",
    surfaceHighlight: "#F1F5F9",
    surfaceSecondary: "#E2E8F0",

    // Accent Color Card
    accentCard: "#FEF3E7",
    accentCardDark: "#E4A076",

    // Primary accent - Soft blue
    primary: "#3B82F6",
    primaryLight: "#DBEAFE",
    primaryDark: "#1D4ED8",

    // Secondary accent - Warm orange (Wardrobe 2.0 style)
    secondary: "#E4A076",
    secondaryLight: "#FEF3E7",
    secondaryDark: "#D18A5A",

    // Text
    text: {
        primary: "#1E293B",
        secondary: "#64748B",
        accent: "#3B82F6",
        muted: "#94A3B8",
        inverse: "#FFFFFF",
        disabled: "#CBD5E1",
    },

    // Borders
    border: "#E2E8F0",
    borderLight: "#F1F5F9",

    // Status
    success: "#22C55E",
    error: "#EF4444",
    warning: "#F59E0B",
    info: "#3B82F6",

    // Actions
    favorite: "#EF4444",
    delete: "#EF4444",
    edit: "#3B82F6",

    // Buttons - Wardrobe 2.0 style
    button: {
        primary: "#3B82F6",
        primaryText: "#FFFFFF",
        secondary: "#F1F5F9",
        secondaryText: "#1E293B",
        ghost: "transparent",
        ghostText: "#64748B",
        cta: "#E4A076",
        ctaText: "#FFFFFF",
    },

    // Glass
    glass: {
        background: "rgba(255, 255, 255, 0.9)",
        border: "rgba(255, 255, 255, 0.5)",
        dark: "rgba(0, 0, 0, 0.05)",
    },

    // Gradients
    gradients: {
        primary: ["#3B82F6", "#60A5FA"],
        secondary: ["#F8FAFC", "#DBEAFE"],
        accent: ["#E4A076", "#F4B494"],
        warm: ["#FEF3E7", "#DBEAFE"],
        dark: ["#1E293B", "#334155"],
        hero: ["#DBEAFE", "#F8FAFC", "#FFFFFF"],
    },
};

// ============================================
// DARK THEME - Premium Alta Daily inspired
// ============================================
export const darkColors = {
    // Backgrounds - Deep navy-black (Alta Daily + DiWander inspired)
    background: "#0A0F1E",
    surface: "#1A2032",
    surfaceHighlight: "#252D45",
    surfaceSecondary: "#1A2032",

    // Accent Color Card
    accentCard: "#1E2A47",
    accentCardDark: "#60A5FA",

    // Primary accent - Bright blue
    primary: "#60A5FA",
    primaryLight: "#1E3A5F",
    primaryDark: "#93C5FD",

    // Secondary accent - Purple (premium feel)
    secondary: "#8B5CF6",
    secondaryLight: "#2E1A47",
    secondaryDark: "#A78BFA",

    // Glow accent - Indigo
    glow: "#6366F1",

    // Text
    text: {
        primary: "#F1F5F9",
        secondary: "#94A3B8",
        accent: "#60A5FA",
        muted: "#64748B",
        inverse: "#0A0F1E",
        disabled: "#475569",
    },

    // Borders
    border: "#252D45",
    borderLight: "#1A2032",

    // Status
    success: "#4ADE80",
    error: "#F87171",
    warning: "#FBBF24",
    info: "#60A5FA",

    // Actions
    favorite: "#F87171",
    delete: "#F87171",
    edit: "#60A5FA",

    // Buttons
    button: {
        primary: "#60A5FA",
        primaryText: "#0A0F1E",
        secondary: "#252D45",
        secondaryText: "#F1F5F9",
        ghost: "transparent",
        ghostText: "#94A3B8",
        cta: "#8B5CF6",
        ctaText: "#FFFFFF",
    },

    // Glass
    glass: {
        background: "rgba(26, 32, 50, 0.9)",
        border: "rgba(255, 255, 255, 0.1)",
        dark: "rgba(0, 0, 0, 0.4)",
    },

    // Gradients
    gradients: {
        primary: ["#60A5FA", "#3B82F6"],
        secondary: ["#1A2032", "#0A0F1E"],
        accent: ["#8B5CF6", "#A78BFA"],
        warm: ["#2E1A47", "#1E2A47"],
        purple: ["#6366F1", "#8B5CF6"],
        dark: ["#0A0F1E", "#1A2032"],
        hero: ["#0A0F1E", "#1A2032", "#252D45"],
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
// SHADOWS
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
        shadowColor: "#8B5CF6",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 12,
        elevation: 6,
    },
};

// ============================================
// ANIMATIONS - Alta Daily inspired
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
    // Alta-style 3D effects
    float: {
        amplitude: 3,       // pixels to float up/down
        duration: 3000,     // ms for full cycle
    },
    tilt: {
        maxAngle: 8,        // degrees of max tilt
        perspective: 1000,  // perspective depth
    },
    glow: {
        minOpacity: 0.2,
        maxOpacity: 0.5,
        duration: 2000,
    },
    orbit: {
        radius: 60,         // px radius
        duration: 4000,     // ms for full rotation
    },
    fadeIn: {
        duration: 600,
        delay: 50,          // stagger delay per item
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
