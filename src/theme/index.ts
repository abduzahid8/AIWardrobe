// iOS 25-style theme with smooth animations
export const colors = {
    background: "#FDFCF8", // Cream
    surface: "#FFFFFF",
    surfaceHighlight: "#F7F7F5", // Stone-ish
    text: {
        primary: "#2D2D2D", // Charcoal
        secondary: "#6B6B6B", // Grey
        accent: "#C4A484", // Light Brown/Tan
    },
    border: "#E5E5E0",
    success: "#4A5D4F", // Muted Green
    error: "#8B4A4A", // Muted Red

    // New accent colors for actions
    favorite: "#FF3B5C", // Heart red
    delete: "#FF3B30", // iOS destructive red
    edit: "#007AFF", // iOS blue

    // Glassmorphism
    glass: {
        background: "rgba(255, 255, 255, 0.7)",
        border: "rgba(255, 255, 255, 0.3)",
    },
};

export const typography = {
    header: {
        fontFamily: "System", // Ideally Playfair Display, but using System Serif fallback
        fontWeight: "700",
    },
    body: {
        fontFamily: "System",
        fontWeight: "400",
    },
};

export const spacing = {
    xs: 4,
    s: 8,
    m: 16,
    l: 24,
    xl: 32,
    xxl: 48,
};

export const shadows = {
    soft: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.05,
        shadowRadius: 12,
        elevation: 2,
    },
    medium: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.08,
        shadowRadius: 24,
        elevation: 4,
    },
    strong: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 12 },
        shadowOpacity: 0.12,
        shadowRadius: 32,
        elevation: 8,
    },
};

// iOS 25-style animation configurations
export const animations = {
    spring: {
        damping: 15,
        stiffness: 150,
        mass: 1,
    },
    springFast: {
        damping: 20,
        stiffness: 300,
        mass: 0.8,
    },
    springBouncy: {
        damping: 10,
        stiffness: 100,
        mass: 1,
    },
    timing: {
        fast: 150,
        normal: 250,
        slow: 400,
    },
    scale: {
        pressed: 0.95,
        normal: 1,
    },
};

// Haptic feedback types
export const haptics = {
    light: 'light',
    medium: 'medium',
    heavy: 'heavy',
    success: 'success',
    warning: 'warning',
    error: 'error',
};
