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
};
