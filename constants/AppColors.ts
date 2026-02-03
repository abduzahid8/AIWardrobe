/**
 * AIWardrobe Unified App Colors
 * MONOCHROME DESIGN - Black and White Only
 */

// ============================================
// UNIFIED APP COLORS - Black & White Only
// ============================================
export const AppColors = {
    // Core - Pure White
    background: '#FFFFFF',
    surface: '#F5F5F5',
    surfaceSecondary: '#FAFAFA',
    surfaceHighlight: '#EEEEEE',

    // Text - Pure Black with Gray Variants
    text: '#000000',
    textSecondary: '#4D4D4D',
    textMuted: '#808080',
    textLight: '#B3B3B3',

    // Accent - All Black (Monochrome)
    primary: '#000000',
    accent: '#000000',

    // Status - All Black (Monochrome)
    success: '#000000',
    error: '#000000',
    warning: '#000000',
    favorite: '#000000',
    delete: '#000000',

    // Borders - Black with varying opacity
    border: '#E0E0E0',
    borderLight: '#F0F0F0',

    // Glass Effects - Monochrome
    glass: 'rgba(255, 255, 255, 0.9)',
    glassDark: 'rgba(0, 0, 0, 0.05)',

    // Premium - Black/Gray (No Gold)
    premium: '#000000',
    premiumDark: '#333333',
    vip: '#000000',
    vipDark: '#1A1A1A',

    // Gradients - Black to Gray
    gradientPrimary: ['#000000', '#333333'] as const,
    gradientAccent: ['#333333', '#000000'] as const,
    gradientPremium: ['#000000', '#1A1A1A'] as const,
    gradientVIP: ['#1A1A1A', '#000000'] as const,
};

// Dark mode colors - Inverted B&W
export const AppColorsDark = {
    // Core - Pure Black
    background: '#000000',
    surface: '#1A1A1A',
    surfaceSecondary: '#0A0A0A',
    surfaceHighlight: '#2A2A2A',

    // Text - Pure White with Gray Variants
    text: '#FFFFFF',
    textSecondary: '#B3B3B3',
    textMuted: '#808080',
    textLight: '#4D4D4D',

    // Accent - All White (Monochrome)
    primary: '#FFFFFF',
    accent: '#FFFFFF',

    // Status - All White (Monochrome)
    success: '#FFFFFF',
    error: '#FFFFFF',
    warning: '#FFFFFF',
    favorite: '#FFFFFF',
    delete: '#FFFFFF',

    // Borders - White with varying opacity
    border: '#333333',
    borderLight: '#2A2A2A',

    // Glass Effects - Monochrome
    glass: 'rgba(0, 0, 0, 0.9)',
    glassDark: 'rgba(255, 255, 255, 0.05)',

    // Premium - White/Gray (No Gold)
    premium: '#FFFFFF',
    premiumDark: '#CCCCCC',
    vip: '#FFFFFF',
    vipDark: '#E0E0E0',

    // Gradients - White to Gray
    gradientPrimary: ['#FFFFFF', '#CCCCCC'] as const,
    gradientAccent: ['#CCCCCC', '#FFFFFF'] as const,
    gradientPremium: ['#FFFFFF', '#E0E0E0'] as const,
    gradientVIP: ['#E0E0E0', '#FFFFFF'] as const,
};

// Export default as light mode
export default AppColors;
