/**
 * LiquidGlass 2026 Design System
 * Next-generation design tokens for iOS 26 Liquid Glass aesthetic
 * with spatial computing preparation and WCAG 3.0 accessibility
 */

import { Platform, Dimensions, PixelRatio } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ============================================
// 2026 LIQUID GLASS COLOR SYSTEM - MONOCHROME
// Black and White Only Design
// ============================================
export const LiquidGlassColors = {
  // Monochrome Glass Materials
  glass: {
    clear: 'rgba(255, 255, 255, 0.12)',
    light: 'rgba(255, 255, 255, 0.25)',
    frosted: 'rgba(255, 255, 255, 0.45)',
    opaque: 'rgba(255, 255, 255, 0.85)',
    tinted: 'rgba(255, 255, 255, 0.08)',
    dark: 'rgba(0, 0, 0, 0.15)',
    darkFrosted: 'rgba(0, 0, 0, 0.35)',
  },

  // Core Background Colors - Pure B&W
  background: {
    primary: '#FFFFFF',
    secondary: '#F5F5F5',
    tertiary: '#EEEEEE',
    elevated: '#FFFFFF',
  },

  // Text - Pure Black/White with gray variants
  text: {
    primary: '#000000',
    secondary: '#4D4D4D',
    tertiary: '#808080',
    disabled: '#B3B3B3',
    onGlass: '#000000',
    onDark: '#FFFFFF',
  },

  // Accent colors - All Black (monochrome)
  accent: {
    primary: '#000000',
    secondary: '#333333',
    tertiary: '#666666',
    success: '#000000',
    warning: '#000000',
    error: '#000000',
  },

  // Gradients - Black to White / White to Black
  gradients: {
    liquidGlass: ['rgba(255,255,255,0.9)', 'rgba(255,255,255,0.6)'] as const,
    primaryAccent: ['#000000', '#333333'] as const,
    warmGlow: ['#333333', '#000000'] as const,
    coolWave: ['#4D4D4D', '#000000'] as const,
    premium: ['#000000', '#1A1A1A'] as const,
    dark: ['#000000', '#1A1A1A'] as const,
  },

  // Border colors - Monochrome
  border: {
    glass: 'rgba(0, 0, 0, 0.1)',
    glassDark: 'rgba(0, 0, 0, 0.15)',
    subtle: 'rgba(0, 0, 0, 0.08)',
    strong: 'rgba(0, 0, 0, 0.2)',
  },
};

// ============================================
// SPATIAL ELEVATION SYSTEM (Z-axis depth)
// ============================================
export const SpatialElevation = {
  // Elevation levels in logical units (maps to shadow intensity)
  levels: {
    surface: 0,       // Flat on background
    raised: 4,        // Subtle lift
    card: 8,          // Standard cards
    floating: 16,     // FABs, floating elements
    modal: 24,        // Bottom sheets, modals
    popover: 32,      // Dropdowns, tooltips
    overlay: 48,      // Full overlays
    alert: 56,        // Critical alerts, dialogs
  },

  // Shadow generator based on elevation
  getShadow: (elevation: number, isDark = false) => {
    const opacity = Math.min(0.05 + (elevation * 0.008), 0.25);
    const blur = Math.min(4 + (elevation * 1.5), 64);
    const offset = Math.min(2 + (elevation * 0.5), 24);

    return {
      shadowColor: isDark ? '#FFFFFF' : '#000000',
      shadowOffset: { width: 0, height: offset },
      shadowOpacity: opacity,
      shadowRadius: blur,
      elevation: Math.min(elevation, 24), // Android max
    };
  },
};

// ============================================
// TYPOGRAPHY SYSTEM (SF Pro inspired)
// ============================================
export const LiquidGlassTypography = {
  // Font family with fallbacks
  fontFamily: Platform.select({
    ios: 'System',
    android: 'Roboto',
    default: 'System',
  }),

  // Type scale with optical sizing
  scale: {
    // Display - for hero sections
    displayLarge: {
      fontSize: 44,
      lineHeight: 52,
      fontWeight: '700' as const,
      letterSpacing: -1.0,
    },
    displayMedium: {
      fontSize: 36,
      lineHeight: 44,
      fontWeight: '700' as const,
      letterSpacing: -0.5,
    },
    displaySmall: {
      fontSize: 32,
      lineHeight: 40,
      fontWeight: '600' as const,
      letterSpacing: -0.3,
    },

    // Headlines
    headlineLarge: {
      fontSize: 28,
      lineHeight: 36,
      fontWeight: '600' as const,
      letterSpacing: -0.2,
    },
    headlineMedium: {
      fontSize: 24,
      lineHeight: 32,
      fontWeight: '600' as const,
      letterSpacing: -0.1,
    },
    headlineSmall: {
      fontSize: 20,
      lineHeight: 28,
      fontWeight: '600' as const,
      letterSpacing: 0,
    },

    // Titles
    titleLarge: {
      fontSize: 18,
      lineHeight: 26,
      fontWeight: '600' as const,
      letterSpacing: 0,
    },
    titleMedium: {
      fontSize: 16,
      lineHeight: 24,
      fontWeight: '600' as const,
      letterSpacing: 0.1,
    },
    titleSmall: {
      fontSize: 14,
      lineHeight: 20,
      fontWeight: '600' as const,
      letterSpacing: 0.1,
    },

    // Body
    bodyLarge: {
      fontSize: 16,
      lineHeight: 24,
      fontWeight: '400' as const,
      letterSpacing: 0.15,
    },
    bodyMedium: {
      fontSize: 14,
      lineHeight: 20,
      fontWeight: '400' as const,
      letterSpacing: 0.2,
    },
    bodySmall: {
      fontSize: 12,
      lineHeight: 16,
      fontWeight: '400' as const,
      letterSpacing: 0.3,
    },

    // Labels
    labelLarge: {
      fontSize: 14,
      lineHeight: 20,
      fontWeight: '500' as const,
      letterSpacing: 0.5,
    },
    labelMedium: {
      fontSize: 12,
      lineHeight: 16,
      fontWeight: '500' as const,
      letterSpacing: 0.5,
    },
    labelSmall: {
      fontSize: 11,
      lineHeight: 14,
      fontWeight: '500' as const,
      letterSpacing: 0.6,
      textTransform: 'uppercase' as const,
    },
  },

  // Apply dynamic type scaling
  withDynamicScale: (baseStyle: Record<string, unknown>, scale: number) => ({
    ...baseStyle,
    fontSize: (baseStyle.fontSize as number) * Math.max(0.8, Math.min(scale, 2.0)),
    lineHeight: (baseStyle.lineHeight as number) * Math.max(0.8, Math.min(scale, 2.0)),
  }),
};

// ============================================
// SPACING & LAYOUT SYSTEM
// ============================================
export const LiquidGlassSpacing = {
  // Base unit: 4dp
  base: 4,

  // Spacing scale
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
  xxxl: 64,

  // Screen-level spacing
  screenPadding: 20,
  cardGap: 16,
  sectionGap: 32,

  // Bento Grid specific
  bento: {
    gap: 12,
    padding: 16,
    itemMinHeight: 100,
  },

  // Touch targets (WCAG 3.0: 44mm minimum)
  touchTarget: {
    minimum: 44,
    comfortable: 48,
    large: 56,
  },
};

// ============================================
// BORDER RADIUS SYSTEM
// ============================================
export const LiquidGlassRadius = {
  // Progressive radius scale
  none: 0,
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,

  // Semantic radius
  card: 24,
  button: 14,
  chip: 20,
  pill: 50,
  input: 12,
  modal: 28,
  bottomSheet: 24,

  // Full circle
  full: 9999,
};

// ============================================
// ANIMATION SYSTEM
// ============================================
export const LiquidGlassAnimation = {
  // Duration presets (in ms)
  duration: {
    instant: 50,
    fast: 150,
    normal: 250,
    slow: 400,
    emphasis: 600,
  },

  // Spring configurations for react-native-reanimated
  spring: {
    // Snappy, responsive feel
    snappy: {
      damping: 20,
      stiffness: 300,
      mass: 0.8,
    },
    // Bouncy, playful feel
    bouncy: {
      damping: 12,
      stiffness: 150,
      mass: 1,
    },
    // Smooth, luxurious feel
    smooth: {
      damping: 25,
      stiffness: 120,
      mass: 1.2,
    },
    // Gentle, subtle feel
    gentle: {
      damping: 30,
      stiffness: 80,
      mass: 1.5,
    },
  },

  // Easing curves
  easing: {
    standard: [0.4, 0.0, 0.2, 1],
    decelerate: [0.0, 0.0, 0.2, 1],
    accelerate: [0.4, 0.0, 1, 1],
    sharp: [0.4, 0.0, 0.6, 1],
  },

  // Reduced motion alternatives
  reducedMotion: {
    duration: 0,
    spring: { damping: 100, stiffness: 1000, mass: 1 },
  },
};

// ============================================
// BLUR CONFIGURATIONS
// ============================================
export const LiquidGlassBlur = {
  // Blur intensity levels for expo-blur
  intensity: {
    subtle: 20,
    light: 40,
    medium: 60,
    heavy: 80,
    extreme: 100,
  },

  // Tint options
  tint: {
    light: 'light' as const,
    dark: 'dark' as const,
    default: 'default' as const,
    chromatic: 'chromaBlur' as const,
  },

  // Get blur config based on scroll position or interaction
  getDynamicBlur: (scrollOffset: number, maxScroll: number = 100) => {
    const progress = Math.min(scrollOffset / maxScroll, 1);
    return {
      intensity: 20 + (progress * 60), // 20-80 range
      backgroundColor: `rgba(255, 255, 255, ${0.6 + (progress * 0.25)})`,
    };
  },
};

// ============================================
// ACCESSIBILITY OVERRIDES
// ============================================
export const AccessibilityOverrides = {
  // High contrast mode replacements
  highContrast: {
    glass: {
      clear: 'rgba(255, 255, 255, 0.95)',
      frosted: '#FFFFFF',
      border: '#000000',
    },
    text: {
      primary: '#000000',
      secondary: '#333333',
    },
    shadows: {
      // Replace soft shadows with solid borders
      getShadow: () => ({
        borderWidth: 2,
        borderColor: '#000000',
        shadowOpacity: 0,
      }),
    },
  },

  // Reduced motion replacements
  reducedMotion: {
    duration: 0,
    spring: { damping: 1000, stiffness: 1000, mass: 1 },
  },

  // Dynamic type support
  dynamicType: {
    minimumScale: 0.8,
    maximumScale: 2.5,
  },
};

// ============================================
// BENTO GRID CONFIGURATIONS
// ============================================
export const BentoGridConfig = {
  // Column configurations by screen width
  columns: {
    compact: 2,   // < 400dp
    regular: 3,   // 400-600dp
    expanded: 4,  // > 600dp
  },

  // Get column count based on screen width
  getColumns: (width: number = SCREEN_WIDTH): number => {
    if (width < 400) return 2;
    if (width < 600) return 3;
    return 4;
  },

  // Item size presets
  itemSizes: {
    small: { colSpan: 1, rowSpan: 1 },
    medium: { colSpan: 2, rowSpan: 1 },
    large: { colSpan: 2, rowSpan: 2 },
    wide: { colSpan: 3, rowSpan: 1 },
    tall: { colSpan: 1, rowSpan: 2 },
    hero: { colSpan: 4, rowSpan: 2 },
  },

  // Calculate item dimensions
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
      minHeight: cellWidth, // Square base
    };
  },
};

// ============================================
// COMBINED THEME EXPORT
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

  // Screen dimensions for responsive design
  screen: {
    width: SCREEN_WIDTH,
    height: SCREEN_HEIGHT,
    pixelRatio: PixelRatio.get(),
  },

  // Platform info
  platform: Platform.OS,
  isIOS: Platform.OS === 'ios',
  isAndroid: Platform.OS === 'android',
};

// Default export
export default LiquidGlass2026Theme;
