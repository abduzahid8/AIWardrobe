/**
 * src/theme/index.ts — Barrel re-export from unified tokens.
 *
 * Everything now lives in ./tokens.ts. This file exists so that
 * existing `import { colors, spacing, ... } from '../src/theme'` statements
 * continue to work.
 */

export {
    // Light / dark color palettes
    lightColors,
    darkColors,
    colors,
    getThemeColors,

    // Typography
    typography,
    LiquidGlassTypography,

    // Spacing
    spacing,
    LiquidGlassSpacing,

    // Border radius
    borderRadius,
    LiquidGlassRadius,

    // Shadows
    shadows,

    // Animations
    animations,
    LiquidGlassAnimation,

    // Haptics
    haptics,

    // LiquidGlass design system
    LiquidGlassColors,
    SpatialElevation,
    LiquidGlassBlur,
    AccessibilityOverrides,
    BentoGridConfig,
    LiquidGlass2026Theme,

    // AppColors compat
    AppColors,
    AppColorsDark,

    // ClosetlyTheme compat
    ClosetlyThemeCompat,
} from './tokens';

export { default } from './tokens';
