// AIWardrobe Component Style Presets
// Inspired by: SurfPad (product cards), DiWander (glassmorphism), Nandly (buttons)

import { StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { colors, spacing, borderRadius, shadows, typography } from './index';

// ============================================
// BUTTON PRESETS
// ============================================
export const buttonStyles = StyleSheet.create({
    // Primary - Bold, high-contrast CTA
    primary: {
        backgroundColor: colors.button.primary,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.xl,
        borderRadius: borderRadius.m,
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        ...shadows.medium,
    } as ViewStyle,

    primaryText: {
        color: colors.button.primaryText,
        fontSize: typography.button.fontSize,
        fontWeight: typography.button.fontWeight,
        letterSpacing: typography.button.letterSpacing,
    } as TextStyle,

    // Secondary - Subtle, surface-level
    secondary: {
        backgroundColor: colors.button.secondary,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.xl,
        borderRadius: borderRadius.m,
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        ...shadows.soft,
    } as ViewStyle,

    secondaryText: {
        color: colors.button.secondaryText,
        fontSize: typography.button.fontSize,
        fontWeight: typography.button.fontWeight,
    } as TextStyle,

    // Ghost - Transparent, text-only
    ghost: {
        backgroundColor: 'transparent',
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.l,
        borderRadius: borderRadius.m,
        alignItems: 'center',
        justifyContent: 'center',
    } as ViewStyle,

    ghostText: {
        color: colors.button.ghostText,
        fontSize: typography.button.fontSize,
        fontWeight: typography.button.fontWeight,
    } as TextStyle,

    // CTA - Warm, inviting (Wardrobe 2.0 inspired)
    cta: {
        backgroundColor: colors.button.cta,
        paddingVertical: spacing.m + 2,
        paddingHorizontal: spacing.xl,
        borderRadius: borderRadius.m,
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        ...shadows.medium,
    } as ViewStyle,

    ctaText: {
        color: colors.button.ctaText,
        fontSize: typography.button.fontSize,
        fontWeight: '700' as const,
        letterSpacing: 0.5,
    } as TextStyle,

    // Glass - DiWander inspired
    glass: {
        backgroundColor: colors.glass.background,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.xl,
        borderRadius: borderRadius.m,
        borderWidth: 1,
        borderColor: colors.glass.border,
        alignItems: 'center',
        justifyContent: 'center',
        ...shadows.soft,
    } as ViewStyle,

    glassText: {
        color: colors.text.primary,
        fontSize: typography.button.fontSize,
        fontWeight: typography.button.fontWeight,
    } as TextStyle,
});

// ============================================
// CARD PRESETS
// ============================================
export const cardStyles = StyleSheet.create({
    // Flat - Minimal elevation
    flat: {
        backgroundColor: colors.surface,
        borderRadius: borderRadius.m,
        padding: spacing.l,
    } as ViewStyle,

    // Elevated - Deep shadows for depth
    elevated: {
        backgroundColor: colors.surface,
        borderRadius: borderRadius.l,
        padding: spacing.l,
        ...shadows.strong,
    } as ViewStyle,

    // Glass - DiWander glassmorphism
    glass: {
        backgroundColor: colors.glass.background,
        borderRadius: borderRadius.l,
        padding: spacing.l,
        borderWidth: 1,
        borderColor: colors.glass.border,
        ...shadows.soft,
    } as ViewStyle,

    // Product - SurfPad inspired
    product: {
        backgroundColor: colors.surface,
        borderRadius: borderRadius.m,
        overflow: 'hidden',
        ...shadows.card,
    } as ViewStyle,

    // Interactive - For pressable cards
    interactive: {
        backgroundColor: colors.surface,
        borderRadius: borderRadius.m,
        padding: spacing.l,
        ...shadows.medium,
    } as ViewStyle,

    // Outline - Border only
    outline: {
        backgroundColor: 'transparent',
        borderRadius: borderRadius.m,
        padding: spacing.l,
        borderWidth: 1,
        borderColor: colors.border,
    } as ViewStyle,
});

// ============================================
// INPUT PRESETS
// ============================================
export const inputStyles = StyleSheet.create({
    // Minimal - Clean, flat
    minimal: {
        backgroundColor: colors.surfaceHighlight,
        borderRadius: borderRadius.m,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.l,
        fontSize: typography.body.fontSize,
        color: colors.text.primary,
    } as ViewStyle & TextStyle,

    // Outlined - Border emphasis
    outlined: {
        backgroundColor: colors.surface,
        borderRadius: borderRadius.m,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.l,
        fontSize: typography.body.fontSize,
        color: colors.text.primary,
        borderWidth: 1.5,
        borderColor: colors.border,
    } as ViewStyle & TextStyle,

    // Glass - DiWander inspired
    glass: {
        backgroundColor: colors.glass.background,
        borderRadius: borderRadius.m,
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.l,
        fontSize: typography.body.fontSize,
        color: colors.text.primary,
        borderWidth: 1,
        borderColor: colors.glass.border,
    } as ViewStyle & TextStyle,
});

// ============================================
// MODAL PRESETS
// ============================================
export const modalStyles = StyleSheet.create({
    // Bottom sheet backdrop
    backdrop: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'flex-end',
    } as ViewStyle,

    // Bottom sheet container (Nandly-inspired)
    bottomSheetContainer: {
        backgroundColor: colors.surface,
        borderTopLeftRadius: borderRadius.xl,
        borderTopRightRadius: borderRadius.xl,
        paddingTop: spacing.s,
        paddingHorizontal: spacing.l,
        paddingBottom: spacing.xxl,
        ...shadows.strong,
    } as ViewStyle,

    // Bottom sheet handle
    bottomSheetHandle: {
        width: 40,
        height: 4,
        backgroundColor: colors.border,
        borderRadius: 2,
        alignSelf: 'center',
        marginBottom: spacing.m,
    } as ViewStyle,

    // Center modal
    centerContainer: {
        backgroundColor: colors.surface,
        borderRadius: borderRadius.xl,
        padding: spacing.xl,
        margin: spacing.xl,
        ...shadows.strong,
    } as ViewStyle,

    // Full screen modal
    fullScreenContainer: {
        flex: 1,
        backgroundColor: colors.background,
    } as ViewStyle,
});

// ============================================
// AVATAR PRESETS
// ============================================
export const avatarStyles = StyleSheet.create({
    // Small avatar (32x32)
    small: {
        width: 32,
        height: 32,
        borderRadius: 16,
        overflow: 'hidden',
    } as ViewStyle,

    // Medium avatar (48x48)
    medium: {
        width: 48,
        height: 48,
        borderRadius: 24,
        overflow: 'hidden',
    } as ViewStyle,

    // Large avatar (64x64)
    large: {
        width: 64,
        height: 64,
        borderRadius: 32,
        overflow: 'hidden',
    } as ViewStyle,

    // XL avatar (96x96) - Alta-style floating
    xlarge: {
        width: 96,
        height: 96,
        borderRadius: 48,
        overflow: 'hidden',
        ...shadows.glow,
    } as ViewStyle,

    // Avatar placeholder
    placeholder: {
        backgroundColor: colors.surfaceHighlight,
        alignItems: 'center',
        justifyContent: 'center',
    } as ViewStyle,
});

// ============================================
// CHIP/TAG PRESETS
// ============================================
export const chipStyles = StyleSheet.create({
    // Default chip
    default: {
        backgroundColor: colors.surfaceHighlight,
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
        borderRadius: borderRadius.full,
        flexDirection: 'row',
        alignItems: 'center',
    } as ViewStyle,

    // Selected chip
    selected: {
        backgroundColor: colors.primary,
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
        borderRadius: borderRadius.full,
        flexDirection: 'row',
        alignItems: 'center',
    } as ViewStyle,

    // Outlined chip
    outlined: {
        backgroundColor: 'transparent',
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
        borderRadius: borderRadius.full,
        borderWidth: 1,
        borderColor: colors.border,
        flexDirection: 'row',
        alignItems: 'center',
    } as ViewStyle,

    defaultText: {
        fontSize: typography.bodySmall.fontSize,
        fontWeight: '500' as const,
        color: colors.text.secondary,
    } as TextStyle,

    selectedText: {
        fontSize: typography.bodySmall.fontSize,
        fontWeight: '600' as const,
        color: colors.button.primaryText,
    } as TextStyle,
});

// ============================================
// DIVIDER PRESETS
// ============================================
export const dividerStyles = StyleSheet.create({
    horizontal: {
        height: 1,
        backgroundColor: colors.border,
        marginVertical: spacing.m,
    } as ViewStyle,

    vertical: {
        width: 1,
        backgroundColor: colors.border,
        marginHorizontal: spacing.m,
    } as ViewStyle,

    thick: {
        height: 2,
        backgroundColor: colors.border,
        marginVertical: spacing.l,
    } as ViewStyle,
});
