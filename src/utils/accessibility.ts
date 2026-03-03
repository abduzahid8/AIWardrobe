/**
 * Accessibility Helper Utilities
 *
 * Makes it easy to add proper accessibility props to all interactive elements.
 * Import and spread onto components:
 *
 * @example
 * <TouchableOpacity {...a11yButton('Add new item')} onPress={handleAdd}>
 * <Image source={src} {...a11yImage('Red t-shirt')} />
 * <Text {...a11yHeader('My Closet')} style={styles.title}>My Closet</Text>
 */

import { AccessibilityRole } from 'react-native';

/**
 * Returns accessibility props for a button-like element.
 */
export const a11yButton = (label: string, hint?: string) => ({
    accessibilityLabel: label,
    accessibilityRole: 'button' as AccessibilityRole,
    ...(hint ? { accessibilityHint: hint } : {}),
});

/**
 * Returns accessibility props for an image.
 */
export const a11yImage = (label: string) => ({
    accessibilityLabel: label,
    accessibilityRole: 'image' as AccessibilityRole,
    accessible: true,
});

/**
 * Returns accessibility props for a header/title.
 */
export const a11yHeader = (label: string) => ({
    accessibilityLabel: label,
    accessibilityRole: 'header' as AccessibilityRole,
});

/**
 * Returns accessibility props for a text input.
 */
export const a11yInput = (label: string, hint?: string) => ({
    accessibilityLabel: label,
    ...(hint ? { accessibilityHint: hint } : {}),
});

/**
 * Returns accessibility props for a link.
 */
export const a11yLink = (label: string) => ({
    accessibilityLabel: label,
    accessibilityRole: 'link' as AccessibilityRole,
});

/**
 * Returns accessibility state for toggleable elements.
 */
export const a11yToggle = (label: string, isSelected: boolean) => ({
    accessibilityLabel: label,
    accessibilityRole: 'switch' as AccessibilityRole,
    accessibilityState: { checked: isSelected },
});

/**
 * Returns accessibility props for a tab item.
 */
export const a11yTab = (label: string, isSelected: boolean) => ({
    accessibilityLabel: label,
    accessibilityRole: 'tab' as AccessibilityRole,
    accessibilityState: { selected: isSelected },
});

export default {
    a11yButton,
    a11yImage,
    a11yHeader,
    a11yInput,
    a11yLink,
    a11yToggle,
    a11yTab,
};
