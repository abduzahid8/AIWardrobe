/**
 * Hooks Library
 * Export all custom hooks from this barrel file
 */

// Accessibility
export {
    useAccessibility,
    useReducedMotion,
    useScreenReader,
} from './useAccessibility';

// Subscription gating
export { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';

// Typed navigation
export { useAppNavigation } from './useAppNavigation';
export type { AppNavigationProp } from './useAppNavigation';

// Re-export types
export type { UseAccessibilityReturn } from './useAccessibility';

// Language
export { default as useLanguageStore } from '../store/languageStore';
export type { Language } from '../store/languageStore';
