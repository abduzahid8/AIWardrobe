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

// Re-export types
export type { UseAccessibilityReturn } from './useAccessibility';
