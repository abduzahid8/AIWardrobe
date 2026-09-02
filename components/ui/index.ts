/**
 * UI Component Library
 * 
 * Export all reusable UI components from this barrel file.
 * Import as: import { Card, Input, Header, EmptyState, LoadingState, toast } from '@/components/ui';
 */

/**
 * UI Components - AIWardrobe Design System
 * Premium components inspired by industry-leading designs
 */

// Core UI Components
export { Card } from './Card';
export { Input } from './Input';
export { Header } from './Header';
export { EmptyState } from './EmptyState';
export { LoadingState } from './LoadingState';
export { ToastContainer, toast } from './Toast';
export { Skeleton } from './Skeleton';
export { CachedImage } from './CachedImage';
export { OutfitCollagePreview } from './OutfitCollagePreview';
export type { OutfitCollageItem } from './OutfitCollagePreview';
export { OutfitSwipeStack } from './OutfitSwipeStack';
export type { OutfitSwipeStackHandle, SwipeStackCard } from './OutfitSwipeStack';
export { FeatureHint } from './FeatureHint';

// Premium Components (New)
export { Avatar } from './Avatar';
export { BottomSheet } from './BottomSheet';
export { ProductCard } from './ProductCard';
export { ChatBubble } from './ChatBubble';
export { StatsCard } from './StatsCard';
export { SuggestionChip } from './SuggestionChip';
export { ScreenWrapper } from './ScreenWrapper';
export { ActionCard } from './ActionCard';
export { QuickStat } from './QuickStat';
export { CelebrityClothingCard } from './CelebrityClothingCard';

// 2026 Liquid Glass Design System Components
export {
    BentoGrid,
    BentoItem,
    BentoSmall,
    BentoMedium,
    BentoLarge,
    BentoTall,
    BentoWide,
    BentoHero,
} from './BentoGrid';

export {
    LiquidGlassCard,
    ClearGlassCard,
    LightGlassCard,
    FrostedGlassCard,
    OpaqueGlassCard,
    DarkGlassCard,
    PressableGlassCard,
} from './LiquidGlassCard';

// Navigation Transition Components
export { ScreenTransitionWrapper } from '../ScreenTransitionWrapper';
export { CrossfadeTabView, TabTransitionContext } from '../CrossfadeTabView';

// Re-export types
export type { BentoGridProps, BentoItemProps } from './BentoGrid';
export type { LiquidGlassCardProps, GlassVariant, GlassElevation } from './LiquidGlassCard';
