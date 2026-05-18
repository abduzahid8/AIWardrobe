/**
 * Shared types and constants for the outfit generator feature.
 */

/**
 * Represents the lifecycle states of the outfit generation pipeline.
 * - idle: no generation in progress
 * - selecting_items: edge function is selecting outfit items
 * - composing_image: AI image composition is in progress
 * - fallback_active: local rule-based fallback is running (edge fn timed out)
 * - complete: generation finished successfully
 * - error: generation failed with an error
 */
export type GenerationStatus =
  | 'idle'
  | 'selecting_items'
  | 'composing_image'
  | 'fallback_active'
  | 'complete'
  | 'error';

/**
 * Represents a clothing item from the user's closet (MyClosetScreen).
 * Renamed from the local `ClothingItem` interface in MyClosetScreen.tsx
 * to avoid collision with the domain ClothingItem type.
 */
export interface ClosetClothingItem {
  _id: string;
  id?: string;
  type?: string;
  itemType?: string;
  color?: string;
  colorHex?: string;
  style?: string;
  description?: string;
  imageUrl?: string;
  image?: string;
  category?: string;
  wearCount?: number;
  lastWorn?: string;
  createdAt?: string;
  isFavorite?: boolean;
}

export interface OutfitItem {
  name?: string;
  image?: string | number;
  id?: string | number;
  color?: string;
  type?: string;
  macroCategory?: string;
  isShopItem?: boolean;
  price?: number;
  brand?: string;
  shopUrl?: string;
}

export type OutfitSlotId = 'outerwear' | 'top' | 'bottom' | 'shoes';

export interface GeneratedOutfit {
  id: string;
  mainImage?: string | number;
  matchScore: number;
  description: string;
  items: OutfitItem[];
  stylingTips?: string | string[];
  wardrobeItemCount?: number;
  shopItemCount?: number;
  /** Slots the wardrobe couldn't fill; used to trigger shop auto-suggest. */
  missingSlots?: OutfitSlotId[];
  /** Whether this outfit follows the 4-slot layered schema. */
  layered?: boolean;
}

export interface OutfitVisual {
  loading: boolean;
  image: string | null;
}

export interface WardrobeDisplayItem {
  id: string;
  image: string | number;
  type: string;
  color?: string;
  name?: string;
  brand?: string;
  price?: number;
  macroCategory?: string;
  category?: string;
  isShopItem?: boolean;
}

export interface AIStyle {
  id: string;
  label: string;
  icon: string;
  desc: string;
}

export const AI_STYLES: AIStyle[] = [
  { id: 'old_money', label: 'Old Money', icon: 'diamond', desc: 'Classic, refined pieces with a subtle focus on pure luxury.' },
  { id: 'semi_classic', label: 'Semi-Classic', icon: 'umbrella', desc: 'Refined everyday elegance — tailored touches with relaxed comfort.' },
  { id: 'minimalist', label: 'Minimalist', icon: 'remove', desc: 'Clean lines, neutral colors, and essential wardrobe staples.' },
  { id: 'casual', label: 'Casual', icon: 'coffee', desc: 'Relaxed, effortless style with well-fitted basics and clean combos.' },
  { id: 'business_casual', label: 'Modern Professional', icon: 'briefcase', desc: 'Sharp, tailored looks perfect for the modern workplace.' },
];

export const STYLE_PERSONALITY_MAP: Record<string, string> = {
  classic: 'old_money',
  semi_classic: 'semi_classic',
  'semi-classic': 'semi_classic',
  minimalist: 'minimalist',
  casual: 'casual',
  old_money: 'old_money',
};

export const CATEGORY_SECTIONS = [
  { category: 'top', label: 'Tops', icon: 'shirt' as const },
  { category: 'outerwear', label: 'Outerwear', icon: 'shirt' as const },
  { category: 'bottom', label: 'Bottoms', icon: 'resize' as const },
  { category: 'shoes', label: 'Shoes', icon: 'footsteps' as const },
] as const;
