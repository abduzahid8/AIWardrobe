/**
 * Shared types and constants for the outfit generator feature.
 */

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
  { id: 'streetwear', label: 'Streetwear', icon: 'flash', desc: 'Edgy, oversized aesthetics blending comfort with high fashion.' },
  { id: 'minimalist', label: 'Minimalist', icon: 'remove', desc: 'Clean lines, neutral colors, and essential wardrobe staples.' },
  { id: 'y2k', label: 'Y2K', icon: 'sparkles', desc: 'Bold colors, nostalgic 2000s vibes, and striking accessories.' },
  { id: 'business_casual', label: 'Modern Professional', icon: 'briefcase', desc: 'Sharp, tailored looks perfect for the modern workplace.' },
];

export const STYLE_PERSONALITY_MAP: Record<string, string> = {
  classic: 'old_money',
  trendy: 'streetwear',
  minimalist: 'minimalist',
  bohemian: 'y2k',
  edgy: 'streetwear',
  romantic: 'old_money',
  sporty: 'streetwear',
};

export const CATEGORY_SECTIONS = [
  { category: 'top', label: 'Tops', icon: 'shirt' as const },
  { category: 'outerwear', label: 'Outerwear', icon: 'shirt' as const },
  { category: 'bottom', label: 'Bottoms', icon: 'resize' as const },
  { category: 'shoes', label: 'Shoes', icon: 'footsteps' as const },
] as const;
