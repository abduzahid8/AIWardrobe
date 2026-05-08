/**
 * useEssentialItems - Hook for essential wardrobe item classification
 * Extracted from HomeScreen.tsx to improve maintainability
 */

import { useMemo } from 'react';
import type { ShopCatalogItem } from '../../../features/try-on/types';

type EssentialSlot = 'outerwear' | 'shirts' | 'knitwear' | 'tees' | 'bottoms' | 'shoes';

const ESSENTIALS_REQUIREMENTS: Record<EssentialSlot, number> = {
  outerwear: 2,
  shirts: 2,
  knitwear: 2,
  tees: 2,
  bottoms: 3,
  shoes: 3,
};

const ESSENTIALS_LIMIT = Object.values(ESSENTIALS_REQUIREMENTS).reduce(
  (sum, count) => sum + count,
  0
);

// Keywords for classification
const OUTERWEAR_KEYWORDS = [
  'blazer', 'jacket', 'coat', 'overcoat', 'suit', 'trench', 'parka',
  'bomber', 'puffer', 'vest', 'waistcoat', 'cardigan', 'hoodie'
];

const SHIRT_KEYWORDS = [
  'shirt', 'oxford', 'dress shirt', 'broadcloth', 'poplin', 'linen shirt',
  'flannel', 'chambray', 'button-down', 'button down'
];

const KNITWEAR_KEYWORDS = [
  'sweater', 'knit', 'pullover', 'turtleneck', 'crewneck', 'v-neck',
  'cashmere', 'merino', 'wool', 'chunky', 'cable', 'cardigan'
];

const TEE_KEYWORDS = [
  't-shirt', 'tshirt', 'tee', 'polo', 'henley', 'tank', 'top'
];

const BOTTOM_KEYWORDS = [
  'pants', 'trousers', 'chinos', 'jeans', 'shorts', 'joggers',
  'slacks', 'cargos', 'corduroy'
];

const SHOE_KEYWORDS = [
  'shoes', 'sneakers', 'loafers', 'boots', 'oxfords', 'derby',
  'trainers', 'slip-on', 'mules'
];

function classifyUpperBodyItem(item: ShopCatalogItem): 'outerwear' | 'shirts' | 'knitwear' | 'tees' | null {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  
  if (OUTERWEAR_KEYWORDS.some((k) => text.includes(k))) return 'outerwear';
  if (SHIRT_KEYWORDS.some((k) => text.includes(k))) return 'shirts';
  if (KNITWEAR_KEYWORDS.some((k) => text.includes(k))) return 'knitwear';
  if (TEE_KEYWORDS.some((k) => text.includes(k))) return 'tees';
  
  if (text.includes('blazer') || text.includes('jacket') || text.includes('coat')) return 'outerwear';
  if (text.includes('shirt')) return 'shirts';
  if (text.includes('sweater') || text.includes('knit')) return 'knitwear';
  
  return 'tees';
}

function classifyBottomItem(item: ShopCatalogItem): 'bottoms' | null {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  if (BOTTOM_KEYWORDS.some((k) => text.includes(k))) return 'bottoms';
  if (item.garmentType === 'lower_body') return 'bottoms';
  return null;
}

function classifyShoeItem(item: ShopCatalogItem): 'shoes' | null {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  if (SHOE_KEYWORDS.some((k) => text.includes(k))) return 'shoes';
  if (item.garmentType === 'shoes') return 'shoes';
  return null;
}

export function useEssentialItems(items: ShopCatalogItem[]) {
  return useMemo(() => {
    const pickedCounts: Record<EssentialSlot, number> = {
      outerwear: 0,
      shirts: 0,
      knitwear: 0,
      tees: 0,
      bottoms: 0,
      shoes: 0,
    };

    const selected: ShopCatalogItem[] = [];
    const selectedIds = new Set<string>();
    
    const shuffled = [...items].sort(() => Math.random() - 0.5);

    for (const item of shuffled) {
      if (selectedIds.has(item.id)) continue;
      
      let slot: EssentialSlot | null = null;

      if (item.garmentType === 'upper_body') {
        slot = classifyUpperBodyItem(item);
      } else if (item.garmentType === 'lower_body') {
        slot = classifyBottomItem(item);
      } else if (item.garmentType === 'shoes') {
        slot = classifyShoeItem(item);
      }

      if (!slot) continue;
      if (pickedCounts[slot] >= ESSENTIALS_REQUIREMENTS[slot]) continue;

      selected.push(item);
      selectedIds.add(item.id);
      pickedCounts[slot] += 1;

      if (selected.length >= ESSENTIALS_LIMIT) break;
    }

    return {
      items: selected,
      counts: pickedCounts,
      isComplete: selected.length >= ESSENTIALS_LIMIT,
    };
  }, [items]);
}

export { ESSENTIALS_LIMIT, ESSENTIALS_REQUIREMENTS };
export type { EssentialSlot };
