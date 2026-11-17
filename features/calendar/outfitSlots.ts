import type { OutfitItem } from './types';
import { canonicalizeMacroCategory, getMacroCategory } from '../../src/utils/categoryMapper';
import { detectSlotMismatch } from '../../backend/supabase/functions/_shared/occasionRules';
import { fourSlotOutfitReason } from '../../backend/supabase/functions/_shared/outfitStructure';

/** Use saved slot metadata, repairing legacy mistags from the garment name. */
export function calendarItemMacro(item: Pick<OutfitItem, 'type' | 'name' | 'macroCategory' | 'category'>) {
  const stored = canonicalizeMacroCategory(item.macroCategory || item.category || item.type);
  return (
    detectSlotMismatch(stored === 'other' ? 'top' : stored, item.name || item.type, item.type)
    || (stored !== 'other' ? stored : getMacroCategory(item.type, item.name)));
}

export function resolveCalendarSlots(items: readonly OutfitItem[]) {
  const slots: { layer: OutfitItem | null; top: OutfitItem | null; pants: OutfitItem | null; shoes: OutfitItem | null; extras: OutfitItem[] ;
  } = {
    layer: null, top: null, pants: null, shoes: null, extras: [],
  };
  const keys = { outerwear: 'layer', top: 'top', bottom: 'pants', shoes: 'shoes' } as const;
  const seen = new Set<string>();
  for (const item of items) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    const macro = calendarItemMacro(item);
    const key = keys[macro as keyof typeof keys];
    if (key && !slots[key]) slots[key] = item;
    else slots.extras.push(item);
  }
  return slots;
}

/** Personal Stylist saves the same four identified pieces shown in its results. */
export function stylistCalendarItems(items: Array<{ id?: string; name?: string; type?: string; category?: string; macroCategory?: string; image?: string; imageUrl?: string; color?: string ;
  }>,
  compositionProtocol?: string
): OutfitItem[] {
  const normalized = items.map((item ) => ({ ...item, id: item.id || '',
    type: item.type || item.macroCategory || item.category || '',
    image: item.imageUrl || item.image || '',
    macroCategory: calendarItemMacro({ ...item, type: item.type || '' }),
  }));
  const reason = fourSlotOutfitReason(normalized, compositionProtocol !== 'optional-layer-v1');
  if (reason) throw new Error(`Invalid stylist calendar outfit: ${reason}`);
  return normalized;
}
