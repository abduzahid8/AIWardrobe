import { fourSlotOutfitReason } from '../domain/outfitStructure';
import { passesStyleJudge } from '../../features/outfit-generator/utils/occasionRules';
import { isShopOnlyDeck, type GeneratedOutfit } from './outfitGenerationService';

export function dailyOutfitCacheKey(userId: string, style: string, occasion: string): string {
    // Older entries may lack authoritative currency. Regenerate instead of guessing units.
    return `daily_ai_outfit_v21_stylist_quality::${encodeURIComponent(userId)}::${encodeURIComponent(style)}::${encodeURIComponent(occasion)}`;
}

export function validatedDailyOutfits(value: unknown, style: string, occasion: string, weather?: { temp: number; condition: string } | null): GeneratedOutfit[] {
    if (!Array.isArray(value)) return [];
    const signatures = new Set<string>();
    return value.filter((outfit): outfit is GeneratedOutfit => {
        if (!outfit || fourSlotOutfitReason(outfit.items)) return false;
        if (!passesStyleJudge(outfit.items, style, occasion, weather)) return false;
        if (isShopOnlyDeck(style, occasion) && outfit.items.some((item: any) => item.isShopItem !== true)) return false;
        const signature = outfit.items.map((item: any) => item.id).sort().join('|');
        if (signatures.has(signature)) return false;
        signatures.add(signature);
        return true;
    });
}
