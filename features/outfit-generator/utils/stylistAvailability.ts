import { needsLayering, type CompositionWeather } from '../../../backend/supabase/functions/_shared/outfitComposition';
import { isFourSlotCandidate } from '../../../backend/supabase/functions/_shared/outfitStructure';
import { outfitChoices } from '../../../backend/supabase/functions/_shared/outfitChoices';
import { passesStyleJudge, rejectForStyleOccasion, rejectForClimate, rejectForActivity, rejectForRequestIntent } from './occasionRules';
import { scoreItemForStyle, normalizeStyleId } from './styleInference';

interface InventoryItem {
  id: string;
  name?: string;
  type?: string;
  macroCategory?: string;
  color?: string;
  description?: string;
}
export interface StylistCombination { style: string; occasion: string }

/** Offer a combination only after finding a complete, validated look in real inventory. */
export function availableStylistCombinations(
  inventory: InventoryItem[], styles: string[], occasions: string[],
  weather?: CompositionWeather, prompt = '', lockedItemId?: string,
  preferences: { avoidColors?: string[]; dislikedPatterns?: string[] } = {},
): StylistCombination[] {
  const result: StylistCombination[] = [];
  for (const style of styles) for (const occasion of occasions) {
    const request = `${occasion} ${prompt}`.trim();
    const items = inventory.filter(item => isFourSlotCandidate(item)
      && !(preferences.avoidColors || []).some(color => (item.color || '').toLowerCase().includes(color.toLowerCase())
        || ` ${item.name || ''} ${item.description || ''} `.toLowerCase().includes(` ${color.toLowerCase()} `))
      && !(preferences.dislikedPatterns || []).some(pattern => `${item.name || ''} ${item.description || ''}`.toLowerCase().includes(pattern.toLowerCase()))
      && !rejectForStyleOccasion(item, item.id === lockedItemId ? 'casual' : style, request)
      && !rejectForClimate(item, weather)
      && !rejectForActivity(item, request) && !rejectForRequestIntent(item, request, weather));
    const ranked = items.map(item => ({ item, score: scoreItemForStyle(item, normalizeStyleId(style)) }))
      .sort((a, b) => b.score - a.score).map(({ item }) => item);
    const pool = ['top', 'bottom', 'shoes', 'outerwear'].flatMap(slot => {
      const group = ranked.filter(item => item.macroCategory === slot);
      const anchor = group.find(item => item.id === lockedItemId);
      return anchor ? [anchor, ...group.filter(item => item !== anchor).slice(0, 15)] : group.slice(0, 16);
    });
    const choices = outfitChoices(pool, needsLayering(style, weather, prompt),
      parts => passesStyleJudge(parts, style, request, weather, lockedItemId), () => false,
      lockedItemId ? [lockedItemId] : [], 1, 1024);
    if (choices.length) result.push({ style, occasion });
  }
  return result;
}
