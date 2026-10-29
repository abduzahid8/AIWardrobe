export type CompositionSlot = 'outerwear' | 'top' | 'bottom' | 'shoes';

export interface CompositionWeather {
  temp?: number | null;
  condition?: string | null;
}

export interface OutfitCompositionPolicy {
  layered: boolean;
  activeSlots: CompositionSlot[];
}

const LAYERED_SLOTS: CompositionSlot[] = ['outerwear', 'top', 'bottom', 'shoes'];
export const DISPLAY_DECK_SIZE = 3;

/** Every generated outfit has an outer layer, top, bottom and shoes. */
export function resolveOutfitComposition(
  _style: string | null | undefined,
  _weather?: CompositionWeather | null,
  _prompt?: string | null,
): OutfitCompositionPolicy {
  return { layered: true, activeSlots: [...LAYERED_SLOTS] };
}

/** A destination temperature in the request overrides the current local weather. */
export function resolveStylistWeather(prompt?: string | null, weather?: CompositionWeather | null): CompositionWeather | undefined {
  const text = prompt || '';
  const explicit = text.match(/(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*C|C\b|degrees?\s*(?:C|celsius)|celsius)\b/i);
  if (explicit) return {temp: Number(explicit[1]), condition: 'requested temperature'};
  if (/\b(freezing|snow|winter|sub[ -]?zero)\b|мороз|зим[ауы]/i.test(text)) return {temp: 0, condition: 'winter'};
  if (/\bsummer\b|лет[оа]/i.test(text)) return {temp: 30, condition: 'summer'};
  return weather || undefined;
}

export function needsLayering(
  style: string | null | undefined,
  weather?: CompositionWeather | null,
  prompt?: string | null,
): boolean {
  return resolveOutfitComposition(style, weather, prompt).layered;
}
