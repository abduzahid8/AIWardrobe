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

/** Explicit composition intent takes precedence over the usual four-slot deck. */
export function resolveOutfitComposition(
  style: string | null | undefined,
  weather?: CompositionWeather | null,
  prompt?: string | null,
): OutfitCompositionPolicy {
  const noLayer = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?(?:jackets?|coats?|layers?|outerwear)\b|без\s+(?:куртк|пиджак|пальто)/i.test(prompt || '');
  const text = `${style || ''} ${prompt || ''}`.toLowerCase();
  const explicitLayer = /\b(layer(?:ed|ing)?|jacket|coat|blazer|outerwear|parka|puffer|trench|cardigan)\b|слоя|куртк|пиджак|пальто/i.test(text);
  const temp = Number(weather?.temp);
  const coldWeather = Number.isFinite(temp)
    ? temp < 18
    : /\b(cold|freezing|snow|winter|wind|rain)\b|мороз|снег|зим|ветер|дожд/i.test(
      `${weather?.condition || ''} ${prompt || ''}`,
    );
  const warmWeather = Number.isFinite(temp) && temp >= 20;
  const layered = !noLayer && (explicitLayer || (coldWeather && !warmWeather));
  return { layered, activeSlots: layered ? [...LAYERED_SLOTS] : ['top', 'bottom', 'shoes'] };
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
