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
const FLAT_SLOTS: CompositionSlot[] = ['top', 'bottom', 'shoes'];

/**
 * Resolve the outfit shape once for the request. Style affects ranking and
 * taste, but not the structural rule: top + bottom + shoes by default, with a
 * layer only for cold weather or an explicit formal request.
 */
export function resolveOutfitComposition(
  _style: string | null | undefined,
  weather?: CompositionWeather | null,
  prompt?: string | null,
): OutfitCompositionPolicy {
  const promptBlob = String(prompt || '').toLowerCase();
  const explicitFormal = /\b(black[ -]?tie|tuxedo|smoking|смокинг)\b/.test(promptBlob);
  const explicitFlat = /\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob);
  const condition = String(weather?.condition || '').toLowerCase();
  const temp = typeof weather?.temp === 'number' ? weather.temp : null;
  const cold = (temp != null && temp < 18)
    || /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm|cool)\b/.test(condition);
  const layered = explicitFormal || (!explicitFlat && cold);
  return {
    layered,
    activeSlots: [...(layered ? LAYERED_SLOTS : FLAT_SLOTS)],
  };
}

export function needsLayering(
  style: string | null | undefined,
  weather?: CompositionWeather | null,
  prompt?: string | null,
): boolean {
  return resolveOutfitComposition(style, weather, prompt).layered;
}
