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

/** Positive cold intent can be qualitative; it never creates a temperature. */
export function hasColdLayerIntent(weather?: CompositionWeather | null): boolean {
  return (weather?.temp != null && weather.temp < 12)
    || /winter|freezing|snow|зим|мороз|qish|қиш/i.test(weather?.condition || '');
}

/** Four-slot clients remain compatible. Only negotiated clients may omit a physical layer. */
export function resolveOutfitComposition(
  _style: string | null | undefined,
  _weather?: CompositionWeather | null,
  _prompt?: string | null,
  optionalLayer = false,
): OutfitCompositionPolicy {
  const cold = (_weather?.temp != null && _weather.temp < 18) || /winter|freezing|snow|зим|мороз/i.test(_weather?.condition || '');
  // Respect explicit customer garment constraints in all supported copy
  // languages. 'No black jacket' excludes a color, not the entire layer:
  // only negative layer phrases without a color modifier omit outerwear.
  const brief = _prompt || '';
  const noLayerIntent = /\b(?:no|without|skip)\s+(?:(?:an?|any|extra|additional|unnecessary)\s+){0,3}(?:layers?|outerwear|jacket|blazer|coat|cardigan)\b/i.test(brief)
    || /без\s+(?:(?:лишнего|дополнительного)\s+)?(?:пиджака|куртки|пальто|кардигана|верхней\s+одежды|слоёв|слоя)/i.test(brief)
    || /\b(?:pidjaksiz|kurtkasiz|paltosiz)\b/i.test(brief);
  const noLayer = optionalLayer && !cold && (noLayerIntent || (_weather?.temp != null && _weather.temp >= 25));
  return noLayer ? {layered:false,activeSlots:['top','bottom','shoes']} : { layered: true, activeSlots: [...LAYERED_SLOTS] };
}

/** A destination temperature in the request overrides the current local weather. */
export function resolveStylistWeather(prompt?: string | null, weather?: CompositionWeather | null): CompositionWeather | undefined {
  const text = prompt || '';
  const explicit = text.match(/(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*C|C\b|degrees?\s*(?:C|celsius)|celsius)\b/i);
  if (explicit) return {temp: Number(explicit[1]), condition: 'requested temperature'};
  // A season keyword conveys intent, not a temperature. Preserve a
  // measured/explicit destination temperature already supplied by the client:
  // dropping 32°C when the brief said 'summer' was incorrectly dressing
  // Old Money Work in blazers and cardigans. Only the condition is replaced.
  const measured = typeof weather?.temp === 'number' && Number.isFinite(weather.temp)
    ? {temp: weather.temp} : {};
  if (/\b(freezing|snow|winter|sub[ -]?zero|qish)\b|мороз|зим[ауы]|қиш/i.test(text))
    return {...measured,condition: 'requested winter season'};
  if (/\b(summer|yoz)\b|лет[оа]/i.test(text))
    return {...measured,condition: 'requested summer season'};
  return weather || undefined;
}

export function needsLayering(
  style: string | null | undefined,
  weather?: CompositionWeather | null,
  prompt?: string | null,
): boolean {
  return resolveOutfitComposition(style, weather, prompt).layered;
}
