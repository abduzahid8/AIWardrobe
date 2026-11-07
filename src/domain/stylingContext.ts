/** Request intent is independent of catalog identity, I/O and UI. */
import { normalizeStyleId } from './styleInference.ts';
import type { BudgetFxContext } from './budgetFx.ts';
export interface StylingPreferences {
  favoriteColors?: readonly string[];
  avoidColors?: readonly string[];
  likedPatterns?: readonly string[];
  dislikedPatterns?: readonly string[];
  fitPreference?: string;
}
export interface StylingContext {
  ruleVersion: 'stylist-quality-v1';
  style: string;
  occasion: string;
  dressCode: 'formal' | 'smart' | 'casual' | 'athletic';
  activity: string;
  request: string;
  weather: { temp?: number; condition?: string } | null;
  anchorId: string | null;
  sourceMode: 'wardrobe' | 'catalog' | 'mixed';
  budget?: { amount: number; currency?: string; fx?: BudgetFxContext };
  preferences?: StylingPreferences;
  uncertainty: string[];
  /** Private aggregate concept preferences; soft and bounded, never admission. */
  conceptFeedback?: Readonly<Record<string,number>>;
}
export function isDateNightOccasion(occasion?: string | null): boolean {
  return /\bdate\b|свидан/i.test(String(occasion || '').replace(/[_-]+/g, ' '));
}
export function stylingContext(input: {
  style?: string | null;
  occasion?: string | null;
  request?: string | null;
  weather?: { temp?: number | null; condition?: string | null } | null;
  anchorId?: string | null;
  sourceMode?: StylingContext['sourceMode'];
  budget?: StylingContext['budget'];
  preferences?: StylingPreferences;
}): StylingContext {
  const occasion = (input.occasion || '').toLowerCase().trim();
  const request = input.request || '';
  const athletic = /\b(workout|gym|training|fitness|sport)\b|трениров|спортзал/.test(occasion);
  const formal = /\bformal\b|black[ -]?tie|формаль|торжеств/.test(occasion);
  const smart = /\b(work|office|business)\b|работ/.test(occasion) || isDateNightOccasion(occasion);
  const weather =
    input.weather && (Number.isFinite(input.weather.temp) || !!input.weather.condition)
      ? {
          condition: input.weather.condition || undefined,
          temp: Number.isFinite(input.weather.temp) ? (input.weather.temp as number) : undefined,
        }
      : null;
  return {
    ruleVersion: 'stylist-quality-v1',
    style: normalizeStyleId(input.style || 'casual'),
    occasion,
    dressCode: athletic ? 'athletic' : formal ? 'formal' : smart ? 'smart' : 'casual',
    activity: occasion,
    request,
    weather,
    anchorId: input.anchorId || null,
    sourceMode: input.sourceMode || 'mixed',
    budget: input.budget,
    preferences: input.preferences
      ? {
          ...input.preferences,
          avoidColors: input.preferences.avoidColors
            ? [...input.preferences.avoidColors]
            : undefined,
          dislikedPatterns: input.preferences.dislikedPatterns
            ? [...input.preferences.dislikedPatterns]
            : undefined,
        }
      : undefined,
    uncertainty: [
      ...(!weather ? ['weather_unknown'] : []),
      ...(!occasion ? ['occasion_unknown'] : []),
    ],
  };
}
