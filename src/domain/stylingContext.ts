/** Request intent is independent of catalog identity, I/O and UI. */
import { normalizeStyleId } from './styleInference.ts';
import type { BudgetFxContext } from './budgetFx.ts';
export type StylingPresentation = 'menswear' | 'womenswear' | 'unisex';

export interface StylingBodyProfile {
  gender?: string | null;
  heightCm?: number | null;
  weightKg?: number | null;
  bodyType?: string | null;
  measurements?: Record<string, number> | null;
  fitPreference?: string | null;
}

export interface StylingAppearanceProfile {
  consentGranted: boolean;
  skinUndertone?: string;
  skinDepth?: string;
  contrastLevel?: string;
  colorSeason?: string;
  flatteringColors?: string[];
  avoidColors?: string[];
  metalTone?: string;
  photoSha256?: string;
}

export interface OwnedWardrobeSummary {
  totalItems: number;
  dominantColors: string[];
  dominantCategories: string[];
  topBrands: string[];
  formalityMix: Record<string, number>;
}

export interface StylingPreferences {
  favoriteColors?: readonly string[];
  avoidColors?: readonly string[];
  likedPatterns?: readonly string[];
  dislikedPatterns?: readonly string[];
  fitPreference?: string;
}
export interface StylingContext {
  legacyFourSlot?: boolean;
  optionalLayer?: boolean;
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
  presentation?: StylingPresentation;
  bodyProfile?: StylingBodyProfile;
  appearanceProfile?: StylingAppearanceProfile;
  ownedWardrobeSummary?: OwnedWardrobeSummary;
  dislikedPatterns?: string[];
  uncertainty: string[];
  /** Private aggregate concept preferences; soft and bounded, never admission. */
  conceptFeedback?: Readonly<Record<string, number>>;
  /** Display language the produced copy (deck headlines/descriptions) must
   * be written in. Anything unsupported degrades to English. */
  language?: 'en' | 'ru' | 'uz';
}
export function isDateNightOccasion(occasion?: string | null): boolean {
  return /\bdate\b|свидан/i.test(String(occasion || '').replace(/[_-]+/g, ' '));
}
/** The copy languages the deck editorials support. Anything else — including
 * regional variants like 'ru-RU' — degrades to English rather than mixing. */
const STYLIST_COPY_LANGUAGES = ['en', 'ru', 'uz'] as const;
export type StylistCopyLanguage = typeof STYLIST_COPY_LANGUAGES[number];
export function normalizeStylistLanguage(value: unknown): StylistCopyLanguage {
  const raw = String(value || '').trim().toLowerCase();
  return (STYLIST_COPY_LANGUAGES as readonly string[]).find((lng) => raw === lng || raw.startsWith(`${lng}-`)) as StylistCopyLanguage | undefined
    || 'en';
}
export function stylingContext(input: {
  optionalLayer?: boolean;
  style?: string | null;
  occasion?: string | null;
  request?: string | null;
  weather?: { temp?: number | null; condition?: string | null } | null;
  anchorId?: string | null;
  sourceMode?: StylingContext['sourceMode'];
  budget?: StylingContext['budget'];
  preferences?: StylingPreferences;
  presentation?: StylingPresentation | null;
  bodyProfile?: StylingBodyProfile | null;
  appearanceProfile?: StylingAppearanceProfile | null;
  ownedWardrobeSummary?: OwnedWardrobeSummary | null;
  dislikedPatterns?: readonly string[];
  language?: StylistCopyLanguage | string | null;
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
  const presentation =
    input.presentation === 'menswear' || input.presentation === 'womenswear' || input.presentation === 'unisex'
      ? input.presentation
      : undefined;
  const bodyProfile = input.bodyProfile
    ? {
        ...input.bodyProfile,
        measurements: input.bodyProfile.measurements
          ? { ...input.bodyProfile.measurements }
          : input.bodyProfile.measurements,
      }
    : undefined;
  const appearanceProfile = input.appearanceProfile
    ? input.appearanceProfile.consentGranted === true
      ? {
          consentGranted: true,
          ...(input.appearanceProfile.skinUndertone ? { skinUndertone: input.appearanceProfile.skinUndertone } : {}),
          ...(input.appearanceProfile.skinDepth ? { skinDepth: input.appearanceProfile.skinDepth } : {}),
          ...(input.appearanceProfile.contrastLevel ? { contrastLevel: input.appearanceProfile.contrastLevel } : {}),
          ...(input.appearanceProfile.colorSeason ? { colorSeason: input.appearanceProfile.colorSeason } : {}),
          ...(input.appearanceProfile.flatteringColors
            ? { flatteringColors: [...input.appearanceProfile.flatteringColors] }
            : {}),
          ...(input.appearanceProfile.avoidColors
            ? { avoidColors: [...input.appearanceProfile.avoidColors] }
            : {}),
          ...(input.appearanceProfile.metalTone ? { metalTone: input.appearanceProfile.metalTone } : {}),
          ...(input.appearanceProfile.photoSha256 ? { photoSha256: input.appearanceProfile.photoSha256 } : {}),
        }
      : { consentGranted: false }
    : undefined;
  const ownedWardrobeSummary = input.ownedWardrobeSummary
    ? {
        totalItems: input.ownedWardrobeSummary.totalItems,
        dominantColors: [...input.ownedWardrobeSummary.dominantColors],
        dominantCategories: [...input.ownedWardrobeSummary.dominantCategories],
        topBrands: [...input.ownedWardrobeSummary.topBrands],
        formalityMix: { ...input.ownedWardrobeSummary.formalityMix },
      }
    : undefined;
  const dislikedPatterns = Array.isArray(input.dislikedPatterns)
    ? [...input.dislikedPatterns]
    : undefined;
  return {
    optionalLayer: input.optionalLayer === true,
    ruleVersion: 'stylist-quality-v1',
    style: normalizeStyleId(input.style || 'casual'),
    occasion,
    dressCode: athletic ? 'athletic' : formal ? 'formal' : smart ? 'smart' : 'casual',
    activity: occasion,
    request,
    weather,
    anchorId: input.anchorId || null,
    sourceMode: input.sourceMode || 'mixed',
    language: normalizeStylistLanguage(input.language),
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
    ...(presentation ? { presentation } : {}),
    ...(bodyProfile ? { bodyProfile } : {}),
    ...(appearanceProfile ? { appearanceProfile } : {}),
    ...(ownedWardrobeSummary ? { ownedWardrobeSummary } : {}),
    ...(dislikedPatterns ? { dislikedPatterns } : {}),
    uncertainty: [
      ...(!weather ? ['weather_unknown'] : []),
      ...(!occasion ? ['occasion_unknown'] : []),
    ],
  };
}
