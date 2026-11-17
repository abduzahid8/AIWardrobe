import { inferItemAttributes } from '../../../features/outfit-generator/utils/styleInference';
import { fitCategory, fitSubcategory, productFitCategory } from './fitCategory';
import type {
  FitEvidence,
  FitProfile,
  BrandSizePrior,
  SizeRecommendation,
  PhysicalDataEvidence,
} from '../../types/purchase/fit';
import type { ShopCatalogItem } from '../../types/shop';
import type { BodyProfile } from '../../types/bodyProfile';
import type { GarmentPhysicalProfile } from '../../types/garment';
import { normalizeCatalogBrand, normalizeIdentityText } from '../../utils/catalogIdentity';
import { inferShopRowMacro } from '../../utils/shopCatalogOrder';
import { assessFit } from '../../lib/fit/fitEngine';
const DAY = 86400000;
export function evidenceWeight(e: FitEvidence, now = Date.now()): number {
  if (!Number.isFinite(e.confidence)) return 0;
  const age = Math.max(0, (now - Date.parse(e.createdAt)) / DAY);
  if (!Number.isFinite(age) || Date.parse(e.createdAt) > now + DAY) return 0;
  const source = e.source === 'body_profile' ? 0.25 : e.source === 'reference_garment' ? 1 : 0.9;
  return Math.max(0, Math.min(1, e.confidence)) * Math.pow(0.5, age / 365) * source;
}
export function deriveFitProfile(evidence: readonly FitEvidence[], now = Date.now()): FitProfile {
  const owner = evidence.find((e) => e && typeof e.userId === 'string' && e.userId)?.userId;
  const unique = new Map<string, FitEvidence>();
  for (const e of evidence.slice(-160)) {
    if (
      !e ||
      e.userId !== owner ||
      typeof e.category !== 'string' ||
      !e.category ||
      !Number.isFinite(e.confidence) ||
      !Number.isFinite(Date.parse(e.createdAt))
    )
      continue;
    // latest feedback for one garment/size is one observation
    const key = JSON.stringify([e.userId, e.productId || e.id, e.brand, e.category, e.sizeLabel]);
    const prior = unique.get(key);
    if (!prior || Date.parse(e.createdAt) >= Date.parse(prior.createdAt)) unique.set(key, e);
  }
  const rows = [...unique.values()],
    buckets = new Map<string, BrandSizePrior>();
  const ease = new Map<string, Map<string, number>>();
  const zones = new Map<string, Map<string, number>>();
  for (const e of rows) {
    const w = evidenceWeight(e, now),
      category = fitCategory(e.category),
      brand = normalizeCatalogBrand(e.brand || '');
    if (w <= 0 || e.overallFit === 'unknown') continue;
    if (
      (e.source !== 'body_profile' && e.preferredFit) ||
      (e.source !== 'body_profile' && e.overallFit === 'perfect' && !!e.cut && !e.returned)
    ) {
      const preference =
        e.preferredFit ||
        (/slim|snug/i.test(e.cut || '')
          ? 'snug'
          : /relaxed|oversized/i.test(e.cut || '')
            ? 'relaxed'
            : 'regular');
      if (!ease.has(category)) ease.set(category, new Map());
      const map = ease.get(category)!;
      map.set(preference, (map.get(preference) || 0) + w);
    }
    for (const [zone, outcome] of Object.entries(e.zones || {})) {
      if (e.source === 'body_profile' || outcome !== 'good' || e.overallFit !== 'perfect') continue;
      if (!zones.has(zone)) zones.set(zone, new Map());
      const m = zones.get(zone)!;
      m.set(outcome, (m.get(outcome) || 0) + w);
    }
    if (!brand || !e.sizeLabel) continue;
    const materialKey = (Array.isArray(e.materials) ? e.materials : [])
      .map(normalizeIdentityText)
      .sort()
      .join('|');
    const subcategory = fitSubcategory(e.subcategory || e.category);
    const key = JSON.stringify([
      brand,
      category,
      subcategory,
      e.sizeLabel,
      e.cut || '',
      materialKey,
      e.stretch || '',
    ]);
    if (!buckets.has(key))
      buckets.set(key, {
        brand,
        category,
        subcategory,
        sizeLabel: e.sizeLabel,
        cut: e.cut,
        materialKey,
        stretch: e.stretch,
        evidenceIds: [],
        sources: [],
        effectiveCount: 0,
        positiveWeight: 0,
        negativeWeight: 0,
        zoneNotes: {},
      });
    const b = buckets.get(key)!;
    b.evidenceIds.push(e.id);
    b.sources = [...new Set([...b.sources, e.source])];
    b.effectiveCount += w;
    if (
      ['too_small', 'too_large'].includes(e.overallFit) ||
      (e.returned && /fit|size|tight|loose|размер/i.test(e.returnReason || ''))
    )
      b.negativeWeight += w;
    else if (e.overallFit === 'perfect' || (e.kept && ['snug', 'relaxed'].includes(e.overallFit)))
      b.positiveWeight += w;
    for (const [zone, outcome] of Object.entries(e.zones || {})) {
      const key = zone as keyof typeof b.zoneNotes;
      b.zoneNotes[key] = [...new Set([...(b.zoneNotes[key] || []), outcome])];
    }
  }
  const references = rows
    .filter(
      (e) =>
        e.source !== 'body_profile' &&
        e.brand &&
        e.sizeLabel &&
        e.category &&
        !e.returned &&
        !e.sold &&
        e.confidence >= 0.7 &&
        ['perfect', 'snug', 'relaxed'].includes(e.overallFit) &&
        (e.reference || e.overallFit === 'perfect')
    )
    .map((e) => ({
      evidenceId: e.id,
      productId: e.productId,
      brand: normalizeCatalogBrand(e.brand!),
      category: fitCategory(e.category),
      subcategory: fitSubcategory(e.subcategory || e.category),
      productFamilyKey: e.productFamilyKey,
      sizeLabel: e.sizeLabel!,
      cut: e.cut,
      materials: e.materials,
      stretch: e.stretch,
      zones: e.zones,
      physicalProfile: e.physicalProfile,
      confidence: evidenceWeight(e, now),
    }));
  const preferredEaseByCategory: FitProfile['preferredEaseByCategory'] = {};
  for (const [category, values] of ease)
    preferredEaseByCategory[category] = [...values].sort((a, b) => b[1] - a[1])[0][0] as 'regular';
  const preferredEaseByZone: FitProfile['preferredEaseByZone'] = {};
  for (const [zone, values] of zones)
    preferredEaseByZone[zone as keyof typeof preferredEaseByZone] = [...values].sort(
      (a, b) => b[1] - a[1]
    )[0][0] as 'good';
  const weight = rows
    .filter(
      (e) =>
        e.source !== 'body_profile' && (e.preferredFit || (e.overallFit === 'perfect' && e.cut))
    )
    .reduce((n, e) => n + evidenceWeight(e, now), 0);
  return {
    version: 1,
    userId: owner,
    computedAt: new Date(now).toISOString(),
    evidenceCount: rows.length,
    lastUpdatedAt:
      rows
        .map((e) => e.createdAt)
        .sort()
        .at(-1) || null,
    fitPreferenceConfidence: weight >= 5 ? 'high' : weight >= 2 ? 'medium' : 'low',
    preferredEaseByCategory,
    preferredEaseByZone,
    brandCategorySizePriors: [...buckets.values()].slice(0, 80),
    referenceGarments: references.sort((a, b) => b.confidence - a.confidence).slice(0, 24),
  };
}
export function garmentPhysicalData(
  item: ShopCatalogItem,
  size: string,
  options: {
    exactCharts?: GarmentPhysicalProfile[];
    merchantVariants?: GarmentPhysicalProfile[];
    references?: FitProfile['referenceGarments'];
    categoryPrior?: { category: string; cut?: string; provenance: string };
  } = {}
): PhysicalDataEvidence {
  const exact = (profiles?: GarmentPhysicalProfile[]) =>
    profiles?.find(
      (p) =>
        p.garmentId === item.id &&
        p.sizeLabel === size &&
        Object.values(p.measurements).some((m) => m && Number.isFinite(m.valueCm) && m.valueCm > 0)
    );
  for (const [source, profiles] of [
    ['exact_chart', options.exactCharts],
    ['merchant_variant', options.merchantVariants],
    ['stored_profile', item.physicalProfiles],
  ] as const) {
    const profile = exact(profiles);
    if (profile)
      return {
        source,
        profile,
        confidence: source === 'stored_profile' ? 'medium' : 'high',
        notes: [`Exact ${size} garment measurements from ${source}`],
      };
  }
  const reference = options.references?.find(
    (r) => r.productId === item.id && r.sizeLabel === size
  );
  if (reference)
    return {
      source: 'reference',
      profile:
        reference.physicalProfile?.garmentId === item.id &&
        reference.physicalProfile.sizeLabel === size
          ? reference.physicalProfile
          : undefined,
      confidence: 'medium',
      notes: ['Same concrete garment and size is a fit reference; measurements remain unknown'],
    };
  if (
    options.categoryPrior &&
    fitCategory(options.categoryPrior.category) === productFitCategory(item)
  )
    return {
      source: 'category_prior',
      confidence: 'low',
      notes: [
        `Category/cut prior from ${options.categoryPrior.provenance}; exact measurements remain unknown`,
      ],
    };
  const inferredCut = `${item.name} ${item.description || ''}`.match(
    /\b(slim|regular|relaxed|oversized)[ -]?fit\b/i
  )?.[0];
  if (inferredCut)
    return {
      source: 'text',
      confidence: 'low',
      notes: [`Text describes ${inferredCut}; no garment centimeters inferred`],
    };
  return {
    source: 'unknown',
    confidence: 'low',
    notes: ['Exact garment measurements are unavailable; no centimeters inferred'],
  };
}
export function recommendPurchaseSize(
  item: ShopCatalogItem,
  profile?: FitProfile,
  body?: BodyProfile,
  now = Date.now()
): SizeRecommendation {
  if (profile?.userId && body?.userId && profile.userId !== body.userId) profile = undefined;
  const category = productFitCategory(item);
  const subtype = fitSubcategory(item.name),
    materials = inferItemAttributes(item).materials;
  const brand = normalizeCatalogBrand(item.brand),
    cut = normalizeIdentityText(
      item.description?.match(/\b(slim|regular|relaxed|oversized)[ -]?fit\b/i)?.[0] ||
        item.name.match(/\b(slim|regular|relaxed|oversized)[ -]?fit\b/i)?.[0] ||
        ''
    );
  const aggregateAge = profile
    ? Math.max(
        0,
        (now -
          Date.parse(profile.computedAt || profile.lastUpdatedAt || new Date(now).toISOString())) /
          DAY
      )
    : 0;
  const aggregateFreshness = Number.isFinite(aggregateAge) ? Math.pow(0.5, aggregateAge / 365) : 0;
  const candidates = new Map<
    string,
    {
      support: number;
      ids: string[];
      notes: string[];
      zones: string[];
      provenance: string[];
      conflict: number;
    }
  >();
  const sizes = item.availableSizes || [
    ...new Set([
      ...(item.physicalProfiles || []).map((p) => p.sizeLabel),
      ...(profile?.brandCategorySizePriors || [])
        .filter((p) => p.brand === brand && p.category === category)
        .map((p) => p.sizeLabel),
    ]),
  ];
  for (const size of sizes.slice(0, 30)) {
    let support = 0,
      conflict = 0;
    const ids: string[] = [],
      notes: string[] = [],
      zones: string[] = [],
      provenance: string[] = [];
    for (const prior of profile?.brandCategorySizePriors || []) {
      if (
        prior.brand !== brand ||
        prior.category !== category ||
        prior.sizeLabel !== size ||
        (prior.subcategory && subtype && prior.subcategory !== subtype)
      )
        continue;
      const cutPenalty =
        !cut || !prior.cut
          ? 0.65
          : normalizeIdentityText(prior.cut)
                .replace(/\bfit\b/g, '')
                .trim() === cut.replace(/\bfit\b/g, '').trim()
            ? 1
            : 0.3;
      const materialPenalty =
        prior.materialKey && materials.length
          ? materials.some((material) => prior.materialKey.split('|').includes(material))
            ? 0.9
            : 0.25
          : 0.6;
      const actualStretch = item.physicalProfiles?.find((p) => p.sizeLabel === size)?.stretch;
      const weight =
        cutPenalty *
        materialPenalty *
        (prior.subcategory && subtype ? 1 : 0.7) *
        (prior.stretch && actualStretch && prior.stretch === actualStretch ? 1 : 0.8);
      support += prior.positiveWeight * weight * aggregateFreshness;
      conflict += prior.negativeWeight * weight * aggregateFreshness;
      ids.push(...prior.evidenceIds);
      provenance.push('brand/category feedback', ...(prior.sources || []));
      for (const [zone, outcomes] of Object.entries(prior.zoneNotes))
        for (const outcome of outcomes || [])
          if (outcome !== 'good' && outcome !== 'unknown')
            zones.push(`${zone}: previously ${outcome}`);
    }
    for (const ref of profile?.referenceGarments || [])
      if (
        ref.brand === brand &&
        ref.category === category &&
        ref.sizeLabel === size &&
        (!ref.subcategory || !subtype || ref.subcategory === subtype)
      ) {
        const exact = ref.productId === item.id;
        support += ref.confidence * aggregateFreshness * (exact ? 2 : 0.25);
        ids.push(ref.evidenceId);
        notes.push(
          exact
            ? 'This exact garment and size is a reference'
            : 'A reference in this brand/category supports this size'
        );
        provenance.push('reference garment');
      }
    const physical = garmentPhysicalData(item, size, { references: profile?.referenceGarments });
    if (body?.status === 'ready' && physical.profile) {
      const result = assessFit(body, physical.profile);
      if (result.overall !== 'unknown' && result.confidence !== 'low') {
        support += result.overall === 'good_fit' ? 1.2 : result.overall === 'relaxed' ? 0.7 : 0;
        conflict += ['too_small', 'too_large', 'tight'].includes(result.overall) ? 1 : 0;
        notes.push(`${size}: measurement engine ${result.overall}`);
        provenance.push(physical.source);
      }
    }
    candidates.set(size, {
      support: support - conflict,
      ids: [...new Set(ids)],
      notes,
      zones,
      provenance: [...new Set(provenance)],
      conflict,
    });
  }
  const ranked = [...candidates].sort(
    (a, b) => b[1].support - a[1].support || a[0].localeCompare(b[0])
  );
  const best = ranked[0],
    margin = best ? best[1].support - (ranked[1]?.[1].support || 0) : 0;
  if (!best || best[1].support <= 0)
    return {
      confidence: 'low',
      notes: ['Not enough relevant fit evidence to recommend a size'],
      zoneNotes: [],
      evidenceIds: [],
      evidenceCount: 0,
      provenance: [],
      alternatives: [],
    };
  const [size, value] = best,
    exactReference = profile?.referenceGarments.some(
      (r) =>
        r.productId === item.id && r.sizeLabel === size && r.confidence * aggregateFreshness >= 0.7
    );
  const confidence =
    value.conflict > 0.4 ||
    margin < 0.25 ||
    (value.provenance.includes('body_profile') &&
      !value.provenance.some((source) =>
        [
          'owned',
          'purchase',
          'try_on',
          'manual',
          'reference_garment',
          'stored_profile',
          'exact_chart',
          'merchant_variant',
        ].includes(source)
      ))
      ? 'low'
      : exactReference ||
          (value.ids.length >= 4 &&
            value.support >= 2 &&
            !!cut &&
            materials.length > 0 &&
            !!item.physicalProfiles?.some((p) => p.sizeLabel === size && p.stretch))
        ? 'high'
        : value.ids.length >= 2 || value.provenance.includes('stored_profile')
          ? 'medium'
          : 'low';
  return {
    recommendedSize: size,
    confidence,
    expectedFitIntent: profile?.preferredEaseByCategory[category],
    notes: [
      ...value.notes,
      `Evidence is brand/category specific; material/cut uncertainty reduces confidence`,
    ],
    zoneNotes: [...new Set(value.zones)],
    evidenceIds: value.ids,
    evidenceCount: value.ids.length,
    provenance: value.provenance,
    alternatives: ranked.map(([sizeLabel, v]) => ({
      sizeLabel,
      support: Math.round(v.support * 100) / 100,
    })),
  };
}
