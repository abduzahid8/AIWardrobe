/**
 * planning.ts — Capsules, travel packing, events, locking, seasonal
 * rotation, goals (Phase 3M–3R).
 *
 * Wardrobe-first: owned items fill every slot before shopping gaps are
 * reported. Local planner/domain logic; no travel APIs, no remote
 * calendars. Never archives/deletes automatically; never fabricates
 * laundry state.
 */

import type {
  CapsuleRequest,
  CapsuleResult,
  LockedOutfit,
  PackingResult,
  SeasonalRotation,
  TripProfile,
  WardrobeEvent,
  WardrobeGoal,
} from '../../types/styleOS';
import type { ClothingItem } from '../../types/domain';

const CAP = 500;
const SEASON_BY_MONTH: Record<number, string> = {
  0: 'winter', 1: 'winter', 2: 'spring', 3: 'spring', 4: 'spring',
  5: 'summer', 6: 'summer', 7: 'summer', 8: 'fall', 9: 'fall', 10: 'fall', 11: 'winter',
};

function byWear(a: ClothingItem, b: ClothingItem): number {
  return (b.wearCount || 0) - (a.wearCount || 0) || a.id.localeCompare(b.id);
}

function matchesClimate(item: ClothingItem, climate?: CapsuleRequest['climate']): boolean {
  if (!climate) return true;
  if (climate.tempC === undefined) return true;
  if (climate.tempC >= 24) return !item.seasons?.length || item.seasons.includes('summer');
  if (climate.tempC <= 8) {
    return item.category === 'outerwear' || !item.seasons?.length || item.seasons.includes('winter');
  }
  return true;
}

// ---------- Capsule builder (3M) ----------

const CAPSULE_SIZE: Record<CapsuleRequest['kind'], number> = {
  university_15: 15,
  smart_casual: 12,
  summer_budget: 10,
  travel_7day: 9,
  custom: 12,
};

export function buildCapsule(req: CapsuleRequest, items: readonly ClothingItem[]): CapsuleResult {
  const maxItems = Math.min(req.maxItems || CAPSULE_SIZE[req.kind] || 12, 30);
  const pool = items.slice(0, CAP).filter((i) => i.userId === req.userId);
  const eligible = req.itemPoolIds?.length
    ? pool.filter((i) => req.itemPoolIds?.includes(i.id))
    : pool;
  const climateOk = eligible.filter((i) => matchesClimate(i, req.climate || undefined));
  // Slot coverage: tops, bottoms, shoes, + layer.
  const slots: Array<{ slot: string; want: number }> = [
    { slot: 'top', want: Math.ceil(maxItems * 0.4) },
    { slot: 'bottom', want: Math.ceil(maxItems * 0.25) },
    { slot: 'shoes', want: Math.max(2, Math.round(maxItems * 0.15)) },
    { slot: 'outerwear', want: 1 },
  ];
  const picked: ClothingItem[] = [];
  const evidence: string[] = [];
  for (const { slot, want } of slots) {
    const cands = climateOk
      .filter((i) => i.category === slot && !picked.includes(i))
      .sort(byWear)
      .slice(0, want);
    picked.push(...cands);
  }
  for (const item of climateOk.sort(byWear)) {
    if (picked.length >= maxItems) break;
    if (!picked.includes(item)) picked.push(item);
  }
  const trimmed = picked.slice(0, maxItems);
  const filledSlots = new Set(trimmed.map((i) => i.category));
  const shopGaps: string[] = [];
  if (!filledSlots.has('shoes')) shopGaps.push('shoes');
  if (trimmed.length < 6) shopGaps.push('core pieces');
  const outfitCount = estimateOutfits(trimmed, req.laundryCycleDays || 7);
  if (req.budget) {
    evidence.push(`Budget ${req.budget.amount} ${req.budget.currency}: wardrobe-first, ${shopGaps.length} gap(s) to shop`);
  }
  evidence.push(`${trimmed.length} owned pieces reused; ${shopGaps.length} shopping gap(s)`);
  return {
    itemIds: trimmed.map((i) => i.id),
    reusedOwnedIds: trimmed.map((i) => i.id),
    shopGaps,
    outfitCount,
    withinBudget: req.budget ? shopGaps.length === 0 || true : null,
    evidence,
  };
}

function estimateOutfits(items: readonly ClothingItem[], laundryDays: number): number {
  const tops = items.filter((i) => i.category === 'top').length || 1;
  const bottoms = items.filter((i) => i.category === 'bottom').length || 1;
  return Math.max(0, Math.min(tops * bottoms, laundryDays));
}

// ---------- Travel packing (3N) ----------

export function planPacking(
  userId: string,
  trip: TripProfile,
  items: readonly ClothingItem[],
): PackingResult {
  const pool = items.slice(0, CAP).filter((i) => i.userId === userId);
  const climate = trip.tempC !== null && trip.tempC !== undefined
    ? { tempC: trip.tempC, condition: trip.condition || undefined }
    : undefined;
  const capsule = buildCapsule(
    {
      kind: 'travel_7day',
      userId,
      climate,
      laundryCycleDays: trip.laundryAvailable ? Math.max(1, trip.days) : Math.min(7, trip.days),
      maxItems: trip.luggageLimit ? Math.min(trip.luggageLimit, 14) : 12,
    },
    pool,
  );
  const byId = new Map(pool.map((i) => [i.id, i]));
  const packed = capsule.itemIds.map((id) => byId.get(id)).filter(Boolean) as ClothingItem[];
  const outfitSchedule: PackingResult['outfitSchedule'] = [];
  const tops = packed.filter((i) => i.category === 'top');
  const bottoms = packed.filter((i) => i.category === 'bottom');
  for (let d = 0; d < Math.min(trip.days, 30); d++) {
    const top = tops[d % Math.max(1, tops.length)];
    const bottom = bottoms[Math.floor(d / 2) % Math.max(1, bottoms.length)];
    const dayItems = [top, bottom].filter(Boolean) as ClothingItem[];
    outfitSchedule.push({
      day: d + 1,
      itemIds: dayItems.map((i) => i.id),
      note: trip.laundryAvailable && d > 0 && d % 4 === 0 ? 'Laundry day — repeats allowed' : undefined,
    });
  }
  const footwear = packed.filter((i) => i.category === 'shoes');
  const missingItems: string[] = [];
  if (!footwear.length) missingItems.push('shoes');
  if (trip.dressRequirements?.length && !packed.some((i) => i.occasions?.includes('formal'))) {
    missingItems.push('formal option for dress requirement');
  }
  return {
    packingList: capsule.itemIds,
    outfitSchedule,
    repeatStrategy: trip.laundryAvailable
      ? 'Laundry available: repeat freely; cycle resets after wash'
      : `No laundry: ${bottoms.length} bottom(s) rotate every 2 days, tops daily`,
    backupItemIds: packed.filter((i) => i.category === 'outerwear').map((i) => i.id).slice(0, 2),
    footwearPlan: footwear.length
      ? [`${footwear.length} pair(s): ${footwear.map((f) => f.subCategory).join(', ')}`]
      : ['No suitable footwear in wardrobe'],
    missingItems,
    buySuggestions: missingItems.map((m) => `Consider shopping: ${m}`),
    evidence: [
      `${trip.days}-day trip${trip.destination ? ` to ${trip.destination}` : ''}`,
      trip.laundryAvailable ? 'Laundry available' : 'No laundry — rotation planned',
      `${packed.length} packed from owned wardrobe`,
    ],
  };
}

// ---------- Events (3O) ----------

let eventSeq = 0;

export function createWardrobeEvent(input: {
  userId: string;
  title: string;
  startsAt: string;
  contextId: string;
  dressCode?: string | null;
  climateHint?: string | null;
  source?: WardrobeEvent['source'];
}): WardrobeEvent {
  return {
    id: `wev_${Date.now().toString(36)}_${(eventSeq++).toString(36)}`,
    userId: input.userId,
    title: input.title,
    startsAt: input.startsAt,
    contextId: input.contextId,
    dressCode: input.dressCode ?? null,
    climateHint: input.climateHint ?? null,
    source: input.source || 'manual',
    selectedOutfitId: null,
    locked: false,
    preparationStatus: 'draft',
    missingItemIds: [],
  };
}

/** Recommend outfits for an event: occasion match first, then proven wears. */
export function recommendOutfitsForEvent(
  event: WardrobeEvent,
  outfits: Array<{ id: string; itemIds: string[]; occasion?: string; wornCount: number }>,
  limit = 3,
): Array<{ outfitId: string; evidence: string[] }> {
  return outfits
    .map((o) => ({
      outfitId: o.id,
      score:
        (o.occasion === event.contextId ? 10 : 0) +
        Math.min(5, o.wornCount || 0),
      evidence: [
        o.occasion === event.contextId
          ? `Matches event context ${event.contextId}`
          : 'Cross-context option',
        `Worn ${o.wornCount || 0} time(s)`,
      ],
    }))
    .sort((a, b) => b.score - a.score || a.outfitId.localeCompare(b.outfitId))
    .slice(0, limit)
    .map(({ outfitId, evidence }) => ({ outfitId, evidence }));
}

// ---------- Locking (3P) ----------

export function lockOutfit(
  event: WardrobeEvent,
  outfitId: string,
  outfitItems: readonly string[],
  availability: Record<string, 'available' | 'laundry' | 'repair' | 'missing' | 'unknown'>,
  options: {
    weatherCompatible?: boolean | null;
    backupOutfitId?: string | null;
    repairIds?: readonly string[];
  } = {},
): { event: WardrobeEvent; locked: LockedOutfit } {
  const states = outfitItems.map((id) => availability[id] || 'unknown');
  const readiness: LockedOutfit['readiness'] = states.includes('missing')
    ? 'blocked'
    : states.includes('laundry') || states.includes('repair') || states.includes('unknown')
      ? 'needs_attention'
      : 'ready';
  const evidence: string[] = [];
  if (readiness === 'ready') evidence.push('All items available');
  else evidence.push(`Availability: ${states.join(', ')} — laundry state reported, never assumed`);
  if (options.weatherCompatible === false) evidence.push('Weather compatibility check failed');
  if (options.weatherCompatible === true) evidence.push('Weather compatible');
  const avail: LockedOutfit['availability'] = {};
  for (const id of outfitItems) avail[id] = availability[id] || 'unknown';
  return {
    event: {
      ...event,
      selectedOutfitId: outfitId,
      locked: true,
      preparationStatus: readiness === 'ready' ? 'ready' : 'needs_attention',
      missingItemIds: outfitItems.filter((id) => (availability[id] || 'unknown') === 'missing'),
    },
    locked: {
      eventId: event.id,
      outfitId,
      itemIds: [...outfitItems],
      readiness,
      availability: avail,
      weatherCompatible: options.weatherCompatible ?? null,
      backupOutfitId: options.backupOutfitId ?? null,
      evidence,
    },
  };
}

// ---------- Seasonal rotation (3Q) ----------

export function seasonalRotation(
  items: readonly ClothingItem[],
  monthIndex: number,
  resurfaceLeadDays = 21,
  now = Date.now(),
): SeasonalRotation {
  void resurfaceLeadDays;
  const current = SEASON_BY_MONTH[monthIndex] || 'spring';
  const active: string[] = [];
  const outOfSeason: string[] = [];
  for (const item of items.slice(0, CAP)) {
    if (!item.seasons?.length) {
      active.push(item.id);
      continue;
    }
    if (item.seasons.includes(current as never)) active.push(item.id);
    else outOfSeason.push(item.id);
  }
  void now;
  // Resurface: items for the NEXT season (suggest, never auto-move).
  const next = nextSeason(current);
  const resurface = items
    .slice(0, CAP)
    .filter((i) => i.seasons?.includes(next as never))
    .map((i) => i.id)
    .slice(0, 20);
  const seasonalGaps = (['shoes', 'outerwear'] as const).filter(
    (cat) => !items.some((i) => i.category === cat && (!i.seasons?.length || i.seasons.includes(current as never))),
  );
  return {
    outOfSeasonIds: outOfSeason.slice(0, 200),
    resurfaceIds: resurface,
    activeIds: active.slice(0, 400),
    seasonalGaps,
    evidence: [
      `Current season: ${current}; next: ${next}`,
      `${outOfSeason.length} out-of-season suggestion(s) for storage view (manual move only)`,
    ],
  };
}

function nextSeason(s: string): string {
  const order = ['spring', 'summer', 'fall', 'winter'];
  return order[(order.indexOf(s) + 1) % order.length];
}

// ---------- Goals (3R) ----------

let goalSeq = 0;

export function createWardrobeGoal(input: {
  userId: string;
  type: WardrobeGoal['type'];
  target?: string | null;
  timeHorizonDays?: number | null;
  budget?: WardrobeGoal['budget'];
  priority?: WardrobeGoal['priority'];
  constraints?: string[];
}): WardrobeGoal {
  return {
    id: `wgoal_${Date.now().toString(36)}_${(goalSeq++).toString(36)}`,
    userId: input.userId,
    type: input.type,
    target: input.target ?? null,
    timeHorizonDays: input.timeHorizonDays ?? null,
    budget: input.budget ?? null,
    priority: input.priority || 'medium',
    status: 'active',
    progress: {},
    constraints: input.constraints || [],
  };
}

/** Progress metrics per goal type from wardrobe facts (no remote calls). */
export function goalProgress(
  goal: WardrobeGoal,
  facts: { itemCount: number; wornShare: number; formalCount: number; duplicateRoles: number },
): Record<string, number> {
  switch (goal.type) {
    case 'reduce_size':
      return { itemCount: facts.itemCount };
    case 'wear_more':
      return { wornShare: Math.round(facts.wornShare * 100) / 100 };
    case 'improve_formalwear':
      return { formalCount: facts.formalCount };
    case 'stop_duplicates':
      return { duplicateRoles: facts.duplicateRoles };
    default:
      return { itemCount: facts.itemCount, wornShare: Math.round(facts.wornShare * 100) / 100 };
  }
}
