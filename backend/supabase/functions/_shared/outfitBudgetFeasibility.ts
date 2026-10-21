/**
 * Lower bound only: ignores style and compatibility, so it cannot prove a look
 * exists. Callers must exclude alternative compositions such as dresses.
 * Unknown prices or missing slots leave feasibility undecided.
 */
export function minimumSlotSpend<T extends { macroCategory?: string }>(
  items: T[], slots: string[], costOf: (item: T) => number | null,
): number | null {
  if (!slots.length) return null;
  const minimum = new Map<string, number>();
  for (const item of items) {
    const slot = item.macroCategory || '';
    if (!slots.includes(slot)) continue;
    const cost = costOf(item);
    if (cost === null || !Number.isFinite(cost) || cost < 0) return null;
    minimum.set(slot, Math.min(minimum.get(slot) ?? Infinity, cost));
  }
  if (slots.some(slot => !minimum.has(slot))) return null;
  const total = [...new Set(slots)].reduce((sum, slot) => sum + minimum.get(slot)!, 0);
  return Number.isFinite(total) ? total : null;
}
