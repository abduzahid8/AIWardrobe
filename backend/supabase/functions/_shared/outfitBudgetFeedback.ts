/** Explain rejected spending using catalog prices; never repair model selections. */
export function outfitBudgetFeedback(
  outfits: any[], items: Map<string, any>, budget?: { amount: number; currency?: string } | null,
  convert?: (amount: number, from: string, to: string) => number,
  eligible?: (item: any) => boolean,
): string {
  if (!budget || !Number.isFinite(budget.amount) || budget.amount <= 0) return '';
  const currency = String(budget.currency || '').toUpperCase();
  const price = (item: any): number | null => {
    if (!item) return null;
    if (item.isWardrobe === true && !item.isShopItem) return 0;
    if (!item.isShopItem && item.isWardrobe !== false && !String(item.id || '').startsWith('shop_')) return 0;
    if (!Number.isFinite(item.price) || item.price <= 0) return null;
    const source = String(item.currency || '').toUpperCase();
    const cost = source === currency ? item.price : convert?.(item.price, source, currency);
    return typeof cost === 'number' && Number.isFinite(cost) ? cost : null;
  };
  const totals = outfits.map(outfit => {
    const costs = (outfit.items || []).map((item: any) => price(items.get(String(typeof item === 'string' ? item : item?.id))));
    return costs.length && costs.every((cost: number | null) => cost !== null)
      ? costs.reduce((sum: number, cost: number) => sum + cost, 0) : null;
  });
  if (!totals.some(total => total !== null && total > budget.amount)) return '';
  const counts = new Map<string, number>();
  const candidates = [...items.values()].filter(item => price(item) !== null && (!eligible || eligible(item)))
    .sort((a, b) => price(a)! - price(b)!).filter(item => {
      const slot = String(item.macroCategory || item.category || '');
      const count = counts.get(slot) || 0;
      if (count >= 3) return false;
      counts.set(slot, count + 1); return true;
    }).map(item => ({ id: String(item.id), slot: item.macroCategory || item.category, cost: price(item) }));
  const slots = new Set<string>((outfits[0]?.items || []).map((item: any) =>
    String(items.get(String(typeof item === 'string' ? item : item?.id))?.macroCategory || '')));
  const ordinarySlots = [...slots].every(slot => ['top', 'bottom', 'shoes', 'outerwear'].includes(slot))
    && ['top', 'bottom', 'shoes'].every(slot => slots.has(slot));
  const example = ordinarySlots ? [...slots].map(slot => candidates.find(item => item.slot === slot)) : [];
  const completeExample = example.length >= 3 && example.every(Boolean);
  const exampleTotal = completeExample ? example.reduce((sum, item) => sum + Number(item!.cost), 0) : Infinity;
  const guidance = exampleTotal <= budget.amount
    ? ` A complete candidate combination within the spending cap is ${JSON.stringify(example)}; total ${exampleTotal} ${currency}. You may select it only if it also meets every style and occasion rule, or choose another valid combination.` : '';
  return `\nBUDGET CORRECTION: Hard limit ${budget.amount} ${currency} PER outfit. Previous catalog totals: ${JSON.stringify(totals)}. Add the prices before choosing IDs; every outfit must include all required slots and stay within the limit. Owned pieces cost zero. Lower-cost candidates (costs expressed in the budget currency): ${JSON.stringify(candidates)}.${guidance} Keep all category, occasion and style requirements; do not omit a required slot to reduce spending.`;
}
