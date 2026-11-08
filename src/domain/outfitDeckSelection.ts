/** Exact feasibility and rotation selection for at most three evaluated looks.
 * No garment admission, inference, or wall-clock deadline belongs in this phase.
 * A forward compatibility graph makes every accepted triple a triangle. The
 * quality-sorted bit order finds the best third look without enumerating all
 * triples. Storage is O(n²); feasibility/quality/rotation passes are O(n² log n)
 * for the fixed four garment slots. Only the final editorial utility is sampled.
 */
export const DECK_SELECTION_VERSION = 'outfit-deck-graph-v1';

export interface DeckVertex<T> {
  look: T;
  quality: number;
  priority: readonly number[];
  /** Canonical physical equivalence class per slot; -1 means absent/locked. */
  products: readonly number[];
}
export interface DeckSelectionDiagnostics {
  algorithm: typeof DECK_SELECTION_VERSION;
  requested: number;
  selected: number;
  status: 'complete' | 'evaluated_inventory_limited' | 'no_valid_looks';
  reusePolicy: 'unique' | 'scarce_slots' | 'necessary_reuse';
  repeatedSlots: number[];
  repeatedProducts: number;
  qualityFloor: number;
  pairChecks: number;
  evaluatedLooks: number;
  rejectedPairs: { samePhysicalLook: number; semantic: number; crossSlotIdentity: number };
  utilityEvaluations: number;
  utilityBounded: boolean;
  feasibilityComplete: true;
}
const QUALITY_BAND = 0.035;
const UTILITY_LIMIT = 4096;
export function compareDeckPriority(a: readonly number[], b: readonly number[]): number {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const difference = (a[i] || 0) - (b[i] || 0);
    if (difference) return difference;
  }
  return 0;
}

export function createDeckSelector<T>(
  input: readonly DeckVertex<T>[],
  options: {
    target: number;
    scarceSlots: readonly number[];
    /** Semantic constraints are independent of physical scarcity. */
    compatible: (a: T, b: T) => boolean;
    utility: (looks: T[], ranks: ReadonlyMap<string, number>) => number;
  }
) {
  if (input.length > 512) throw new Error('deck_candidate_bound_exceeded');
  const vertices = [...input].sort((a, b) => b.quality - a.quality);
  const n = vertices.length,
    target = Math.max(1, Math.min(3, Math.trunc(options.target)));
  const bits = vertices.map((_, i) => 1n << BigInt(i));
  const indexByBit = new Map(bits.map((bit, i) => [bit, i]));
  const first = (set: bigint) => indexByBit.get(set & -set)!;
  const prefix = [0n];
  for (const bit of bits) prefix.push(prefix.at(-1)! | bit);
  const graphs = Array.from({ length: 3 }, () => vertices.map(() => 0n));
  const shared = new Uint8Array(n * n);
  const membership = Array.from({ length: 4 }, () => new Map<number, bigint>());
  vertices.forEach((v, i) =>
    v.products.forEach((product, slot) => {
      if (product >= 0)
        membership[slot].set(product, (membership[slot].get(product) || 0n) | bits[i]);
    })
  );
  const scarceMask = options.scarceSlots.reduce((mask, slot) => mask | (1 << slot), 0);
  let pairChecks = 0;
  const rejectedPairs = { samePhysicalLook: 0, semantic: 0, crossSlotIdentity: 0 };
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++) {
      pairChecks++;
      if (vertices[a].products.every((p, slot) => p === vertices[b].products[slot])) {
        rejectedPairs.samePhysicalLook++;
        continue;
      }
      if (!options.compatible(vertices[a].look, vertices[b].look)) {
        rejectedPairs.semantic++;
        continue;
      }
      let collisions = 0,
        crossSlot = false;
      for (let s = 0; s < 4; s++) {
        const product = vertices[a].products[s];
        if (product == null || product < 0) continue;
        for (let t = 0; t < 4; t++)
          if (product === vertices[b].products[t]) {
            if (s !== t) crossSlot = true;
            else collisions |= 1 << s;
          }
      }
      if (crossSlot) {
        rejectedPairs.crossSlotIdentity++;
        continue;
      }
      shared[a * n + b] = collisions;
      if (!collisions) graphs[0][a] |= bits[b];
      if (!(collisions & ~scarceMask)) graphs[1][a] |= bits[b];
      graphs[2][a] |= bits[b];
    }
  const countBits = (mask: number) => {
    let count = 0;
    while (mask) {
      mask &= mask - 1;
      count++;
    }
    return count;
  };
  // For a fixed pair, partition possible thirds by additional product reuse.
  // A product shared three times contributes two repeats, not three pair edges.
  const buckets = (a: number, b: number, common: bigint, policy: number) => {
    if (policy === 0) return [common];
    let result = [common];
    for (let slot = 0; slot < 4; slot++) {
      const hit =
        (membership[slot].get(vertices[a].products[slot]) || 0n) |
        (membership[slot].get(vertices[b].products[slot]) || 0n);
      const next = Array<bigint>(result.length + 1).fill(0n);
      for (let k = 0; k < result.length; k++) {
        next[k] |= result[k] & ~hit;
        next[k + 1] |= result[k] & hit;
      }
      result = next;
    }
    return result;
  };
  const primary = (indices: number[]) => {
    const quality = indices.map((i) => vertices[i].quality);
    return (
      (0.65 * quality.reduce((sum, q) => sum + q, 0)) / quality.length + 0.35 * Math.min(...quality)
    );
  };
  // A visit sees one bitset per (pair, repetition count), never materializes
  // the potentially millions of complete triples.
  const visit = (
    size: number,
    policy: number,
    fn: (a: number, b: number, choices: bigint, repeats: number) => void
  ) => {
    const graph = graphs[policy];
    for (let a = 0; a < n; a++) {
      if (size === 1) {
        fn(a, a, bits[a], 0);
        continue;
      }
      let neighbors = graph[a];
      while (neighbors) {
        const b = first(neighbors);
        neighbors &= neighbors - 1n;
        const base = countBits(shared[a * n + b]);
        if (size === 2) {
          fn(a, b, bits[b], base);
          continue;
        }
        const common = graph[a] & graph[b];
        if (!common) continue;
        const partition = buckets(a, b, common, policy);
        for (let k = 0; k < partition.length; k++)
          if (partition[k]) fn(a, b, partition[k], base + k);
      }
    }
  };
  let size = target,
    policy = 0,
    repeats = Infinity,
    bestQuality = -Infinity;
  // Cardinality first, zero reuse first; only an exhaustive failed pass can
  // authorize repetition. Each stage is exact over the evaluated look set.
  search: for (; size >= 1; size--) {
    for (policy = 0; policy < (size === 1 ? 1 : 3); policy++) {
      visit(size, policy, (a, b, choices, repeated) => {
        const c = first(choices);
        const q = primary(size === 1 ? [a] : size === 2 ? [a, b] : [a, b, c]);
        if (repeated < repeats) {
          repeats = repeated;
          bestQuality = q;
        } else if (repeated === repeats) bestQuality = Math.max(bestQuality, q);
      });
      if (bestQuality > -Infinity) break search;
    }
  }
  const qualityFloor = bestQuality - QUALITY_BAND;
  const qualityPrefix = (minimum: number) => {
    let lo = 0,
      hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (vertices[mid].quality + 1e-12 >= minimum) lo = mid + 1;
      else hi = mid;
    }
    return prefix[lo];
  };
  const priorityOrder = vertices
    .map((_, i) => i)
    .sort((a, b) => compareDeckPriority(vertices[b].priority, vertices[a].priority) || a - b);
  // Prefix masks permit a logarithmic best-priority query against any set
  // of feasible thirds, including the quality/repetition restrictions.
  const priorityPrefix = [0n];
  for (const i of priorityOrder) priorityPrefix.push(priorityPrefix.at(-1)! | bits[i]);
  const bestPriorityIndex = (choices: bigint) => {
    let lo = 1,
      hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (choices & priorityPrefix[mid]) hi = mid;
      else lo = mid + 1;
    }
    return priorityOrder[lo - 1];
  };
  const select = (ranks: ReadonlyMap<string, number> = new Map()) => {
    let best: number[] = [],
      bestOrder: number[] = [],
      bestUtility = -Infinity;
    let utilityBounded = false;
    let finalists: number[][] = [];
    if (size >= 1)
      visit(size, policy, (a, b, set, repeated) => {
        if (repeated !== repeats) return;
        const base = size === 1 ? [] : size === 2 ? [a] : [a, b];
        // For sorted a<b<c, c is the minimum-quality member. Solve the
        // primary-quality inequality directly rather than scanning each third.
        const minimum =
          size === 1
            ? qualityFloor
            : (qualityFloor -
                (0.65 * base.reduce((sum, i) => sum + vertices[i].quality, 0)) / size) /
              (0.65 / size + 0.35);
        const choices = set & qualityPrefix(minimum);
        if (!choices) return;
        const third = bestPriorityIndex(choices),
          indices = [...base, third];
        const order: number[] = [];
        for (const i of indices)
          vertices[i].priority.forEach((v, k) => (order[k] = (order[k] || 0) + v));
        const comparison = finalists.length ? compareDeckPriority(order, bestOrder) : 1;
        if (comparison > 0) {
          bestOrder = order;
          finalists = [indices];
          utilityBounded = false;
        } else if (comparison === 0) {
          if (finalists.length < UTILITY_LIMIT) finalists.push(indices);
          else utilityBounded = true;
        }
      });
    // Finish the exact rotation pass before spending any editorial work.
    // Even adversarial priority order cannot exceed this fixed utility budget.
    for (const indices of finalists) {
      const utility = options.utility(
        indices.map((i) => vertices[i].look),
        ranks
      );
      if (!best.length || utility > bestUtility) {
        best = indices;
        bestUtility = utility;
      }
    }
    const utilityEvaluations = finalists.length;
    const looks = best.map((i) => vertices[i].look);
    const repeatedSlots = [0, 1, 2, 3].filter((slot) => {
      const products = best.map((i) => vertices[i].products[slot]).filter((p) => p >= 0);
      return new Set(products).size < products.length;
    });
    const diagnostics: DeckSelectionDiagnostics = {
      algorithm: DECK_SELECTION_VERSION,
      requested: target,
      selected: looks.length,
      status:
        looks.length === target
          ? 'complete'
          : looks.length
            ? 'evaluated_inventory_limited'
            : 'no_valid_looks',
      reusePolicy:
        !looks.length || policy === 0
          ? 'unique'
          : policy === 1
            ? 'scarce_slots'
            : 'necessary_reuse',
      repeatedSlots,
      repeatedProducts: Number.isFinite(repeats) ? repeats : 0,
      qualityFloor: Number.isFinite(qualityFloor) ? qualityFloor : 0,
      pairChecks,
      evaluatedLooks: n,
      rejectedPairs,
      utilityEvaluations,
      utilityBounded,
      feasibilityComplete: true,
    };
    return { looks, diagnostics };
  };
  return { select };
}
