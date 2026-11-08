import {
  createDeckSelector,
  compareDeckPriority,
  type DeckVertex,
} from '../../src/domain/outfitDeckSelection';
type Look = { id: string };
const vertex = (
  id: string,
  products: number[],
  quality = 0.9,
  priority: number[] = []
): DeckVertex<Look> => ({ look: { id }, products, quality, priority });
const choose = (
  rows: DeckVertex<Look>[],
  scarceSlots: number[] = [],
  compatible: (a: Look, b: Look) => boolean = () => true
) =>
  createDeckSelector(rows, {
    target: 3,
    scarceSlots,
    compatible,
    utility: () => 0,
  }).select();

test('finds the only unique triple across 323 evaluated looks, beyond any old reserve', () => {
  const decoys = Array.from({ length: 320 }, (_, i) =>
    vertex(`decoy${i}`, [100, 1000 + i, 102, 103])
  );
  const required = [
    vertex('a', [100, 10, 200, 300]),
    vertex('b', [201, 11, 102, 301]),
    vertex('c', [202, 12, 302, 103]),
  ];
  const result = choose([...decoys, ...required]);
  expect(result.looks.map((l) => l.id).sort()).toEqual(['a', 'b', 'c']);
  expect(result.diagnostics).toMatchObject({
    selected: 3,
    repeatedProducts: 0,
    feasibilityComplete: true,
    pairChecks: 52003,
  });
});

test('two shoes can complete three distinct looks with exactly one necessary repeat', () => {
  const rows = Array.from({ length: 6 }, (_, i) =>
    vertex(String(i), [i, 100 + i, 200 + i, 300 + (i % 2)])
  );
  const result = choose(rows, [3]);
  expect(result.looks).toHaveLength(3);
  expect(result.diagnostics).toMatchObject({
    reusePolicy: 'scarce_slots',
    repeatedSlots: [3],
    repeatedProducts: 1,
  });
});

test('conditional scarcity is proved over complete looks, not raw slot counts', () => {
  const rows = Array.from({ length: 5 }, (_, i) =>
    vertex(String(i), [1, 100 + i, 200 + i, 300 + i])
  );
  const result = choose(rows);
  expect(result.looks).toHaveLength(3);
  expect(result.diagnostics).toMatchObject({
    reusePolicy: 'necessary_reuse',
    repeatedSlots: [0],
    repeatedProducts: 2,
  });
});

test('four-shoe history includes the fresh shoe and uses three unique physical shoes', () => {
  const rows = Array.from({ length: 16 }, (_, i) =>
    vertex(String(i), [i, 100 + i, 200 + i, 300 + (i % 4)], 0.9, [i % 4 === 3 ? 0 : -1, 1])
  );
  const result = choose(rows, [3]);
  const picked = rows.filter((v) => result.looks.includes(v.look));
  expect(new Set(picked.map((v) => v.products[3])).size).toBe(3);
  expect(picked.some((v) => v.products[3] === 303)).toBe(true);
  expect(result.diagnostics.repeatedProducts).toBe(0);
});

test('aliases cannot manufacture three cards and scarcity never waives semantic separation', () => {
  const aliases = Array.from({ length: 4 }, (_, i) => vertex(String(i), [1, 2, 3, 4]));
  expect(choose(aliases, [0, 1, 2, 3]).looks).toHaveLength(1);
  const rows = Array.from({ length: 4 }, (_, i) =>
    vertex(String(i), [i, 100 + i, 200 + i, 300 + i])
  );
  expect(choose(rows, [], () => false).diagnostics).toMatchObject({
    selected: 1,
    status: 'evaluated_inventory_limited',
    feasibilityComplete: true,
  });
});

test('utility work is bounded without making feasibility a timeout lottery', () => {
  const rows = Array.from({ length: 128 }, (_, i) =>
    vertex(String(i), [i, 1000 + i, 2000 + i, 3000 + i])
  );
  const utility = jest.fn(() => 0);
  const result = createDeckSelector(rows, {
    target: 3,
    scarceSlots: [],
    compatible: () => true,
    utility,
  }).select();
  expect(result.looks).toHaveLength(3);
  expect(result.diagnostics).toMatchObject({
    pairChecks: 8128,
    utilityBounded: true,
    feasibilityComplete: true,
  });
  expect(utility.mock.calls.length).toBeLessThanOrEqual(4096);
});

// Exhaustive small-instance oracle checks the algorithmic contract rather
// than implementation-specific choices or a machine-dependent time threshold.
test('matches exhaustive feasibility, least reuse, quality band and rotation priority', () => {
  for (let seed = 0; seed < 24; seed++) {
    const rows = Array.from({ length: 8 }, (_, i) =>
      vertex(
        String(i),
        [0, 1, 2, 3].map((s) => s * 100 + ((i * (s + 1) + seed + s) % 5)),
        0.78 + ((i * 7 + seed) % 20) / 100,
        [-((i + seed) % 3), i % 2]
      )
    );
    const semantic = (a: Look, b: Look) => (Number(a.id) + Number(b.id) + seed) % 4 !== 0;
    const result = choose(rows, [3], semantic);
    let candidates: { indices: number[]; reuse: number; quality: number; priority: number[] }[] =
      [];
    outer: for (let size = 3; size >= 1; size--)
      for (let policy = 0; policy < 3; policy++) {
        const visit = (indices: number[], start: number) => {
          if (indices.length < size) {
            for (let i = start; i < rows.length; i++) visit([...indices, i], i + 1);
            return;
          }
          for (let a = 0; a < indices.length; a++)
            for (let b = a + 1; b < indices.length; b++) {
              const x = rows[indices[a]],
                y = rows[indices[b]];
              if (!semantic(x.look, y.look) || x.products.every((p, s) => p === y.products[s]))
                return;
              if (
                x.products.some(
                  (p, s) => p === y.products[s] && (policy === 0 || (policy === 1 && s !== 3))
                )
              )
                return;
            }
          const reuse = [0, 1, 2, 3].reduce(
            (n, s) => n + size - new Set(indices.map((i) => rows[i].products[s])).size,
            0
          );
          const qs = indices.map((i) => rows[i].quality);
          candidates.push({
            indices,
            reuse,
            quality: (0.65 * qs.reduce((n, q) => n + q, 0)) / size + 0.35 * Math.min(...qs),
            priority: [0, 1].map((k) => indices.reduce((n, i) => n + rows[i].priority[k], 0)),
          });
        };
        visit([], 0);
        if (candidates.length) break outer;
      }
    const repeats = Math.min(...candidates.map((c) => c.reuse));
    candidates = candidates.filter((c) => c.reuse === repeats);
    const floor = Math.max(...candidates.map((c) => c.quality)) - 0.035;
    candidates = candidates.filter((c) => c.quality >= floor - 1e-12);
    const best = candidates.sort((a, b) => compareDeckPriority(b.priority, a.priority))[0];
    const chosen = rows.filter((r) => result.looks.includes(r.look));
    expect(chosen).toHaveLength(best.indices.length);
    expect(result.diagnostics.repeatedProducts).toBe(repeats);
    expect(
      compareDeckPriority(
        [0, 1].map((k) => chosen.reduce((n, r) => n + r.priority[k], 0)),
        best.priority
      )
    ).toBe(0);
    const qs = chosen.map((r) => r.quality);
    expect(
      (0.65 * qs.reduce((n, q) => n + q, 0)) / qs.length + 0.35 * Math.min(...qs)
    ).toBeGreaterThanOrEqual(floor - 1e-12);
  }
});
