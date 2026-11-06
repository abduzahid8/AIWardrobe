/** @jest-environment node */
// Focused CPU-bound regression for the deck selector and shoe reassignment.
// A full catalog pool must resolve through the deterministic greedy path
// (never the 250k-combination exhaustive walk) with identical results run
// over run; shoe reassignment must stay bounded for large shoe pools too.
const fs = require('fs'), path = require('path'), ts = require('typescript');

const domain = path.resolve(__dirname, '../../src/domain');
function loadDomain(files, returns) {
  const combined = files.map((file) => fs.readFileSync(path.join(domain, file), 'utf8')
    .replace(/^import .*$/gm, '')
    .replace(/\bexport\s+/g, '')).join('\n');
  const compiled = ts.transpileModule(combined, {
    compilerOptions: { module: ts.ModuleKind.None, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return new Function(`${compiled}\nreturn {${returns}};`)();
}

const deck = loadDomain(
  ['outfitDiversity.ts', 'outfitRotation.ts'],
  'assembleDiverseDeck,assignRotatingShoes,productIdentityKeys,diversitySlotOf',
);

const SLOTS = ['outerwear', 'top', 'bottom', 'shoes'];
function candidate(i) {
  const items = SLOTS.map((slot) => ({
    id: `${slot}-${i}`,
    name: `${slot} product ${i}`,
    macroCategory: slot,
    color: 'navy',
    imageUrl: `https://example.invalid/${slot}-${i}.jpg`,
  }));
  return { id: `look${i}`, items };
}
const itemsOf = (c) => c.items;
const keysOf = (item) => deck.productIdentityKeys(item);

function deckOptions(pool, debug) {
  return {
    candidates: pool,
    itemsOf,
    qualityOf: (choice) => 0.5 + (Number(String(choice.id).replace('look', '')) % 10) / 20,
    keysOf,
    usedKeys: new Set(),
    target: 3,
    debug,
  };
}

test('an 80-candidate pool resolves through the bounded greedy path deterministically', () => {
  const pool = Array.from({ length: 80 }, (_, i) => candidate(i));
  // C(80,3) = 82,160 > MAX_COMBINATIONS: exhaustive walk is off the table.
  const seen = [];
  const first = deck.assembleDiverseDeck(deckOptions(pool, (m, d) => seen.push([m, d])));
  expect(first.deck).toHaveLength(3);
  expect(first.relaxReason).not.toBe('repeated_products_in_strict_slot');
  const combos = seen.filter(([m]) => m === 'SELECT_COMBINATION');
  expect(combos.length).toBeGreaterThan(0);
  expect(combos.every(([, d]) => d.exhaustive === false)).toBe(true);
  // Four-slot correctness: every look covers every slot, no product repeats.
  for (const look of first.deck) {
    expect(look.items.map((i) => i.macroCategory).sort()).toEqual([...SLOTS].sort());
  }
  const keys = first.deck.flatMap((look) => look.items.flatMap(keysOf));
  expect(new Set(keys).size).toBe(keys.length);
  // Deterministic: same pool, same deck, same order.
  const second = deck.assembleDiverseDeck(deckOptions(pool, undefined));
  expect(second.deck.map((l) => l.id)).toEqual(first.deck.map((l) => l.id));
});

test('shoe reassignment stays bounded and deterministic for a 30-shoe pool', () => {
  const looks = [candidate(1), candidate(2), candidate(3)].map((look, i) => ({
    ...look,
    items: look.items.map((item) => ({ ...item, id: `${item.macroCategory}-base-${i}` })),
  }));
  const shoes = Array.from({ length: 30 }, (_, i) => ({
    id: `shoes-pool-${i}`,
    name: `shoe ${i}`,
    macroCategory: 'shoes',
    color: 'black',
    imageUrl: `https://example.invalid/shoe-${i}.jpg`,
  }));
  // 30^3 = 27,000 assignment combinations > MAX_SHOE_ASSIGNMENT_COMBINATIONS.
  const accepts = () => true;
  const priority = () => [0];
  const quality = (parts) => parts.length;
  const run = () => deck.assignRotatingShoes(looks, shoes, accepts, priority, quality, undefined);
  const first = run();
  expect(first.deck).toHaveLength(3);
  const chosenShoes = first.deck.map((look) => look.items.find((i) => i.macroCategory === 'shoes').id);
  expect(new Set(chosenShoes).size).toBe(3);
  const second = run();
  expect(second.deck.map((l) => l.items.map((i) => i.id).join('|')))
    .toEqual(first.deck.map((l) => l.items.map((i) => i.id).join('|')));
});

test('the Edge operation budget caches identity and bounds ranking work', () => {
  const pool = Array.from({length: 64}, (_, i) => candidate(i));
  let identityCalls = 0, priorities = 0, scores = 0;
  const run = () => deck.assembleDiverseDeck({
    ...deckOptions(pool), maxSearchSteps: 512,
    keysOf: item => {identityCalls++; return deck.productIdentityKeys(item);},
    qualityOf: () => {scores++; return 0.9;},
    priorityOfDeck: () => {priorities++; return [0];},
  });
  const selected = run();
  expect(selected.deck).toHaveLength(3);
  expect(scores).toBe(64);
  expect(priorities).toBeLessThanOrEqual(1024);
  expect(identityCalls).toBeLessThan(1200);
  expect(selected.productDiversity.illegalProductReuseCount).toBe(0);
  expect(run().deck.map(l => l.id)).toEqual(selected.deck.map(l => l.id));
});
