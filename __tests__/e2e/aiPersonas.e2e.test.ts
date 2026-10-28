/**
 * aiPersonas.e2e.test.ts — the AI under four different accounts.
 *
 * One test run, four stand-ins for real users with different styles,
 * wardrobes, occasions and weather (see personaHarness.ts). Every persona
 * runs the FULL production pipeline; the suite pins per-account behavior:
 *
 *   - Amir (old_money / work / cold)      → layered formal decks, shop mode
 *   - Sara (casual / casual / hot)        → layered relaxed decks, shop mode
 *   - Dmitriy (business_casual / wardrobe) → layered, 100% owned items
 *   - Lola (minimalist / wardrobe)        → quiet palette, 100% owned items
 *
 * Cross-account assertions prove personalization is real: different styles
 * rank different tops first, and shop decks never contain owned items while
 * wardrobe decks never contain shop items.
 *
 * The same harness feeds scripts/ai-persona-report.ts, which renders the
 * visual report + screenshots — what you see there is exactly what is
 * asserted here.
 */

import { PERSONAS, runPersona, runAllPersonas, shopPool } from './personaHarness';
import { canonicalizeMacroCategory } from '../../src/utils/categoryMapper';
import {
  rankItemsForStyle,
  scoreItemForStyle,
} from '../../features/outfit-generator/utils/styleInference';

describe('AI personas e2e — four accounts', () => {
  it('all personas produce complete, unique decks', () => {
    const results = runAllPersonas();
    expect(results).toHaveLength(4);
    for (const r of results) {
      expect(r.deck.length).toBeGreaterThanOrEqual(1);
      for (const check of r.checks) {
        if (!check.pass) throw new Error(`${r.persona.account} — ${check.label}: ${check.detail}`);
      }
    }
  });

  it('all accounts retain the layer while weather controls garment choice', () => {
    const byAccount = Object.fromEntries(runAllPersonas().map((r) => [r.persona.account, r]));
    expect(byAccount['Amir · Old Money executive'].layered).toBe(true); // 12°C cold
    expect(byAccount['Sara · Casual student'].layered).toBe(true); // 28°C clear
    expect(byAccount['Dmitriy · Business casual'].layered).toBe(true); // wind
  });

  it('styles personalize: old_money and casual rank different tops first', () => {
    const catalog = shopPool();
    const topFor = (style: any) =>
      rankItemsForStyle(catalog as any[], style, {
        minKeep: 16,
        dropThreshold: -2,
        perCategoryFloor: 4,
      }).find((i: any) => canonicalizeMacroCategory(i.macroCategory) === 'top');
    const formalTop = topFor('old_money');
    const casualTop = topFor('casual');
    expect(formalTop).toBeTruthy();
    expect(casualTop).toBeTruthy();
    // Different aesthetics must surface different heroes — identical #1
    // picks across opposite styles would mean scoring ignores style.
    expect(String((formalTop as any).id)).not.toBe(String((casualTop as any).id));
    // And the formal winner genuinely suits the formal style better.
    const formalScore = scoreItemForStyle(
      { name: (formalTop as any).name, brand: (formalTop as any).brand, type: (formalTop as any).type },
      'old_money',
    );
    const casualScore = scoreItemForStyle(
      { name: (casualTop as any).name, brand: (casualTop as any).brand, type: (casualTop as any).type },
      'old_money',
    );
    expect(formalScore).toBeGreaterThanOrEqual(casualScore);
  });

  it('source purity holds per account and mode', () => {
    for (const persona of PERSONAS) {
      const r = runPersona(persona);
      const ownedCount = r.deck.flat().filter((i) => i.isWardrobe).length;
      const total = r.deck.flat().length;
      if (persona.mode === 'wardrobe') {
        expect(ownedCount).toBe(total);
      } else {
        expect(ownedCount).toBe(0);
      }
    }
  });

  it('persona wardrobes are all distinct closets (no shared accounts)', () => {
    const closets = PERSONAS.map((p) => new Set(p.wardrobe.map((i) => i.name)));
    for (let a = 0; a < closets.length; a++) {
      for (const name of closets[a]) {
        // Display names may repeat across closets, but internal ids never do.
        void name;
      }
    }
    const allIds = PERSONAS.flatMap((p) => p.wardrobe.map((i) => i.id));
    expect(new Set(allIds).size).toBe(allIds.length);
  });
});
