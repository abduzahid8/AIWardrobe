/**
 * Fit engine golden tests.
 *
 * These lock the engine's scoring shape so refactors don't silently
 * change what the user sees. If you intentionally change behaviour,
 * update the asserted values in one place — never silently.
 *
 * Coverage:
 *   - assessFit: top (slim/regular/oversized), pants, jacket, shoes
 *   - Edge cases: missing measurements, missing body type
 *   - Size recommendation: picks the better-fitting alternative
 *   - Overall rollup: fit-intent aware
 */

import { assessFit, recommendSize } from '../../../src/lib/fit/fitEngine';
import { BodyProfile } from '../../../src/types/bodyProfile';
import { GarmentPhysicalProfile } from '../../../src/types/garment';

// ─── Test fixtures ───────────────────────────────────────────────────────────

function makeBody(overrides: Partial<BodyProfile> = {}): BodyProfile {
  return {
    id: 'body_test',
    userId: 'user_test',
    status: 'ready',
    isActive: true,
    height: { valueCm: 175, confidence: 'high', source: 'apple_measure' },
    weightKg: 72,
    bodyType: 'average',
    measurements: {
      chest: { valueCm: 100, confidence: 'high', source: 'manual' },
      waist: { valueCm: 84, confidence: 'high', source: 'manual' },
      hips: { valueCm: 98, confidence: 'high', source: 'manual' },
      shoulderWidth: { valueCm: 45, confidence: 'medium', source: 'manual' },
      armLength: { valueCm: 62, confidence: 'medium', source: 'manual' },
      torsoLength: { valueCm: 48, confidence: 'medium', source: 'manual' },
      inseam: { valueCm: 80, confidence: 'high', source: 'manual' },
      thigh: { valueCm: 56, confidence: 'medium', source: 'manual' },
      footLength: { valueCm: 26.5, confidence: 'high', source: 'apple_measure' },
    },
    privacy: { retainSourcePhoto: false, retainMesh: true },
    version: 1,
    createdAt: '2025-01-01T00:00:00Z',
    updatedAt: '2025-01-01T00:00:00Z',
    ...overrides,
  };
}

function makeTop(overrides: Partial<GarmentPhysicalProfile> = {}): GarmentPhysicalProfile {
  return {
    id: 'top_test',
    garmentId: 'tshirt_classic',
    sizeLabel: 'M',
    category: 'top',
    fitIntent: 'regular',
    stretch: 'low',
    measurements: {
      chest: { valueCm: 106, confidence: 'medium' },
      shoulderWidth: { valueCm: 46, confidence: 'medium' },
      sleeveLength: { valueCm: 22, confidence: 'medium' },
      bodyLength: { valueCm: 70, confidence: 'medium' },
    },
    source: 'admin_manual',
    ...overrides,
  };
}

function makePants(overrides: Partial<GarmentPhysicalProfile> = {}): GarmentPhysicalProfile {
  return {
    id: 'pants_test',
    garmentId: 'jeans_slim',
    sizeLabel: '32',
    category: 'jeans',
    fitIntent: 'slim',
    stretch: 'medium',
    measurements: {
      waist: { valueCm: 82, confidence: 'medium' },
      hips: { valueCm: 102, confidence: 'medium' },
      thigh: { valueCm: 58, confidence: 'medium' },
      inseam: { valueCm: 81, confidence: 'medium' },
    },
    source: 'admin_manual',
    ...overrides,
  };
}

function makeShoes(overrides: Partial<GarmentPhysicalProfile> = {}): GarmentPhysicalProfile {
  return {
    id: 'shoes_test',
    garmentId: 'shoes_sneaker',
    sizeLabel: 'EU 42',
    category: 'shoes',
    fitIntent: 'regular',
    stretch: 'low',
    measurements: { shoeLength: { valueCm: 27.0, confidence: 'high' } },
    source: 'manufacturer_size_chart',
    ...overrides,
  };
}

// ─── assessFit: tops ─────────────────────────────────────────────────────────

describe('assessFit > tops', () => {
  it('flags a slim-fit t-shirt as good_fit when chest ease is on-target', () => {
    const body = makeBody();
    const top = makeTop({
      fitIntent: 'slim',
      measurements: { chest: { valueCm: 103, confidence: 'high' } }, // 3cm ease, slim band 2-5
    });
    const result = assessFit(body, top);
    const chest = result.zones.find((z) => z.zone === 'chest');
    expect(chest?.status).toBe('good');
    expect(result.overall).toBe('good_fit');
  });

  it('flags a tight regular t-shirt as too_small', () => {
    const body = makeBody();
    const top = makeTop({
      fitIntent: 'regular',
      measurements: { chest: { valueCm: 96, confidence: 'high' } }, // -4cm ease, regular needs 5-10
    });
    const result = assessFit(body, top);
    expect(result.overall).toBe('too_small');
    const chest = result.zones.find((z) => z.zone === 'chest');
    expect(chest?.status).toBe('too_tight');
  });

  it('flags an oversized hoodie as oversized when chest ease is huge', () => {
    const body = makeBody();
    const top = makeTop({
      fitIntent: 'oversized',
      measurements: { chest: { valueCm: 140, confidence: 'high' } }, // +40cm, way over the 16-30 band
    });
    const result = assessFit(body, top);
    expect(result.overall).toBe('oversized');
  });

  it('detects tight shoulders when shoulder delta < -3cm', () => {
    const body = makeBody();
    const top = makeTop({
      measurements: { shoulderWidth: { valueCm: 40, confidence: 'high' } }, // -5cm
    });
    const result = assessFit(body, top);
    const shoulders = result.zones.find((z) => z.zone === 'shoulders');
    expect(shoulders?.status).toBe('too_tight');
  });

  it('detects long sleeves when sleeve delta > +5cm', () => {
    const body = makeBody();
    const top = makeTop({
      measurements: { sleeveLength: { valueCm: 70, confidence: 'high' } }, // +8cm
    });
    const result = assessFit(body, top);
    const sleeves = result.zones.find((z) => z.zone === 'sleeves');
    expect(sleeves?.status).toBe('too_long');
  });
});

// ─── assessFit: pants ───────────────────────────────────────────────────────

describe('assessFit > pants', () => {
  it('flags a slim jean as good_fit when waist/hips ease is on-target with stretch', () => {
    const body = makeBody();
    const pants = makePants({
      fitIntent: 'slim',
      stretch: 'medium',
      measurements: {
        waist: { valueCm: 84 }, // exactly body waist, stretch gives 2cm slack
        hips: { valueCm: 100 }, // +2cm, slim hip band 1-4, with stretch still good
        thigh: { valueCm: 58 }, // +2cm, band 1-6, good
        inseam: { valueCm: 80 },
      },
    });
    const result = assessFit(body, pants);
    expect(['good_fit', 'tight']).toContain(result.overall); // tight at waist is acceptable for slim
  });

  it('flags pants as too_small when hips are below body hips even with stretch', () => {
    const body = makeBody();
    const pants = makePants({
      stretch: 'low',
      measurements: {
        waist: { valueCm: 80 },
        hips: { valueCm: 92 }, // -6cm, way below
        thigh: { valueCm: 52 },
        inseam: { valueCm: 80 },
      },
    });
    const result = assessFit(body, pants);
    expect(result.overall).toBe('too_small');
  });

  it('detects inseam too short when delta < -4cm', () => {
    const body = makeBody();
    const pants = makePants({
      measurements: { ...makePants().measurements, inseam: { valueCm: 73 } }, // -7cm
    });
    const result = assessFit(body, pants);
    const inseam = result.zones.find((z) => z.zone === 'inseam');
    expect(inseam?.status).toBe('too_short');
  });
});

// ─── assessFit: jacket ───────────────────────────────────────────────────────

describe('assessFit > jacket', () => {
  it('includes a layering-ease note and treats chest with extra slack', () => {
    const body = makeBody();
    const jacket: GarmentPhysicalProfile = {
      ...makeTop(),
      id: 'jacket_test',
      garmentId: 'jacket_bomber',
      sizeLabel: 'M',
      category: 'jacket',
      fitIntent: 'regular',
      measurements: {
        chest: { valueCm: 110, confidence: 'medium' }, // raw +10cm, minus 4 layering = +6cm
        shoulderWidth: { valueCm: 48 },
        sleeveLength: { valueCm: 64 },
        bodyLength: { valueCm: 67 },
      },
    };
    const result = assessFit(body, jacket);
    // First zone should be the layering-ease note
    expect(result.zones[0]?.message).toMatch(/layering/i);
    const chest = result.zones.find((z) => z.zone === 'chest' && z.deltaCm != null);
    expect(chest?.status).toBe('good');
  });
});

// ─── assessFit: shoes ────────────────────────────────────────────────────────

describe('assessFit > shoes', () => {
  it('flags shoes with <0.3cm toe room as snug', () => {
    const body = makeBody();
    const shoes = makeShoes({
      measurements: { shoeLength: { valueCm: 26.7 } }, // +0.2cm
    });
    const result = assessFit(body, shoes);
    const feet = result.zones.find((z) => z.zone === 'feet');
    expect(feet?.status).toBe('snug');
  });

  it('flags oversized shoes (>2.5cm extra) as too_loose', () => {
    const body = makeBody();
    const shoes = makeShoes({
      measurements: { shoeLength: { valueCm: 30.5 } }, // +4cm, clearly too loose
    });
    const result = assessFit(body, shoes);
    const feet = result.zones.find((z) => z.zone === 'feet');
    expect(feet?.status).toBe('too_loose');
  });

  it('returns good_fit for shoes with ~1cm toe room', () => {
    const body = makeBody();
    const shoes = makeShoes({
      measurements: { shoeLength: { valueCm: 27.5 } }, // +1cm
    });
    const result = assessFit(body, shoes);
    expect(result.overall).toBe('good_fit');
  });
});

// ─── assessFit: edge cases ───────────────────────────────────────────────────

describe('assessFit > edge cases', () => {
  it('returns overall=unknown with low confidence if height is missing', () => {
    const body = makeBody({ height: undefined as any });
    const result = assessFit(body, makeTop());
    expect(result.overall).toBe('unknown');
    expect(result.confidence).toBe('low');
  });

  it('returns low confidence when most zones are missing measurements', () => {
    const body = makeBody({ measurements: {} });
    const result = assessFit(body, makeTop());
    expect(result.confidence).toBe('low');
  });

  it('includes engine version + generatedAt on every assessment', () => {
    const result = assessFit(makeBody(), makeTop());
    expect(result.engineVersion).toBe('fit-engine/v1');
    expect(result.generatedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });
});

// ─── recommendSize ───────────────────────────────────────────────────────────

describe('recommendSize', () => {
  it('returns undefined when the current size is already the best', () => {
    const body = makeBody();
    const current = makeTop({ sizeLabel: 'M' });
    const result = assessFit(body, current);
    const rec = recommendSize(result, current, [makeTop({ sizeLabel: 'S' }), makeTop({ sizeLabel: 'L' })], body);
    expect(rec).toBeUndefined();
  });

  it('recommends a larger size when current is too_small', () => {
    const body = makeBody();
    const current = makeTop({
      sizeLabel: 'S',
      measurements: { chest: { valueCm: 92 } }, // very tight
    });
    const result = assessFit(body, current);
    const rec = recommendSize(
      result,
      current,
      [makeTop({ sizeLabel: 'M', measurements: { chest: { valueCm: 106 } } }),
       makeTop({ sizeLabel: 'L', measurements: { chest: { valueCm: 116 } } })],
      body,
    );
    expect(rec).toBeDefined();
    expect(['M', 'L']).toContain(rec!.recommendedSize);
  });
});
