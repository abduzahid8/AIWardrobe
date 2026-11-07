/** Deterministic local fixtures. No account identities, network or image downloads. */
export function makeWardrobe(count, prefix) {
  const garments = [
    ['upper_body', 'White cotton oxford shirt', 'top'],
    ['lower_body', 'Navy cotton chino trousers', 'bottom'],
    ['shoes', 'Brown leather penny loafers', 'shoes'],
    ['upper_body', 'Beige merino cardigan', 'outerwear'],
    ['upper_body', 'Grey cotton T-shirt', 'top'],
    ['lower_body', 'Black straight jeans', 'bottom'],
    ['shoes', 'White leather sneakers', 'shoes'],
    ['upper_body', 'Navy lightweight blazer', 'outerwear'],
  ];
  return Array.from({ length: count }, (_, i) => {
    const [garmentType, name, macroCategory] = garments[i % garments.length];
    return {
      id: `synthetic-${prefix}-${i}`,
      name: Math.floor(i / garments.length)
        ? `${['Navy', 'Grey', 'Olive', 'Cream', 'Black', 'Blue', 'Burgundy', 'Tan'][Math.floor(i / garments.length) % 8]} ${name.replace(/^(White|Navy|Brown|Beige|Grey|Black) /, '')} edition ${Math.floor(i / garments.length)}`
        : name,
      garmentType,
      macroCategory,
      brand: 'Synthetic',
      price: 0,
      currency: 'USD',
      imageUrl: 'assets/images/basic_white_tshirt.png',
      isWardrobe: true,
    };
  });
}
export function makePersonas() {
  return [
    {
      id: 'A',
      label: 'Student',
      count: 32,
      style: 'casual',
      budget: 50,
      repeatTolerance: 'medium',
      fitSensitive: false,
    },
    {
      id: 'B',
      label: 'Office',
      count: 48,
      style: 'business_casual',
      budget: 120,
      repeatTolerance: 'medium',
      fitSensitive: true,
    },
    {
      id: 'C',
      label: 'Minimal',
      count: 20,
      style: 'minimalist',
      budget: 25,
      repeatTolerance: 'high',
      fitSensitive: false,
    },
    {
      id: 'D',
      label: 'Fashion-forward',
      count: 72,
      style: 'casual',
      budget: 180,
      repeatTolerance: 'low',
      fitSensitive: true,
    },
    {
      id: 'E',
      label: 'Incomplete',
      count: 3,
      style: 'casual',
      budget: 30,
      repeatTolerance: 'unknown',
      fitSensitive: false,
    },
  ].map(({ count, ...persona }) => ({
    ...persona,
    wardrobe: makeWardrobe(count, persona.id),
    fitEvidence: [],
    styleEvidence: [],
    testIntent:
      persona.id === 'D' ? 'recreate_reference' : persona.id === 'C' ? 'no_buy' : 'plan_outfit',
  }));
}
if (process.argv[1]?.endsWith('personas.mjs'))
  console.log(
    JSON.stringify(
      {
        schemaVersion: 1,
        synthetic: true,
        personas: makePersonas(),
        stress: {
          wardrobe: makeWardrobe(220, 'stress'),
          referenceImages: Array.from({ length: 24 }, (_, i) => ({
            id: `reference-${i}`,
            localAsset: 'assets/images/basic_outfit_1.png',
          })),
        },
      },
      null,
      2
    )
  );
