/**
 * aiOccasion.e2e.test.ts — taste enforcement for style × occasion.
 *
 * Production incident: "minimalist date night" decks shipped jacquard tees,
 * espadrilles, boat shoes and a quilted jacket, with raw feed captions
 * ("Front view of a ... - Image 0") on the cards. Every case below is a
 * real item from those logs. The judge (occasionRules.ts) must reject the
 * offenders and spare the correct picks.
 */

import {
  occasionFormality,
  rejectForStyleOccasion,
  rejectForClimate,
  rejectOutfitClimate,
  inferSeasonalWeather,
  rejectOutfitIncoherence,
  detectSlotMismatch,
  passesStyleJudge,
  humanizeProductName,
  matchesOccasionSilhouette,
} from '../../features/outfit-generator/utils/occasionRules';

describe('occasion formality', () => {
  it.each([
    ['work', 'formal'],
    ['date', 'formal'],
    ['formal', 'formal'],
    ['casual', 'casual'],
    ['sport', 'casual'],
    ['travel', 'casual'],
    [null, 'casual'],
    ['', 'casual'],
  ])('occasion %s → %s', (occasion, expected) => {
    expect(occasionFormality(occasion)).toBe(expected);
  });
});

describe('rejectForStyleOccasion — real log offenders (minimalist + date)', () => {
  const DIE = [
    [{ name: 'Leather boat shoe', type: 'boat shoe' }, 'boat shoes are not date-night footwear'],
    [{ name: 'Split suede leather espadrille with piece', type: 'espadrille' }, 'espadrilles are not date-night footwear'],
    [{ name: 'Zara RELAXED FIT STRIPED JACQUARD T-SHIRT - Blue', type: 't-shirt' }, 'jacquard has no place in minimalist date night'],
    [{ name: 'Front view of a black quilted jacket', type: 'jacket' }, 'quilted layer underdresses date night'],
  ] as const;
  it.each(DIE)('rejects %s', (item) => {
    expect(rejectForStyleOccasion(item, 'minimalist', 'date')).not.toBeNull();
  });

  const KEEP = [
    { name: '100% cool wool double-breasted suit blazer', type: 'blazer' },
    { name: 'Slim fit 100% cotton Oxford shirt', type: 'shirt' },
    { name: '100% linen regular fit pinstriped shirt', type: 'shirt' },
    { name: 'Cold wool suit trousers', type: 'trousers' },
    { name: 'Smart nappa loafers', type: 'loafers' },
    { name: '100% linen slim fit shirt', type: 'shirt' },
  ];
  it.each(KEEP.map((k) => [k] as const))('keeps %s', (item) => {
    // Pinstripes are tailoring, not prints — must survive the judge.
    expect(rejectForStyleOccasion(item, 'minimalist', 'date')).toBeNull();
  });

  it('relaxed rules stay relaxed for casual occasions', () => {
    expect(rejectForStyleOccasion({ name: 'White Sneakers' }, 'casual', 'casual')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Split suede leather espadrille' }, 'casual', 'casual')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Beige Cardigan' }, 'casual', 'casual')).toBeNull();
  });

  it('refined-casual DNA bans hoodies, athletic sneakers and basic tees', () => {
    expect(rejectForStyleOccasion({ name: 'Grey Hoodie' }, 'casual', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Chunky running sneakers' }, 'casual', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Basic cotton t-shirt', macroCategory: 'top' }, 'casual', 'casual')).not.toBeNull();
    // Knit polos and minimal white sneakers stay legal.
    expect(rejectForStyleOccasion({ name: 'Navy knit polo', macroCategory: 'top' }, 'casual', 'casual')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'White leather sneakers' }, 'casual', 'casual')).toBeNull();
  });

  it('vetoes plain tees as tops for formal occasions', () => {
    expect(rejectForStyleOccasion({ name: 'White heavyweight tee', macroCategory: 'top' }, 'casual', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Navy knit tee', macroCategory: 'top' }, 'old_money', 'work')).toBeNull();
  });

  it('classic styles ban boat shoes, basic tees and track pants even on weekends', () => {
    expect(rejectForStyleOccasion({ name: 'Leather boat shoe', macroCategory: 'shoes' }, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'White cotton t-shirt', macroCategory: 'top' }, 'classic', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Retro Velour Track Trousers', macroCategory: 'bottom' }, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Brown penny loafers', macroCategory: 'shoes' }, 'old_money', 'casual')).toBeNull();
  });

  it('minimalist bans ripped baggy jeans', () => {
    expect(rejectForStyleOccasion({ name: 'Ripped Baggy Fit Jeans', macroCategory: 'bottom' }, 'minimalist', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Black straight trousers', macroCategory: 'bottom' }, 'minimalist', 'casual')).toBeNull();
  });

  it('minimalist overlay fires in every occasion', () => {
    expect(rejectForStyleOccasion({ name: 'Neon graphic tee' }, 'minimalist', 'casual')).not.toBeNull();
    // A basic cotton tee is gym wear under Refined Casual DNA (knit polos
    // are the standard) — so the contrast case uses a loud-but-legal piece.
    expect(rejectForStyleOccasion({ name: 'Neon windbreaker' }, 'casual', 'casual')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Neon graphic tee' }, 'casual', 'casual')).not.toBeNull();
  });
});

describe('passesStyleJudge over whole outfits', () => {
  const goodDateNight = [
    { name: '100% cool wool double-breasted suit blazer' },
    { name: 'Slim fit 100% cotton Oxford shirt' },
    { name: 'Cold wool suit trousers' },
    { name: 'Smart nappa loafers' },
  ];
  const badDateNight = [
    { name: 'Wool-blend short-sleeve knit sweater' },
    { name: 'Zara RELAXED FIT STRIPED JACQUARD T-SHIRT' },
    { name: 'Stonewashed tapered-fit jeans' },
    { name: 'Split suede leather espadrille with piece' },
  ];
  it('passes the good date-night deck, fails the beachy one', () => {
    expect(passesStyleJudge(goodDateNight, 'minimalist', 'date')).toBe(true);
    expect(passesStyleJudge(badDateNight, 'minimalist', 'date')).toBe(false);
  });
  it('empty deck never passes', () => {
    expect(passesStyleJudge([], 'minimalist', 'date')).toBe(false);
  });
});

describe('rejectForStyleOccasion — sleepwear veto (old_money + work log offender)', () => {
  const PAJAMA = {
    name: 'Red and white striped pajama set. Long-sleeved shirt with buttons and front pockets, and long pants.',
    type: 'pajama set',
  };
  it.each([
    ['old_money', 'work'],
    ['classic', 'work'],
    ['business_casual', 'work'],
    ['casual', 'work'],
    ['casual', 'date'],
    ['casual', 'formal'],
  ])('rejects sleepwear for %s / %s', (style, occasion) => {
    expect(rejectForStyleOccasion(PAJAMA, style, occasion)).not.toBeNull();
  });

  it.each([['old_money'], ['classic'], ['business_casual']])(
    'rejects sleepwear for classic style %s even on casual occasions',
    (style) => {
      expect(rejectForStyleOccasion(PAJAMA, style, 'casual')).not.toBeNull();
      expect(rejectForStyleOccasion({ name: 'Cotton bathrobe' }, style, 'casual')).not.toBeNull();
    },
  );

  it('rejects sleepwear even on casual decks', () => {
    expect(rejectForStyleOccasion(PAJAMA, 'casual', 'casual')).not.toBeNull();
  });

  it('judge fails a work deck carrying the pajama top', () => {
    const deck = [
      { name: 'Cool wool suit blazer' },
      PAJAMA,
      { name: 'Wool blend cool wool darted pants' },
      { name: 'Leather loafers with penny strap' },
    ];
    expect(passesStyleJudge(deck, 'old_money', 'work')).toBe(false);
  });
});

describe('rejectForStyleOccasion — shoe taxonomy (loafers vs drivers)', () => {
  it.each([
    ['old_money', 'work'],
    ['classic', 'work'],
    ['business_casual', 'work'],
    ['casual', 'work'],
    ['casual', 'date'],
  ])('rejects driving moccasins for %s / %s', (style, occasion) => {
    expect(rejectForStyleOccasion({ name: 'Soft leather loafers with tie', type: 'driving moccasin' }, style, occasion)).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Khaki moccasin-style shoe', type: 'shoes' }, style, occasion)).not.toBeNull();
  });

  it.each([['old_money'], ['classic'], ['business_casual']])(
    'rejects drivers for classic style %s even on casual occasions',
    (style) => {
      expect(rejectForStyleOccasion({ name: 'Suede driving moccasin', type: 'shoes' }, style, 'casual')).not.toBeNull();
    },
  );

  it('loafer subtypes always survive formal occasions', () => {
    for (const name of ['Leather loafers with penny strap', 'Black bit loafer', 'Tassel loafer', 'Smart nappa loafers']) {
      expect(rejectForStyleOccasion({ name, type: 'loafers' }, 'old_money', 'work')).toBeNull();
    }
  });

  it('rejects driving loafers by name even without a driving type tag', () => {
    expect(rejectForStyleOccasion({ name: 'Soft leather loafers with tie', type: 'shoes', color: 'olive' }, 'old_money', 'casual')).not.toBeNull();
  });

  it('rejects olive casual shoes in classic decks', () => {
    expect(rejectForStyleOccasion({ name: 'Suede loafers', type: 'shoes', color: 'olive', macroCategory: 'shoes' }, 'classic', 'casual')).not.toBeNull();
  });

  it('rejects mules for formal occasions', () => {
    expect(rejectForStyleOccasion({ name: 'Leather mule slippers', type: 'mules' }, 'casual', 'date')).not.toBeNull();
  });
});

describe('rejectForStyleOccasion — loud prints on cloth (souvenir jacket log offender)', () => {
  const JACKET = {
    name: "Front view of mustard jacket with brown collar, snap buttons, floral, bird, and fire prints, and 'South Lake 75' text.",
    type: 'jacket',
    macroCategory: 'outerwear',
  };
  it.each([
    ['old_money', 'work'],
    ['classic', 'work'],
    ['business_casual', 'work'],
    ['casual', 'work'],
    ['old_money', 'casual'],
  ])('rejects the souvenir jacket for %s / %s', (style, occasion) => {
    expect(rejectForStyleOccasion(JACKET, style, occasion)).not.toBeNull();
  });

  it('rejects floral/graphic tops for classic styles even when casual', () => {
    expect(rejectForStyleOccasion({ name: 'Floral print shirt', macroCategory: 'top' }, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Graphic logo tee', macroCategory: 'top' }, 'classic', 'casual')).not.toBeNull();
  });

  it('rejects patch-embroidery souvenir layers and soccer polos for formal/classic', () => {
    const croppedSouvenir = {
      name: 'Zara Cropped-fit Patch And Embroidery Jacket - Orange',
      macroCategory: 'outerwear',
    };
    const soccerPolo = {
      name: 'Zara Retro Soccer Relaxed-fit Polo - Green / Ecru',
      macroCategory: 'top',
    };
    for (const [style, occasion] of [['old_money', 'work'], ['classic', 'work'], ['casual', 'work'], ['old_money', 'casual']] as const) {
      expect(rejectForStyleOccasion(croppedSouvenir, style, occasion)).not.toBeNull();
      expect(rejectForStyleOccasion(soccerPolo, style, occasion)).not.toBeNull();
    }
    // Rugby/tennis/golf stay legal — preppy staples, not fan wear.
    expect(rejectForStyleOccasion({ name: 'Rugby shirt', macroCategory: 'top' }, 'old_money', 'casual')).toBeNull();
  });

  it('spares tailored patterns and plain cloth', () => {
    for (const name of [
      'Navy blazer',
      'Glen check sport coat',
    ]) {
      expect(rejectForStyleOccasion({ name, macroCategory: 'outerwear' }, 'old_money', 'work')).toBeNull();
    }
    for (const name of [
      '100% linen regular fit pinstriped shirt',
      'Ecru shirt with vertical green stripes',
      'Blue and white striped oxford shirt',
      'White long-sleeve polo shirt',
    ]) {
      expect(rejectForStyleOccasion({ name, macroCategory: 'top' }, 'old_money', 'work')).toBeNull();
    }
  });

  it('never print-vetoes bottoms or shoes (slot-scoped)', () => {
    expect(rejectForStyleOccasion({ name: 'Floral swim shorts', macroCategory: 'bottom' }, 'old_money', 'casual')).toBeNull();
  });

  it('rejects printed drawstring pants as work bottoms', () => {
    const pants = {
      name: 'Zara Printed Barrel-fit Drawstring Pants - Khaki',
      macroCategory: 'bottom',
    };
    expect(rejectForStyleOccasion(pants, 'old_money', 'work')).not.toBeNull();
  });

  it('rejects plain drawstring bottoms for formal occasions, spares tailoring', () => {
    expect(
      rejectForStyleOccasion({ name: 'Linen drawstring pants', macroCategory: 'bottom' }, 'casual', 'work'),
    ).not.toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Wool blend cool wool darted pants', macroCategory: 'bottom' }, 'old_money', 'work'),
    ).toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Tapered fit linen pants', macroCategory: 'bottom' }, 'old_money', 'work'),
    ).toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Linen textured co-ord pants', macroCategory: 'bottom', description: 'Elasticated drawstring waist' }, 'old_money', 'formal'),
    ).not.toBeNull();
  });

  it('spares drawstring beach linen for classic styles on casual occasions', () => {
    expect(
      rejectForStyleOccasion({ name: 'Linen drawstring beach pants', macroCategory: 'bottom' }, 'old_money', 'casual'),
    ).toBeNull();
  });

  it('rejects elastic-waist trousers for formal occasions, spares side-elastic tailoring', () => {
    expect(
      rejectForStyleOccasion({ name: '100% Linen Elastic-Waist Pants', macroCategory: 'bottom' }, 'old_money', 'work'),
    ).not.toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Cotton twill pants with elasticated waistband', macroCategory: 'bottom' }, 'classic', 'work'),
    ).not.toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Trousers with elastic side panels', macroCategory: 'bottom' }, 'classic', 'work'),
    ).toBeNull();
  });
});

describe('rejectOutfitIncoherence — pairing rules from the transcripts', () => {
  const blazer = { name: 'Navy blazer', macroCategory: 'outerwear' };
  const sneakers = { name: 'White leather sneakers', macroCategory: 'shoes' };
  const knitPolo = { name: 'Cream knit polo', macroCategory: 'top' };
  const oxford = { name: 'White oxford shirt', macroCategory: 'top' };
  const chinos = { name: 'Beige chinos', macroCategory: 'bottom' };

  it('vetoes sneakers with tailoring, spares Smart-Low knit pairings', () => {
    expect(rejectOutfitIncoherence([blazer, knitPolo, chinos, sneakers], 'old_money', 'work')).not.toBeNull();
    expect(rejectOutfitIncoherence([oxford, chinos, sneakers], 'business_casual', 'work')).not.toBeNull();
    expect(rejectOutfitIncoherence([knitPolo, chinos, sneakers], 'casual', 'work')).toBeNull();
    expect(rejectOutfitIncoherence([knitPolo, chinos, sneakers], 'casual', 'casual')).toBeNull();
  });

  it('vetoes basic tees under tailoring, spares knit polos and knit tees', () => {
    const tee = { name: 'White heavyweight tee', macroCategory: 'top' };
    expect(rejectOutfitIncoherence([blazer, tee, chinos, sneakers], 'classic', 'casual')).not.toBeNull();
    expect(rejectOutfitIncoherence([blazer, knitPolo, chinos, sneakers], 'classic', 'casual')).not.toBeNull(); // sneakers, not the polo
    expect(
      rejectOutfitIncoherence(
        [blazer, { name: 'Navy knit tee', macroCategory: 'top' }, chinos, { name: 'Brown loafers', macroCategory: 'shoes' }],
        'old_money', 'work',
      ),
    ).toBeNull();
  });

  it('lets black pair with white, vetoes lonely black (minimalist exempt)', () => {
    const blackBlazer = { name: 'Black blazer', macroCategory: 'outerwear', color: 'black' };
    const whiteShirt = { name: 'White shirt', macroCategory: 'top', color: 'white' };
    const navyTrousers = { name: 'Navy trousers', macroCategory: 'bottom', color: 'navy' };
    const loafers = { name: 'Brown loafers', macroCategory: 'shoes', color: 'brown' };
    expect(rejectOutfitIncoherence([blackBlazer, whiteShirt, navyTrousers, loafers], 'old_money', 'work')).toBeNull();
    const blueOxford = { name: 'Blue oxford shirt', macroCategory: 'top', color: 'light blue' };
    expect(rejectOutfitIncoherence([blackBlazer, blueOxford, navyTrousers, loafers], 'old_money', 'work')).not.toBeNull();
    expect(
      rejectOutfitIncoherence(
        [{ name: 'Black shirt', macroCategory: 'top' }, { name: 'Black trousers', macroCategory: 'bottom' }],
        'minimalist', 'date',
      ),
    ).toBeNull();
  });

  it('vetoes striped layers over foreign bottoms, spares matching suits', () => {
    const stripedBlazer = { name: 'Navy pinstripe blazer', macroCategory: 'outerwear', color: 'navy' };
    const charcoalTrousers = { name: 'Charcoal trousers', macroCategory: 'bottom', color: 'charcoal' };
    const navyPinstripeTrousers = { name: 'Navy pinstripe trousers', macroCategory: 'bottom', color: 'navy' };
    expect(rejectOutfitIncoherence([stripedBlazer, oxford, charcoalTrousers], 'classic', 'work')).not.toBeNull();
    expect(rejectOutfitIncoherence([stripedBlazer, oxford, navyPinstripeTrousers], 'classic', 'work')).toBeNull();
    expect(rejectOutfitIncoherence([stripedBlazer, oxford, charcoalTrousers], 'casual', 'casual')).toBeNull();
  });

  it('reads black/white from the color column, not just the name', () => {
    // "Linen suit trousers" in black must fail even with a colorless name.
    const blackTrousers = { name: 'Linen suit trousers', macroCategory: 'bottom', color: 'black' };
    const navyBlazer = { name: 'Navy blazer', macroCategory: 'outerwear', color: 'navy' };
    const whiteShirt = { name: 'Oxford shirt', macroCategory: 'top', color: 'white' };
    const loafers = { name: 'Brown loafers', macroCategory: 'shoes', color: 'brown' };
    expect(rejectOutfitIncoherence([navyBlazer, blackTrousers, loafers], 'old_money', 'work')).not.toBeNull();
    expect(rejectOutfitIncoherence([navyBlazer, whiteShirt, blackTrousers, loafers], 'old_money', 'work')).toBeNull();
  });

  it('vetoes square toes everywhere, exotics with classics', () => {
    expect(rejectForStyleOccasion({ name: 'Square-toe derby', macroCategory: 'shoes' }, 'casual', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Crocodile loafers', macroCategory: 'shoes' }, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Crocodile loafers', macroCategory: 'shoes' }, 'casual', 'casual')).toBeNull();
  });
});

describe('rejectOutfitIncoherence — fabric and pairing doctrine', () => {
  const blazer = { name: 'Navy blazer', macroCategory: 'outerwear' };
  const chinos = { name: 'Beige chinos', macroCategory: 'bottom' };
  const loafers = { name: 'Brown penny loafers', macroCategory: 'shoes' };

  it('vetoes piqué polos under tailoring, spares them with chinos', () => {
    const pique = { name: 'Blue piqué polo shirt', macroCategory: 'top' };
    expect(rejectOutfitIncoherence([blazer, pique, chinos, loafers], 'old_money', 'work')).not.toBeNull();
    expect(rejectOutfitIncoherence([pique, chinos, loafers], 'old_money', 'casual')).toBeNull();
    expect(
      rejectOutfitIncoherence([{ name: 'Cream knit polo', macroCategory: 'top' }, blazer, chinos, loafers], 'old_money', 'work'),
    ).toBeNull();
  });

  it('vetoes cheap short-sleeve shirts, spares Hawaiian camp collars', () => {
    const cheap = { name: 'Classic short-sleeve shirt', macroCategory: 'top' };
    expect(rejectOutfitIncoherence([cheap, chinos, loafers], 'classic', 'casual')).not.toBeNull();
    expect(
      rejectOutfitIncoherence(
        [{ name: 'Short-sleeve Hawaiian camp collar shirt', macroCategory: 'top' }, chinos, loafers],
        'classic', 'casual',
      ),
    ).toBeNull();
  });

  it('vetoes suede shoes with suits, spares them with blazers', () => {
    const suede = { name: 'Snuff suede loafers', macroCategory: 'shoes' };
    const suitJacket = { name: 'Charcoal suit jacket', macroCategory: 'outerwear' };
    expect(rejectOutfitIncoherence([suitJacket, chinos, suede], 'business_casual', 'work')).not.toBeNull();
    expect(rejectOutfitIncoherence([blazer, chinos, suede], 'business_casual', 'work')).toBeNull();
  });

  it('vetoes white shirts with jeans', () => {
    const whiteShirt = { name: 'White oxford shirt', macroCategory: 'top' };
    const jeans = { name: 'Dark indigo jeans', macroCategory: 'bottom' };
    expect(rejectOutfitIncoherence([whiteShirt, jeans, loafers], 'old_money', 'casual')).not.toBeNull();
    expect(
      rejectOutfitIncoherence(
        [{ name: 'Blue oxford shirt', macroCategory: 'top' },
          { name: 'Beige chinos', macroCategory: 'bottom' }, loafers],
        'old_money', 'casual',
      ),
    ).toBeNull();
  });
});

describe('rejectForStyleOccasion — mules, mandarin collars, work denim', () => {
  it('rejects backless loafers for formal and classic decks', () => {
    const mule = { name: 'Side view of a brown backless loafer', macroCategory: 'shoes' };
    expect(rejectForStyleOccasion(mule, 'casual', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(mule, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion(mule, 'casual', 'casual')).toBeNull();
  });

  it('rejects mandarin collars on cloth for formal and classic', () => {
    const mandarin = { name: 'Navy blazer, stand-up collar', macroCategory: 'outerwear' };
    expect(rejectForStyleOccasion(mandarin, 'casual', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(mandarin, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'White polo shirt', macroCategory: 'top' }, 'old_money', 'casual')).toBeNull();
  });

  it('rejects denim bottoms for classic styles and workish occasions', () => {
    const jeans = { name: 'Tapered fit jeans', macroCategory: 'bottom' };
    expect(
      rejectOutfitIncoherence(
        [{ name: 'Navy blazer', macroCategory: 'outerwear' }, jeans,
          { name: 'Brown loafers', macroCategory: 'shoes' }],
        'old_money', 'casual',
      ),
    ).not.toBeNull();
    expect(
      rejectOutfitIncoherence(
        [{ name: 'Navy knit polo', macroCategory: 'top' }, jeans,
          { name: 'White sneakers', macroCategory: 'shoes' }],
        'casual', 'work',
      ),
    ).not.toBeNull();
    expect(
      rejectOutfitIncoherence(
        [{ name: 'Navy knit polo', macroCategory: 'top' }, jeans,
          { name: 'White sneakers', macroCategory: 'shoes' }],
        'casual', 'casual',
      ),
    ).toBeNull();
  });
});

describe('detectSlotMismatch — feed mistags', () => {
  it('repairs a short-sleeved shirt tagged as pants (screenshot: black shirt as PANTS)', () => {
    expect(detectSlotMismatch('bottom', 'Black short-sleeved cotton shirt', 'bottoms')).toBe('top');
    expect(detectSlotMismatch('bottom', 'Black short-sleeved cropped overshirt', 'lower_body')).toBe('top');
  });

  it('repairs a suit jacket tagged as top back to outerwear', () => {
    expect(detectSlotMismatch('top', 'Cool wool suit jacket', 'tops')).toBe('outerwear');
  });

  it('remaps a tee tagged as outerwear back to top', () => {
    expect(detectSlotMismatch('outerwear', '100% cotton short sleeve T-shirt', 'tops')).toBe('top');
    expect(detectSlotMismatch('top', 'Wool-blend short-sleeve knit sweater')).toBeNull();
    expect(detectSlotMismatch('outerwear', 'Wool-blend short-sleeve knit sweater')).toBe('top');
  });

  it('keeps boot-cut jeans as bottoms and oxford shirts as tops', () => {
    expect(detectSlotMismatch('bottom', 'Boot-cut dark jeans', 'bottoms')).toBeNull();
    expect(detectSlotMismatch('top', 'White oxford shirt', 'tops')).toBeNull();
  });

  it('keeps vague names untouched (no false repair)', () => {
    expect(detectSlotMismatch('top', 'Blue summer thing', 'tops')).toBeNull();
    expect(detectSlotMismatch('shoes', 'Leather loafers with penny strap', 'shoes')).toBeNull();
  });

  it('judge vetoes slot-mismatched cards in any style', () => {
    const bad = [
      { name: 'Navy blazer', macroCategory: 'outerwear' },
      { name: 'Short sleeve linen T-shirt', macroCategory: 'bottom' },
      { name: 'Brown loafers', macroCategory: 'shoes' },
    ];
    expect(passesStyleJudge(bad, 'casual', 'casual')).toBe(false);
  });
});

describe('rejectForStyleOccasion — chunky soles', () => {
  it('rejects chunky-soled shoes for formal and classic decks', () => {
    const chunky = { name: 'Textured black shoe with chunky lug sole', macroCategory: 'shoes' };
    expect(rejectForStyleOccasion(chunky, 'casual', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(chunky, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion(chunky, 'casual', 'casual')).toBeNull();
  });

  it('leaves chunky knitwear alone for date/casual (shoes-scoped)', () => {
    expect(rejectForStyleOccasion({ name: 'Chunky knit sweater', macroCategory: 'outerwear' }, 'old_money', 'date')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Chunky knit sweater', macroCategory: 'outerwear' }, 'old_money', 'casual')).toBeNull();
  });
});

describe('rejectForStyleOccasion — zip jackets, relaxed plaid, casual loafers', () => {
  it('rejects total-look / zip layers for classic and formal decks', () => {
    const zip = { name: '100% linen total look jacket', macroCategory: 'outerwear' };
    expect(rejectForStyleOccasion(zip, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(zip, 'old_money', 'casual')).not.toBeNull();
    expect(rejectForStyleOccasion(zip, 'casual', 'casual')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Navy wool blazer', macroCategory: 'outerwear' }, 'old_money', 'work')).toBeNull();
  });

  it('rejects relaxed-fit plaid shirts and split-leather loafers for old money work', () => {
    expect(rejectForStyleOccasion({ name: 'Zara Relaxed Fit Plaid Shirt', macroCategory: 'top' }, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'White oxford shirt', macroCategory: 'top' }, 'old_money', 'work')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Split leather loafers with white sole', macroCategory: 'shoes' }, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Brown penny loafers', macroCategory: 'shoes' }, 'old_money', 'work')).toBeNull();
    expect(rejectForStyleOccasion({ name: 'Split leather loafers with moc toe', macroCategory: 'footwear' }, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Split leather loafers with moc toe' }, 'old_money', 'work')).not.toBeNull();
  });

  it('rejects washed fashion jackets, splatter pants, and piqué polos for work', () => {
    expect(rejectForStyleOccasion({ name: 'Red washed-effect jacket', macroCategory: 'outerwear' }, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Light tan bell-bottom pants with white splatter effect', macroCategory: 'bottom' }, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Micro textured cotton piqué polo shirt', macroCategory: 'top' }, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: '100% cool wool double-breasted suit blazer', macroCategory: 'outerwear' }, 'old_money', 'work')).toBeNull();
  });
});

describe('humanizeProductName', () => {
  it.each([
    [
      'Front view of a white long-sleeve button-down shirt with a front button closure and buttoned cuffs.',
      'White long-sleeve button-down shirt with a front button closure and buttoned cuffs.',
    ],
    [
      'Zara RELAXED FIT STRIPED JACQUARD T-SHIRT - Blue - Image 0',
      'Zara Relaxed Fit Striped Jacquard T-shirt - Blue',
    ],
    [
      'Front view of a black quilted jacket with a stand-up collar - Image 3',
      'Black quilted jacket with a stand-up collar',
    ],
    ['100% algodón mercerised cotton T-shirt', '100% algodón mercerised cotton T-shirt'],
    ['Slim fit 100% linen shirt', 'Slim fit 100% linen shirt'],
    ['Side view of black loafers', 'Black loafers'],
    ['', ''],
  ])('humanizes %s', (raw, expected) => {
    expect(humanizeProductName(raw)).toBe(expected);
  });
});

describe('work vs date silhouettes (log replay)', () => {
  const jeans = { name: 'Selvedge fit jeans', macroCategory: 'bottom' };
  const denimShirt = { name: '100% cotton regular fit denim shirt', macroCategory: 'top' };
  const knitLayer = { name: 'Cotton and linen mouliné knit sweater', macroCategory: 'outerwear' };
  const woolBlazer = { name: 'Navy wool blazer', macroCategory: 'outerwear' };
  const linenBlazer = { name: '100% linen blazer', macroCategory: 'outerwear' };
  const boardroom = { name: '100% cool wool double-breasted suit blazer', macroCategory: 'outerwear' };
  const oxford = { name: '100% cotton regular fit Oxford shirt', macroCategory: 'top' };

  it('vetoes jeans, denim shirts, and knit-only layers for work', () => {
    expect(rejectForStyleOccasion(jeans, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(denimShirt, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(knitLayer, 'old_money', 'work')).not.toBeNull();
    expect(rejectForStyleOccasion(woolBlazer, 'old_money', 'work')).toBeNull();
    expect(rejectForStyleOccasion(oxford, 'old_money', 'work')).toBeNull();
  });

  it('keeps knit layers and denim shirts legal for date, rejects boardroom shells', () => {
    expect(rejectForStyleOccasion(knitLayer, 'old_money', 'date')).toBeNull();
    expect(rejectForStyleOccasion(denimShirt, 'old_money', 'date')).toBeNull();
    expect(matchesOccasionSilhouette(boardroom, 'date')).toBe(false);
    expect(matchesOccasionSilhouette(linenBlazer, 'date')).toBe(true);
    expect(matchesOccasionSilhouette(woolBlazer, 'work')).toBe(true);
    expect(matchesOccasionSilhouette(knitLayer, 'work')).toBe(false);
    expect(matchesOccasionSilhouette(boardroom, 'travel')).toBe(false);
    expect(matchesOccasionSilhouette(linenBlazer, 'travel')).toBe(true);
  });

  it('rejects phone cases as outfit pieces', () => {
    expect(rejectForStyleOccasion({ name: 'Nappa leather iPhone 16 Pro Max case', macroCategory: 'top' }, 'old_money', 'work')).not.toBeNull();
  });

  it('rejects jeans on formal occasions even for minimalist', () => {
    expect(rejectForStyleOccasion({ name: 'Selvedge fit jeans', macroCategory: 'bottom' }, 'minimalist', 'formal')).not.toBeNull();
  });

  it('rejects short-sleeve tees and relaxed trousers on formal occasions', () => {
    expect(
      rejectForStyleOccasion({ name: 'Short-sleeved navy t-shirt', macroCategory: 'top' }, 'old_money', 'formal'),
    ).not.toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Relaxed fit wool blend trousers', macroCategory: 'bottom' }, 'old_money', 'formal'),
    ).not.toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'White oxford shirt', macroCategory: 'top' }, 'old_money', 'formal'),
    ).toBeNull();
    expect(
      rejectForStyleOccasion({ name: 'Darted wool trousers', macroCategory: 'bottom' }, 'old_money', 'formal'),
    ).toBeNull();
  });

  it('rejects boardroom suiting for travel, keeps a linen overshirt', () => {
    expect(rejectForStyleOccasion({ name: '100% cool wool double-breasted suit blazer', macroCategory: 'outerwear' }, 'semi_classic', 'travel')).not.toBeNull();
    expect(rejectForStyleOccasion({ name: 'Linen overshirt', macroCategory: 'outerwear' }, 'semi_classic', 'travel')).toBeNull();
  });
});

describe('climate — linen at 15°C is not a real outfit', () => {
  const cool = { temp: 15, condition: 'cool' };
  const warm = { temp: 28, condition: 'warm' };

  it('infers October as 15°C cool in the northern hemisphere', () => {
    expect(inferSeasonalWeather(new Date(2026, 9, 5))).toEqual({ temp: 15, condition: 'cool' });
  });

  it('rejects 100% linen cloth and espadrilles when it is 15°C', () => {
    expect(rejectForClimate({ name: '100% linen blazer', macroCategory: 'outerwear' }, cool)).not.toBeNull();
    expect(rejectForClimate({ name: '100% linen slim fit shirt', macroCategory: 'top' }, cool)).not.toBeNull();
    expect(rejectForClimate({ name: '100% linen trousers', macroCategory: 'bottom' }, cool)).not.toBeNull();
    expect(rejectForClimate({ name: 'Split leather espadrille', macroCategory: 'shoes' }, cool)).not.toBeNull();
    expect(rejectForClimate({ name: 'Navy wool blazer', macroCategory: 'outerwear' }, cool)).toBeNull();
    expect(rejectForClimate({ name: 'White oxford shirt', macroCategory: 'top' }, cool)).toBeNull();
    expect(rejectForClimate({ name: 'Darted wool trousers', macroCategory: 'bottom' }, cool)).toBeNull();
    expect(rejectForClimate({ name: 'Leather penny loafers', macroCategory: 'shoes' }, cool)).toBeNull();
  });

  it('spares wool-linen sportcoats and linen-blend knits in the cool', () => {
    expect(rejectForClimate({ name: 'Wool-linen silk sport coat', macroCategory: 'outerwear' }, cool)).toBeNull();
    expect(rejectForClimate({ name: 'Cotton and linen mouliné knit sweater', macroCategory: 'outerwear' }, cool)).toBeNull();
  });

  it('allows linen on a 28°C day and blocks a puffer', () => {
    expect(rejectForClimate({ name: '100% linen blazer', macroCategory: 'outerwear' }, warm)).toBeNull();
    expect(rejectForClimate({ name: 'Down puffer jacket', macroCategory: 'outerwear' }, warm)).not.toBeNull();
  });

  it('fails the outfit judge when a 15°C look is all linen', () => {
    const linenLook = [
      { name: '100% linen blazer', macroCategory: 'outerwear' },
      { name: '100% linen slim fit shirt', macroCategory: 'top' },
      { name: '100% linen trousers', macroCategory: 'bottom' },
      { name: 'Leather loafers', macroCategory: 'shoes' },
    ];
    const woolLook = [
      { name: 'Navy wool blazer', macroCategory: 'outerwear' },
      { name: 'White oxford shirt', macroCategory: 'top' },
      { name: 'Darted wool trousers', macroCategory: 'bottom' },
      { name: 'Leather penny loafers', macroCategory: 'shoes' },
    ];
    expect(passesStyleJudge(linenLook, 'semi_classic', 'casual', cool)).toBe(false);
    expect(rejectOutfitClimate(linenLook, cool)).not.toBeNull();
    expect(passesStyleJudge(woolLook, 'semi_classic', 'casual', cool)).toBe(true);
  });
});

test('a footwear brand name is not evidence that loafers are winter boots',()=>{
 const {rejectForRequestIntent}=require('../../backend/supabase/functions/_shared/occasionRules');
 expect(rejectForRequestIntent({name:'Lincoln penny loafers',macroCategory:'shoes',brand:'Thursday Boot Co.',description:'Brown leather loafers'},'winter',{temp:0})).toContain('winter boots');
});
