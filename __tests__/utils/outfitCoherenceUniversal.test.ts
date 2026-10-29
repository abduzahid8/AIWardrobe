import {
  rejectOutfitIncoherence,
  rejectForStyleOccasion,
  styleJudgeReason,
  passesStyleJudge,
  matchesOccasionSilhouette,
  isAccessoryPropItem,
} from '../../features/outfit-generator/utils/occasionRules';

const item = (
  name: string,
  macroCategory: string,
  extra: Record<string, string> = {},
): any => ({ id: name, name, macroCategory, ...extra });

describe('universal outfit coherence vetoes (all styles × occasions)', () => {
  it('veto: sneakers with tailored suit trousers in a casual deck', () => {
    const reason = rejectOutfitIncoherence(
      [
        item('Cool wool suit blazer', 'outerwear'),
        item('100% cotton knit polo shirt', 'top'),
        item('100% Wool Striped Suit Trousers', 'bottom'),
        item('Кроссовки Light Eva Runner', 'shoes'),
      ],
      'casual',
      'casual',
    );
    expect(reason).toBe('sneakers never pair with tailored trousers');
  });

  it('veto: hybrid cupsole derbies with flannel pants (no sneaker word in name)', () => {
    const reason = rejectOutfitIncoherence(
      [
        item('Herringbone suit blazer', 'outerwear'),
        item('Свитер Ls Vertical Texture', 'top'),
        item('Wool-blend flannel pants', 'bottom'),
        item('Туфли Hybrid Clean Derby', 'shoes'),
      ],
      'casual',
      'work',
    );
    expect(reason).toBe('sneakers never pair with tailored trousers');
  });

  it('pass: sneakers with jeans/chinos/joggers stay legal', () => {
    expect(
      rejectOutfitIncoherence(
        [
          item('Cotton overshirt', 'outerwear'),
          item('100% cotton V-neck knit polo', 'top'),
          item('Relaxed Fit Denim Chinos', 'bottom'),
          item('Кеды Harlem Core sneakers', 'shoes'),
        ],
        'casual',
        'casual',
      ),
    ).toBeNull();
    expect(
      rejectOutfitIncoherence(
        [
          item('Oversize cotton shirt', 'top'),
          item('100% wool jogger fit pants', 'bottom'),
          item('Кроссовки Modern Comfort', 'shoes'),
        ],
        'casual',
        'sport',
      ),
    ).toBeNull();
  });

  it('veto: two pullover knits (EN + RU names)', () => {
    const reason = rejectOutfitIncoherence(
      [
        item('100% Cotton knit polo', 'top'),
        item('Micro textured cotton polo', 'top'),
        item('Wide-leg chino trousers', 'bottom'),
        item('Leather penny loafers', 'shoes'),
      ],
      'casual',
      'casual',
    );
    expect(reason).toBe('two pullover knits never layer together');
    expect(
      rejectOutfitIncoherence(
        [
          item('100% Cotton mouliné knit sweater', 'top'),
          item('Свиттер Dc Knitted Zip', 'top'),
          item('Pleated trousers', 'bottom'),
          item('Leather derby shoes', 'shoes'),
        ],
        'classic',
        'work',
      ),
    ).toBe('two pullover knits never layer together');
  });

  it('veto: two outer coats (blazer + puffer, blazer + parka)', () => {
    expect(
      rejectOutfitIncoherence(
        [
          item('100% Wool Suit Blazer', 'outerwear'),
          item('Sbl-runway-m hooded puffer', 'top'),
          item('Wool trousers', 'bottom'),
          item('Oxford shoes', 'shoes'),
        ],
        'casual',
        'casual',
      ),
    ).toBe('two outer coats never layer together');
    expect(
      rejectOutfitIncoherence(
        [
          item('100% Wool Check Relaxed blazer', 'outerwear'),
          item('Long technical parka', 'top'),
          item('Slim tapered fit jeans', 'bottom'),
          item('Leather moc toe shoes', 'shoes'),
        ],
        'semi_classic',
        'casual',
      ),
    ).toBe('two outer coats never layer together');
  });

  it('pass: blazer over sweater / cardigan over shirt stay legal', () => {
    expect(
      rejectOutfitIncoherence(
        [
          item('Cool wool suit blazer', 'outerwear'),
          item('100% cashmere knit sweater', 'top'),
          item('100% wool suit pants', 'bottom'),
          item('Leather Chelsea boots', 'shoes'),
        ],
        'old_money',
        'work',
      ),
    ).toBeNull();
    expect(
      rejectOutfitIncoherence(
        [
          item('Wool knit cardigan', 'outerwear'),
          item('Oxford cotton shirt', 'top'),
          item('Pleated trousers', 'bottom'),
          item('Penny loafers', 'shoes'),
        ],
        'classic',
        'work',
      ),
    ).toBeNull();
  });

  it('styleJudgeReason surfaces the universal veto', () => {
    const items = [
      item('Wool suit blazer', 'outerwear'),
      item('Knit polo shirt', 'top'),
      item('100% Wool Pinstripe Suit Trousers', 'bottom'),
      item('Кроссовки Th Dressy Runner', 'shoes'),
    ];
    expect(styleJudgeReason(items, 'casual', 'casual')).toContain('tailored trousers');
    expect(passesStyleJudge(items, 'casual', 'casual')).toBe(false);
  });
});

describe('sport occasion shoe silhouette', () => {
  it('accepts trainers, rejects loafers/derbies/boots', () => {
    expect(matchesOccasionSilhouette(item('Кроссовки Tjm Classic', 'shoes'), 'sport')).toBe(true);
    expect(matchesOccasionSilhouette(item('Running training shoes', 'shoes'), 'sport')).toBe(true);
    expect(matchesOccasionSilhouette(item('Лоферы Hilfiger Stitchdown', 'shoes'), 'sport')).toBe(false);
    expect(matchesOccasionSilhouette(item('Leather Chelsea boots', 'shoes'), 'sport')).toBe(false);
    expect(matchesOccasionSilhouette(item('Туфли Hybrid Clean Derby', 'shoes'), 'sport')).toBe(false);
  });

  it('leaves non-shoe items untouched at sport', () => {
    expect(matchesOccasionSilhouette(item('100% cotton t-shirt', 'top'), 'sport')).toBe(true);
    expect(matchesOccasionSilhouette(item('Relaxed jogger pants', 'bottom'), 'sport')).toBe(true);
  });
});

describe('accessory props never fill clothing slots', () => {
  it('classifies key rings as accessory props', () => {
    expect(isAccessoryPropItem(item('Metal Chain Key Ring', 'top'))).toBe(true);
    expect(isAccessoryPropItem(item('keyring leather tag', 'accessory'))).toBe(true);
    expect(isAccessoryPropItem(item('100% cotton shirt', 'top'))).toBe(false);
  });

  it('classifies training towels as accessory props (EN + RU)', () => {
    expect(isAccessoryPropItem(item('Large Training Towel', 'top'))).toBe(true);
    expect(isAccessoryPropItem(item('Small Training Towel', 'top'))).toBe(true);
    expect(isAccessoryPropItem(item('Полотенце для тренировок', 'top'))).toBe(true);
  });

  it('veto: towel in TOP slot is rejected by the style judge', () => {
    const items = [
      item('Wool blazer', 'outerwear'),
      item('Large Training Towel', 'top'),
      item('100% cotton double dart pants', 'bottom'),
      item('Leather penny loafers', 'shoes'),
    ];
    expect(styleJudgeReason(items, 'casual', 'casual')).toContain('accessory prop');
    expect(passesStyleJudge(items, 'casual', 'casual')).toBe(false);
  });
});

describe('sloppy bottoms rejected regardless of jogger phrasing', () => {
  // Regression: "Basic Jogging Trousers" slipped old-money decks because the
  // sloppy-bottom regex only matched 'jogger(s)' and never 'jogging'.
  it('rejects "Basic Jogging Trousers" for old_money (style-scoped branch)', () => {
    expect(rejectForStyleOccasion(item('Basic Jogging Trousers', 'bottom'), 'old_money', 'casual'))
      .toContain('sloppy/track bottoms');
  });

  it('rejects "Basic Jogging Trousers" for semi_classic and minimalist', () => {
    expect(rejectForStyleOccasion(item('Basic Jogging Trousers', 'bottom'), 'semi_classic', 'casual'))
      .toContain('sloppy/track bottoms');
    expect(rejectForStyleOccasion(item('Basic Jogging Trousers', 'bottom'), 'minimalist', 'casual'))
      .toContain('minimalist discipline');
  });

  it('rejects it on a formal occasion too (formal branch fires first)', () => {
    expect(rejectForStyleOccasion(item('Basic Jogging Trousers', 'bottom'), 'old_money', 'work'))
      .toContain('sloppy bottoms underdress');
  });

  it('still rejects jogger/sweatpant spellings (no regression)', () => {
    expect(rejectForStyleOccasion(item('Slim jogger pants', 'bottom'), 'old_money', 'casual'))
      .toContain('sloppy/track bottoms');
    expect(rejectForStyleOccasion(item('Grey sweatpants', 'bottom'), 'classic', 'casual'))
      .toContain('sloppy/track bottoms');
  });

  it('styleJudgeReason surfaces the jogger bottom through the full deck', () => {
    const items = [
      item('100% wool suit blazer', 'outerwear'),
      item('Свитер Dc Knitted Zip Polo', 'top'),
      item('Basic Jogging Trousers', 'bottom'),
      item('Туфли Hybrid Clean Derby', 'shoes'),
    ];
    expect(styleJudgeReason(items, 'old_money', 'work')).toContain('sloppy');
    expect(passesStyleJudge(items, 'old_money', 'work')).toBe(false);
  });
});

describe('four-slot quality gates', () => {
  const base = [
    item('Lightweight linen cardigan', 'outerwear', {color:'navy'}),
    item('White cotton polo shirt', 'top', {color:'white'}),
    item('Beige cotton chinos', 'bottom', {color:'beige'}),
    item('Brown leather loafers', 'shoes', {color:'brown'}),
  ];
  it('allows an open lightweight cardigan over a base polo', () => {
    expect(styleJudgeReason(base, 'casual', 'summer, no jacket', {temp:30})).toBeNull();
  });
  it('rejects three competing vivid hues, while allowing one accent with neutrals', () => {
    const loud=base.map((p,i)=>({...p,color:['red','blue','yellow','brown'][i]}));
    expect(styleJudgeReason(loud,'casual','casual')).toBe('too many competing vivid colors');
    expect(styleJudgeReason(base.map((p,i)=>i===1?{...p,color:'red'}:p),'casual','casual')).toBeNull();
  });
  it('keeps weather and no-jacket restrictions when choosing the fourth item', () => {
    const heavy=base.map((p,i)=>i===0?{...p,name:'Insulated down parka jacket'}:p);
    expect(styleJudgeReason(heavy,'casual','summer',{temp:32})).toMatch(/warm/);
    expect(styleJudgeReason(heavy,'casual','no jacket',{temp:18})).toMatch(/non-jacket/);
  });
});
