import {
  detectSlotMismatch,
  passesStyleJudge,
  rejectForStyleOccasion,
  rejectOutfitIncoherence,
  styleJudgeReason,
} from '../../features/outfit-generator/utils/occasionRules';
import { inferShopRowMacro } from '../../src/utils/shopCatalogOrder';

// Russian-named sneakers (кроссовки/кеды) live in shop_catalog, but the
// judge's shoe regexes were English-only: scenario 45 rendered a suit
// blazer + Кроссовки card under old_money/work — a combo the English
// twin ("Sneakers Modern") would have been rejected for.
describe('russian sneaker/athletic detection in the style judge', () => {
  const blazer = { name: '100% cool wool suit blazer', macroCategory: 'outerwear' };
  const shirt = { name: '100% cotton regular fit poplin shirt', macroCategory: 'top' };
  const trousers = { name: 'Pleated Trousers With Side Adjusters', macroCategory: 'bottom' };
  const ruSneakers = { name: 'Кроссовки Modern Court Comfort', macroCategory: 'shoes' };

  it('rejects a suit blazer + russian sneakers outfit for old_money/work (scenario 45 card)', () => {
    const outfit = [blazer, shirt, trousers, ruSneakers];
    expect(passesStyleJudge(outfit, 'old_money', 'work')).toBe(false);
    const reason = styleJudgeReason(outfit, 'old_money', 'work');
    expect(reason).toMatch(/athletic shoes are not evening wear|sneakers never pair/);
  });

  it('rejects russian sneakers as an item for classic styles even on casual occasions', () => {
    const r = rejectForStyleOccasion(ruSneakers, 'old_money', 'casual');
    expect(r).toBe('boat/athletic shoes never pair with classic tailoring');
  });

  it('rejects russian sandals for formal occasions (beach/athletic family)', () => {
    const sandals = { name: 'Сандалии летние', macroCategory: 'shoes' };
    expect(rejectForStyleOccasion(sandals, 'minimalist', 'work')).toBe(
      'beach/athletic shoes are not evening wear',
    );
  });

  it('does NOT overban minimal russian sneakers in refined casual (smart-low lane)', () => {
    const knitPolo = { name: 'Knit polo sweater', macroCategory: 'outerwear' };
    const trousersOnly = { name: 'Chino trousers beige', macroCategory: 'bottom' };
    const whiteRu = { name: 'Кеды White Leather Minimal', macroCategory: 'shoes' };
    const outfit = [knitPolo, whiteRu, trousersOnly];
    expect(rejectOutfitIncoherence(outfit, 'casual', 'casual')).toBeNull();
    expect(passesStyleJudge(outfit, 'casual', 'casual')).toBe(true);
  });

  it('styleJudgeReason and passesStyleJudge stay in sync', () => {
    const good = [
      blazer,
      shirt,
      trousers,
      { name: 'Oxford shoes leather cap-toe', macroCategory: 'shoes' },
    ];
    expect(styleJudgeReason(good, 'old_money', 'work')).toBeNull();
    expect(passesStyleJudge(good, 'old_money', 'work')).toBe(true);
    const bad = [blazer, shirt, trousers, ruSneakers];
    expect(styleJudgeReason(bad, 'old_money', 'work')).not.toBeNull();
    expect(passesStyleJudge(bad, 'old_money', 'work')).toBe(false);
  });

  it('detectSlotMismatch repairs a russian shoe misclaimed as top', () => {
    expect(detectSlotMismatch('top', 'Кроссовки Modern Court Comfort')).toBe('shoes');
    expect(detectSlotMismatch('bottom', 'Ботинки Low Prof Desert')).toBe('shoes');
    expect(detectSlotMismatch('top', 'Лоферы Flex Hilfiger')).toBe('shoes');
  });

  it('inferShopRowMacro classifies russian shoe names from the name', () => {
    expect(inferShopRowMacro('accessories', 'Кроссовки Modern Court Comfort')).toBe('shoes');
    expect(inferShopRowMacro('', 'Туфли кожаные классические')).toBe('shoes');
  });
});
