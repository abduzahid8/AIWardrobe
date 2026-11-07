import { repairEdgeItems, isFourSlotLook } from './normalizeSlots';
import { passesStyleJudge, styleJudgeReason } from './occasionRules';
import { closestAvailableOutfits, type AlternativeItem } from './closestWardrobeOutfits';

type Options = Parameters<typeof closestAvailableOutfits>[1] & {source: 'wardrobe' | 'shop'; includeReserve?: boolean; immutablePlan?: boolean};
interface EdgeLook {
  id?: string; items: AlternativeItem[]; confidence?: number; matchScore?: number;
  qualityAssessment?: {score: number};
}

/** The client boundary also supports older Edge releases without waiving its rules. */
export function prepareStylistResponse(proposals: EdgeLook[], inventory: AlternativeItem[], options: Options) {
  const rejections: Record<string, number> = {};
  const reject = (reason: string) => { rejections[reason] = (rejections[reason] || 0) + 1; };
  const cleaned: EdgeLook[] = proposals.map(look => ({...look, items: options.immutablePlan ? (look.items || []) : repairEdgeItems(look.items || [])}))
    .filter(look => {
      if (isFourSlotLook(look.items)) return true;
      reject('structure'); return false;
    });
  const accepted = cleaned.filter(look => {
    const sourceOK = look.items.every(item => options.source === 'shop'
      ? (item.isShopItem === true && item.isWardrobe !== true) || item.id === options.anchorId
      : item.isShopItem !== true && item.isWardrobe !== false);
    if (!sourceOK || (options.anchorId && !look.items.some(item => item.id === options.anchorId))) {
      reject('source_or_anchor'); return false;
    }
    if (!passesStyleJudge(look.items, options.style, options.occasion, options.weather, options.anchorId, options.preferences)) {
      reject(styleJudgeReason(look.items, options.style, options.occasion, options.weather, options.anchorId, options.preferences) || 'quality');
      return false;
    }
    if (options.accepts && !options.accepts(look.items)) { reject('budget'); return false; }
    return true;
  });
  const recovered = !options.immutablePlan && (accepted.length < 3 || options.includeReserve) ? closestAvailableOutfits([
    ...cleaned.flatMap(look => look.items), ...inventory,
  ], {...options, limit: 64, reserve: true}) : [];
  return {cleaned, accepted, recovered, rejections};
}
