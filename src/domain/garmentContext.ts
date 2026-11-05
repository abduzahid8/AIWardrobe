import { normalizeGarmentText } from './garmentText.ts';
import { patternOf } from './outfitDiversity.ts';
import { STYLING_PROFILES } from './stylingProfiles.ts';
import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
/** Same hard dress/activity exclusions before pool caps and after construction. */
export function garmentContextReason(
  item: GarmentEvidence,
  context: StylingContext
): string | null {
  const shop = item.isShopItem === true || String(item.id || '').startsWith('shop_');
  if (context.sourceMode === 'wardrobe' && shop) return 'source_requires_wardrobe';
  if (context.sourceMode === 'catalog' && !shop) return 'source_requires_catalog';
  const d = describeGarment(item);
  if (context.dressCode === 'smart' && /\b(work|office|business)\b/.test(context.occasion)
    && ['classic','old_money','semi_classic','business_casual'].includes(context.style) && d.slot === 'outerwear') {
    const text = normalizeGarmentText(`${item.name || ''} ${item.description || ''}`);
    const cleanWovenJacket = d.subtype === 'jacket' && /\b(wool|linen)\b/.test(text)
      && !/\b(biker|trucker|racer|denim|leather|suede|utility|military|athletic|running)\b/.test(text);
    if (!['blazer','cardigan','coat'].includes(d.subtype) && !cleanWovenJacket)
      return 'work_requires_smart_layer';
  }
  const explicit =
    typeof item.color === 'string' && !/^(unknown|neutral)$/i.test(item.color)
      ? item.color
      : item.name || '';
  const colorEvidence =
    ' ' +
    normalizeGarmentText(explicit)
      .replace(/[^\p{L}]+/gu, ' ')
      .trim() +
    ' ';
  if (
    context.preferences?.avoidColors?.some((color) => {
      const value = normalizeGarmentText(color)
        .replace(/[^\p{L}]+/gu, ' ')
        .trim();
      return !!value && colorEvidence.includes(' ' + value + ' ');
    })
  )
    return 'excluded_color';
  if (
    d.pattern !== 'unknown' &&
    context.preferences?.dislikedPatterns?.some(
      (pattern) =>
        /stripe|check|plaid|graphic|logo|floral|camo|plain|solid|paisley|jacquard/i.test(pattern) &&
        patternOf({ pattern }) === d.pattern
    )
  )
    return 'excluded_pattern';
  if (context.dressCode === 'formal') {
    if (
      ['top', 'outerwear', 'bottom'].includes(d.slot) &&
      /\b(corduroy|denim|fleece|sweatshirt)\b/i.test(`${item.name || ''} ${item.description || ''}`)
    )
      return 'formal_rejects_casual_fabric';
    if (d.slot === 'outerwear' && d.subtype !== 'blazer') return 'formal_requires_structured_layer';
    if (d.slot === 'top' && (!['shirt', 'dress-shirt'].includes(d.subtype) || (d.texture === 'knit' && d.subtype !== 'dress-shirt')))
      return 'formal_requires_dress_base';
    if (d.slot === 'bottom' && d.subtype !== 'tailored-trouser')
      return 'formal_requires_tailored_bottom';
    if (
      d.slot === 'shoes' &&
      (!['oxford', 'derby', 'monk', 'dress-boot', 'chelsea', 'loafer'].includes(d.subtype) ||
        d.rugged ||
        (d.subtype === 'loafer' && d.formality !== 'formal') ||
        d.athletic)
    )
      return 'formal_requires_dress_footwear';
  }
  const outdoor = /\b(hik(?:e|ing)|trek(?:king)?|trail|mountain|snow)\b/i.test(
    `${context.activity} ${context.request}`
  );
  const severe =
    context.weather?.temp != null &&
    context.weather.temp < 5 &&
    /snow|ice/i.test(context.weather.condition || '');
  if (context.dressCode === 'smart' && d.slot === 'shoes' && d.rugged && !outdoor && !severe)
    return 'rugged_footwear_conflicts_with_smart_context';
  if (context.dressCode === 'athletic') {
    if (d.slot === 'shoes' && !d.athletic) return 'workout_requires_athletic_footwear';
    if (d.slot === 'bottom' && d.structured) return 'workout_rejects_restrictive_tailoring';
  }
  return null;
}

/** Soft profile preference. Unknown evidence stays neutral; never a hard waiver. */
export function scoreGarmentForContext(item: GarmentEvidence, context: StylingContext): number {
  const d = describeGarment(item);
  let score = 0.5;
  const evening = /black[ -]?tie|tuxedo|gala|смокинг/i.test(
    `${context.occasion} ${context.request}`
  );
  if (d.evening && !evening) return 0.05;
  if (d.subtype === 'unknown') return score;
  if (context.dressCode === 'formal')
    return d.structured || ['oxford', 'derby', 'monk', 'dress-boot'].includes(d.subtype) ? 1 : 0.8;
  if (context.dressCode === 'athletic') return d.athletic ? 0.95 : 0.25;
  if (context.dressCode === 'smart') {
    score = [
      'blazer',
      'cardigan',
      'dress-shirt',
      'shirt',
      'fine-knit',
      'tailored-trouser',
      'chino',
      'loafer',
      'chelsea',
      'dress-boot',
      'oxford',
      'derby',
    ].includes(d.subtype)
      ? 0.9
      : 0.45;
    if (d.rugged) score = 0.1;
    // Work's tailoring and clean knit layers should lead the shortlist.
    // Casual leather/denim jackets remain available for other occasions.
    if (/\b(work|office|business)\b/.test(context.occasion) && d.slot === 'outerwear') {
      if (['blazer', 'cardigan'].includes(d.subtype)) score = 1;
      if (/\b(biker|trucker|racer|denim)\b/.test(normalizeGarmentText(`${item.name || ''} ${item.description || ''}`))) score = 0.15;
    }
  } else {
    score = [
      'overshirt',
      'jacket',
      'bomber',
      't-shirt',
      'polo',
      'shirt',
      'chino',
      'jeans',
      'sneaker',
      'minimal-sneaker',
      'driver',
    ].includes(d.subtype)
      ? 0.9
      : 0.6;

  }
  // Travel is an activity, independent of the requested aesthetic. Do not
  // claim all leather shoes are comfortable: rank only supported features.
  const travel = /\b(travel|trip|sightseeing)\b/.test(`${context.occasion} ${context.request}`);
  const businessTravel = /\b(business|meeting|conference)\b/.test(context.request);
  if (travel && !businessTravel) {
    if (['oxford', 'monk', 'blazer'].includes(d.subtype)) score -= 0.3;
    if (d.evening) score = 0.05;
    if (d.bulk === 'heavy' && (!context.weather || (context.weather.temp ?? 15) > 12)) score -= 0.2;
    if (['cardigan', 'overshirt', 'chino', 'minimal-sneaker'].includes(d.subtype)) score += 0.2;
    const text = normalizeGarmentText(`${item.name || ''} ${item.description || ''}`);
    if (d.slot === 'shoes' && /\b(cushioned|flexible|walking|padded insole|comfort)\b/.test(text)) score += 0.2;
  }
  const profile = STYLING_PROFILES[context.style];
  if (profile) {
    score = score * 0.75 + (profile.preferred.includes(d.subtype) ? 1 : 0.4) * 0.25;
    if (profile.textures.includes(d.texture)) score += 0.025;
  }
  if (context.style === 'minimalist' && d.pattern !== 'solid' && d.pattern !== 'unknown')
    score -= 0.08;
  if (!context.weather && d.bulk === 'heavy') score -= 0.1;
  if (d.bulk === 'light' && ['cardigan', 'blazer', 'overshirt'].includes(d.subtype)) score += 0.05;
  return Math.max(0, Math.min(1, score));
}
