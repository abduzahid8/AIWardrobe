import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import type { OutfitCoherence } from './outfitCoherence.ts';
import { isDateNightOccasion, type StylingContext } from './stylingContext.ts';

export interface OutfitQualityFloor {
  passed: boolean;
  score: number;
  minimum: number;
  reason: string | null;
}
/** Acceptance is separate from hard eligibility and recency ranking. Unknown
 * metadata stays neutral; a low result never licenses a hard-rule waiver. */
export function outfitQualityFloor(
  items: GarmentEvidence[],
  context: StylingContext,
  coherence: OutfitCoherence
): OutfitQualityFloor {
  const refinedWork =
    context.dressCode === 'smart' &&
    /\b(work|office|business)\b/.test(context.occasion) &&
    ['old_money', 'classic', 'semi_classic', 'business_casual'].includes(context.style);
  const formal = context.dressCode === 'formal';
  const dateNight = isDateNightOccasion(context.occasion);
  const oldMoney = context.style === 'old_money';
  const minimalistCasual = context.style === 'minimalist' && context.dressCode === 'casual';
  const casualTravel = context.dressCode === 'casual' && /\b(travel|trip|sightseeing)\b/.test(`${context.occasion} ${context.request}`);
  const minimum = formal || refinedWork || (oldMoney && dateNight) ? 0.75 : dateNight ? 0.7 : minimalistCasual ? 0.65 : casualTravel ? 0.6 : 0;
  const descriptors = items.map(describeGarment);
  const bySlot = (slot: string) => descriptors.find(d => d.slot === slot);
  let score = coherence.score;
  let reason: string | null = null;
  if (
    (formal || refinedWork || dateNight) &&
    Math.min(
      coherence.dimensions.layer,
      coherence.dimensions.footwear,
      // A smart casual date can be more polished than the requested casual
      // archetype without becoming unsuitable. Old Money dates and formal
      // settings still need a strong match to their specific style.
      formal || refinedWork || oldMoney ? coherence.dimensions.style : 1
    ) < 0.5
  ) {
    score = Math.min(score, 0.49);
    reason = 'weak_refined_occasion_compatibility';
  }
  if (refinedWork || formal || dateNight) {
    const top = items.find(item => item.macroCategory === 'top');
    const layer = items.find(item => item.macroCategory === 'outerwear');
    const bottom = items.find(item => item.macroCategory === 'bottom');
    const a = bySlot('top'), b = bySlot('outerwear'), trousers = bySlot('bottom');
    if (a?.utilityConstruction && b?.subtype === 'blazer') {
      score = Math.min(score, 0.49); reason = 'utility_shirt_conflicts_with_office_tailoring';
    } else if (b?.technicalLeisure && trousers?.structured) {
      score = Math.min(score, 0.49); reason = 'technical_leisure_conflicts_with_refined_tailoring';
    } else if ((layer && bottom && /\b(?:pure|100%)\s+linen\b/i.test(layer.name || '') && /\bflannel\b/i.test(`${bottom.name || ''} ${bottom.description || ''}`))
      || (top && layer && /\blinen\b/i.test(top.name || '') && /\bvelvet\b/i.test(layer.name || ''))
      || coherence.evidence.includes('heavy_and_airy_texture_contrast')) {
      score = Math.min(score, 0.49); reason = 'incompatible_heavy_and_airy_fabrics';
    } else if ((coherence.evidence.includes('competing_upper_patterns') || coherence.evidence.includes('competing_outfit_motifs'))) {
      score = Math.min(score, 0.49); reason = 'competing_patterns_in_office_tailoring';
    }
  }
  const top = bySlot('top'), outer = bySlot('outerwear'), bottom = bySlot('bottom'), shoe = bySlot('shoes');
  const dressSilhouette = shoe && !shoe.casualSole && ['oxford', 'monk', 'dress-boot'].includes(shoe.subtype);
  const reject = (why: string) => { score = Math.min(score, 0.49); reason = why; };
  if (coherence.evidence.includes('fluid_base_and_short_zip_layer_split_office_register'))
    reject('casual_upper_construction_conflicts_with_office_dress_lower');
  if (coherence.evidence.includes('graphic_volume_tailoring_and_vivid_shoes_lack_shared_register'))
    reject('disconnected_graphic_tailoring_and_statement_footwear');
  if (coherence.evidence.includes('performance_training_shoe_without_active_travel_intent'))
    reject('training_footwear_conflicts_with_non_active_travel_look');
  // Wash alone remains usable. Only a close cut, leisure denim layer AND
  // a formal shoe last establish this specific date-night register clash.
  if (dateNight && bottom?.washedDenim && bottom.volume === 'slim'
    && outer?.relaxedTailoring && outer.materials.includes('denim')
    && shoe && !shoe.casualSole && ['oxford', 'derby', 'monk', 'dress-boot'].includes(shoe.subtype))
    reject('washed_slim_double_denim_conflicts_with_dress_footwear');
  if ((formal || context.dressCode === 'smart') && top?.materials.includes('denim')
    && bottom?.paleDenim && outer?.structured && dressSilhouette)
    reject('pale_double_denim_conflicts_with_tailoring_and_dress_footwear');
  // Whole-look conflicts cannot be averaged away by three strong pieces.
  // Use positive construction evidence; keep chinos, technical travel sets,
  // summer linen drawstrings and slim rubber dress soles available.
  if (bottom?.structured && shoe?.casualSole
    && (shoe.soleProfile === 'sneaker' || formal || context.dressCode === 'smart'
      || outer?.subtype === 'blazer' || top?.subtype === 'dress-shirt')
    && ['oxford', 'derby', 'monk', 'loafer', 'dress-boot'].includes(shoe.subtype))
    reject('hybrid_or_comfort_sole_conflicts_with_dress_trousers');
  const bottomItem = items.find(item => item.macroCategory === 'bottom');
  const bottomText = `${bottomItem?.name || ''} ${bottomItem?.description || ''} ${bottomItem?.material || ''}`;
  if (bottom?.athletic && /\b(jersey|interlock|sweatpants?|joggers?|knitted|knit)\b/i.test(bottomText)
    && shoe?.subtype === 'driver')
    reject('jersey_drawstring_bottom_conflicts_with_driving_loafers');
  if ((formal || context.dressCode === 'smart') && bottom?.fadedDenim
    && (outer?.structured || shoe?.formality === 'formal'))
    reject('washed_denim_conflicts_with_refined_tailoring');
  const airyLinen = descriptors.some(d => d.slot !== 'shoes' && d.bulk === 'light' && d.materials.includes('linen'));
  const winterLayer = descriptors.some((d, index) => d.slot !== 'shoes'
    && (d.insulated || (d.bulk === 'heavy' && /\b(wool|cashmere|flannel|velvet|winter|cable[ -]knit)\b/i.test(
      `${items[index].name || ''} ${items[index].description || ''} ${items[index].material || ''}`))));
  if (airyLinen && winterLayer) reject('lightweight_linen_conflicts_with_heavy_winter_layers');
  // A formal last and heavily casual surface cannot be rescued by three
  // individually high scores. Dark clean denim with loafers remains valid.
  if (dressSilhouette && bottom?.fadedDenim) reject('dress_footwear_conflicts_with_faded_denim');
  if (dressSilhouette && outer?.technicalLeisure) reject('dress_footwear_conflicts_with_technical_layer');
  if ((refinedWork || formal || (oldMoney && dateNight)) && shoe?.casualSole)
    reject('casual_sole_conflicts_with_refined_occasion');
  if (oldMoney) {
    // Preserve the existing owned-anchor aesthetic exception. Relationships
    // (footwear, fabric, layers) still apply to every item including anchors.
    if (items.some((item, i) => String(item.id || '') !== context.anchorId && descriptors[i].loudBranding))
      reject('loud_branding_conflicts_with_old_money');
    if (bottom?.athletic || outer?.sportyZip) reject('sporting_construction_conflicts_with_old_money');
  }
  if (dateNight) {
    if (bottom?.fadedDenim) reject('faded_denim_underdresses_date_night');
    if (bottom?.athletic || bottom?.subtype === 'cargo' || outer?.sportyZip || outer?.subtype === 'hoodie'
      || shoe?.athletic || shoe?.rugged || ['sport-sneaker','sandal','espadrille','driver'].includes(shoe?.subtype || ''))
      reject('sporting_register_underdresses_date_night');
    // Quiet fabrics may carry one patterned/accent piece; avoid competing
    // upper motifs without imposing monochrome or a universal pattern ban.
    if ((coherence.evidence.includes('competing_upper_patterns') || coherence.evidence.includes('competing_outfit_motifs')))
      reject('competing_patterns_undermine_date_night');
    if (coherence.evidence.includes('heavy_and_airy_texture_contrast'))
      reject('incompatible_date_night_fabric_weights');
    if (coherence.dimensions.palette < 0.5) reject('uncoordinated_date_night_palette');
  }
  if (minimalistCasual) {
    const layer = items.find((item) => item.macroCategory === 'outerwear');
    const bottom = items.find((item) => item.macroCategory === 'bottom');
    const text = `${bottom?.name || ''} ${bottom?.description || ''}`.toLowerCase();
    const layerText = `${layer?.name || ''} ${layer?.description || ''}`.toLowerCase();
    const severeDistress =
      /\b(ripped|torn|destroyed|shredded|holes?)\b|\b(heavily|heavy|overly|extremely|severely)[ -]+distressed\b/.test(
        text
      );
    if (
      layer &&
      bottom &&
      describeGarment(layer).subtype === 'blazer' &&
      describeGarment(bottom).subtype === 'jeans' &&
      severeDistress &&
      !/\b(unstructured|casual|unlined|jersey|knitted)\b/.test(layerText)
    ) {
      score = Math.min(score, 0.49);
      reason = 'business_tailoring_with_severely_distressed_denim';
    }
  }
  const passed = coherence.valid && reason === null && score >= minimum;
  return {
    passed,
    score,
    minimum,
    reason: passed ? null : reason || 'below_occasion_style_quality_floor',
  };
}
