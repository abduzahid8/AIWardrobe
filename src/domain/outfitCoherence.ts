import { outfitStyleIntent } from './stylingProfiles.ts';
import { garmentContextReason, scoreGarmentForContext, officeRefinementPenalty } from './garmentContext.ts';
import { describeGarment, competingUpperPatterns, type GarmentEvidence } from './garmentDescriptor.ts';
import { layerCompatibility } from './layerCompatibility.ts';
import { scoreColorHarmony } from './colorHarmony.ts';
import { colorFamilyOf, colorTokensOf, patternOf } from './outfitDiversity.ts';
import type { StylingContext } from './stylingContext.ts';
export interface OutfitCoherence {
  valid: boolean;
  score: number;
  reasons: string[];
  evidence: string[];
  dimensions: {
    layer: number;
    dressCode: number;
    palette: number;
    proportion: number;
    texture: number;
    footwear: number;
    style: number;
  };
}
/** Hard whole-outfit register clashes: tailored layering over gym-wear, gym
 * outerwear over formal bases, formal shoes with athletic bottoms. These fail
 * the look in every context. Single source shared by the coherence judge
 * below and the generation gate (rejectOutfitIncoherence) so the generator
 * never assembles a look the judge rejects. */
const LAYER_REGISTER_CLASH = new Set([
  'tailored_layer_over_athletic_base',
  'hood_under_tailored_blazer',
  'bulky_base_under_tailored_blazer',
  'athletic_outerwear_over_formal_base',
]);

export function wholeOutfitRegisterClash(
  items: GarmentEvidence[],
  context: StylingContext
): string | null {
  if (!items || items.length === 0) return null;
  const descriptors = items.map(describeGarment);
  const evidenceFor = (slot: string) => {
    const index = descriptors.findIndex((d) => d.slot === slot);
    return index >= 0 ? items[index] : undefined;
  };
  const top = evidenceFor('top'),
    layer = evidenceFor('outerwear'),
    bottomIndex = descriptors.findIndex((d) => d.slot === 'bottom'),
    shoeIndex = descriptors.findIndex((d) => d.slot === 'shoes');
  if (top && layer) {
    const pair = layerCompatibility(top, layer, context);
    const hit = pair.reasons.find((reason) => LAYER_REGISTER_CLASH.has(reason));
    if (hit) return hit;
  }
  // A Casual aesthetic cannot waive Formal coherence. A checked/striped
  // base competes with patterned tailoring even when each item alone passes.
  // Read positive catalog pattern evidence directly (including pinstripes).
  const visiblePattern = (item: GarmentEvidence) => ['check','stripe','print'].includes(patternOf(item))
    || /\b(chevron|geometric|paisley|jacquard|pattern(?:ed)?)\b/i.test(`${item.name || ''} ${item.description || ''} ${item.pattern || ''}`);
  if (context.dressCode === 'formal' && top && visiblePattern(top)
    && [layer, bottomIndex >= 0 ? items[bottomIndex] : undefined].some(item => item && visiblePattern(item)))
    return 'formal_patterned_tailoring_requires_quiet_base';
  if (context.dressCode === 'formal' && layer && bottomIndex >= 0) {
    const layerPattern = patternOf(layer), bottomPattern = patternOf(items[bottomIndex]);
    if (visiblePattern(layer) && visiblePattern(items[bottomIndex]) && layerPattern !== bottomPattern)
      return 'formal_tailoring_patterns_conflict';
    if (layerPattern === 'stripe' && bottomPattern !== 'stripe')
      return 'formal_striped_layer_requires_matching_trousers';
  }
  const refined = ['semi_classic', 'classic', 'old_money', 'business_casual'].includes(context.style)
    || context.dressCode === 'smart' || context.dressCode === 'formal';
  if (refined && top && layer) {
    const a = describeGarment(top), b = describeGarment(layer);
    if (a.subtype === 'field-shirt' && b.subtype === 'blazer')
      return 'utility_base_with_tailored_blazer';
  }
  if (refined && bottomIndex >= 0) {
    const outer = descriptors.find(d => d.slot === 'outerwear');
    const outerText = layer ? `${layer.name || ''} ${layer.description || ''}`.toLowerCase() : '';
    if (outer?.bulk === 'heavy' && /\bbiker\b/.test(outerText)
      && descriptors[bottomIndex].structured)
      return 'bulky_biker_with_refined_tailoring';
    if (/\b(work|office|business)\b/.test(context.occasion)
      && /\b(biker|trucker|racer)\b/.test(outerText)
      && shoeIndex >= 0 && descriptors[shoeIndex].formality === 'formal')
      return 'workwear_jacket_with_office_dress_footwear';
  }
  if (bottomIndex >= 0 && shoeIndex >= 0) {
    if (
      descriptors[bottomIndex].athletic &&
      descriptors[shoeIndex].formality === 'formal'
    )
      return 'formal_footwear_with_athletic_bottom';
    if (descriptors[bottomIndex].subtype === 'knit-trouser'
      && descriptors[shoeIndex].formality === 'formal')
      return 'dress_footwear_with_unstructured_knit_bottom';
  }
  // Refined work consistency (classic / old_money / semi_classic + Work):
  // each piece alone can clear its slot gate while the whole look reads
  // vacation or gym. Reject the combination, never the garment globally —
  // the same trainers serve Casual, the same resort layer serves Travel.
  const refinedWork =
    ['classic', 'old_money', 'semi_classic'].includes(context.style) &&
    /\b(work|office|business)\b/.test(context.occasion);
  if (refinedWork) {
    const shoeDesc = shoeIndex >= 0 ? descriptors[shoeIndex] : undefined;
    if (
      shoeDesc &&
      (shoeDesc.athletic ||
        shoeDesc.formality === 'athletic' ||
        ['sneaker', 'sport-sneaker', 'minimal-sneaker'].includes(shoeDesc.subtype))
    )
      return 'classic_work_requires_non_athletic_footwear';
    if (bottomIndex >= 0 && descriptors[bottomIndex].athletic)
      return 'classic_work_requires_non_athletic_bottom';
    const vacationCue = (item: GarmentEvidence | undefined) =>
      !!item &&
      /\b(resort|vacation|holiday|seaside|beach|tropical|hawaiian|camp[ -]?collar)\b/i.test(
        `${item.name || ''} ${item.description || ''}`,
      );
    if (vacationCue(layer) || (bottomIndex >= 0 && vacationCue(items[bottomIndex])))
      return 'classic_work_rejects_vacation_pieces';
  }
  // Formal dress consistency (old_money / classic + Formal): a checked casual
  // base or a leisure-cut layer each clears its individual formal gate, but
  // together they read casual. Dress shirts stay plain; tailoring stays
  // structured. Polo, suede, linen, cardigan and sneaker bans remain global
  // no-ops — only this style + dressCode combination is constrained.
  if (
    context.dressCode === 'formal' &&
    ['old_money', 'classic'].includes(context.style)
  ) {
    if (top && patternOf(top) === 'check')
      return 'formal_checked_base_breaks_dress_shirt_line';
    const layerText = layer ? `${layer.name || ''} ${layer.description || ''}`.toLowerCase() : '';
    if (
      layer &&
      /\b(leisure|unstructured|casual|jersey|knit|sweatshirt|hooded|soft[ -]?shoulder)\b/.test(layerText)
    )
      return 'formal_leisure_layer_breaks_structured_tailoring';
  }
  return null;
}

/** Scores relationships; hard authority (source/budget/ownership) stays with existing gates. */
export function outfitCoherence(
  items: GarmentEvidence[],
  context: StylingContext
): OutfitCoherence {
  const descriptors = items.map(describeGarment),
    bySlot = (s: string) => descriptors.find((d) => d.slot === s);
  const top = items.find((i) => i.macroCategory === 'top'),
    layer = items.find((i) => i.macroCategory === 'outerwear');
  const reasons: string[] = [],
    evidence: string[] = [];
  const dimensions = {
    layer: 0.7,
    dressCode: 0.7,
    palette: 0.7,
    proportion: 0.7,
    texture: 0.7,
    footwear: 0.7,
    style: 0.7,
  };
  if (top && layer) {
    const pair = layerCompatibility(top, layer, context);
    dimensions.layer = pair.score;
    reasons.push(...pair.reasons);
    evidence.push(...pair.evidence);
  }
  const bottom = bySlot('bottom'),
    shoe = bySlot('shoes');
  if (
    bySlot('outerwear')?.insulated &&
    shoe &&
    ['espadrille', 'sandal', 'driver'].includes(shoe.subtype)
  )
    reasons.push('insulated_layer_with_summer_footwear');
  for (const item of items) {
    const reason = garmentContextReason(item, context);
    if (reason) reasons.push(reason);
  }
  if (context.dressCode === 'formal')
    dimensions.dressCode = reasons.some((r) => r.startsWith('formal_')) ? 0 : 1;
  if (context.dressCode !== 'formal') {
    dimensions.dressCode =
      items.reduce((sum, item) => sum + scoreGarmentForContext(item, context), 0) /
      Math.max(1, items.length);
    if (
      descriptors.some((d) => d.evening) &&
      !/black[ -]?tie|tuxedo|gala|смокинг/i.test(`${context.occasion} ${context.request}`)
    ) {
      dimensions.dressCode = Math.min(dimensions.dressCode, 0.2);
      evidence.push('evening_tailoring_overdresses_request');
    }
  }
  if (bottom && shoe) {
    if (
      bottom.structured &&
      ['oxford', 'derby', 'monk', 'loafer', 'chelsea', 'dress-boot'].includes(shoe.subtype)
    ) {
      dimensions.footwear = 1;
      evidence.push('dress_footwear_with_tailored_bottom');
    }
    if (context.dressCode === 'casual') {
      if (['minimal-sneaker','sneaker'].includes(shoe.subtype) && ['chino','jeans','trouser'].includes(bottom.subtype)) {
        dimensions.footwear = 1;
        evidence.push('casual_footwear_with_casual_bottom');
      }
      if (['oxford','monk'].includes(shoe.subtype) && bySlot('outerwear')?.subtype !== 'blazer') {
        dimensions.footwear = 0.45;
        evidence.push('formal_footwear_overdresses_casual_layers');
      }
    }
    if (bottom.structured && shoe.rugged) dimensions.footwear = 0.15;
    if (bottom.structured && shoe.casualSole && ['smart', 'formal'].includes(context.dressCode)) {
      dimensions.footwear = Math.min(dimensions.footwear, 0.35);
      evidence.push('casual_sole_breaks_tailored_footwear_line');
    }
    if (bottom.volume === 'wide' && shoe.subtype === 'driver') {
      dimensions.proportion = 0.45;
      evidence.push('wide_hem_over_delicate_driver');
    }
    // A technical runner under a tailored layer clashes even in smart
    // casual. Minimal sneakers are exempt: they are the refined-casual
    // exception, not gym wear.
    if (
      shoe.subtype === 'sport-sneaker' &&
      bySlot('outerwear') &&
      ['blazer', 'coat'].includes(bySlot('outerwear')!.subtype)
    ) {
      dimensions.footwear = Math.min(dimensions.footwear, 0.4);
      evidence.push('sport_sneaker_with_tailored_layer');
    }
  }
  const volumes = descriptors.filter((d) => d.volume === 'wide').length;
  if (volumes >= 3) {
    dimensions.proportion = 0.5;
    evidence.push('multiple_high_volume_pieces');
  }
  const outer = bySlot('outerwear');
  const base = bySlot('top');
  if (base && outer && competingUpperPatterns(base, outer)) {
    dimensions.palette = Math.min(dimensions.palette, 0.55);
    evidence.push('competing_upper_patterns');
  }
  if (outer?.structured && !outer.relaxedTailoring && bottom?.structured && ['smart', 'formal'].includes(context.dressCode)) {
    if ((outer.volume === 'slim' && bottom.volume === 'wide') || (outer.volume === 'wide' && bottom.volume === 'slim')) {
      dimensions.proportion = Math.min(dimensions.proportion, 0.45);
      evidence.push('tailoring_volume_contrast');
    } else if (outer.volume !== 'unknown' && outer.volume === bottom.volume && volumes < 3) {
      dimensions.proportion = 0.9;
      evidence.push('consistent_tailoring_volume');
    }
    if (outer.materials.some(material => ['wool', 'linen', 'cotton'].includes(material) && bottom.materials.includes(material))) {
      dimensions.texture = 0.9;
      evidence.push('compatible_tailoring_fabrics');
    }
  }
  const heavy = descriptors.some((d) => d.bulk === 'heavy'),
    airy = descriptors.some((d) => d.materials.includes('linen'));
  if (heavy && airy) {
    dimensions.texture = 0.4;
    evidence.push('heavy_and_airy_texture_contrast');
  }
  const dressShoe = shoe && !shoe.casualSole
    && (shoe.formality === 'formal' || ['oxford', 'derby', 'monk', 'dress-boot'].includes(shoe.subtype));
  // Several casual cues must agree before lowering a refined look. Tencel,
  // denim tailoring and zip jackets remain legitimate on their own.
  if (outer?.relaxedTailoring && bottom?.subtype === 'jeans' && dressShoe) {
    dimensions.footwear = Math.min(dimensions.footwear, 0.5);
    dimensions.texture = Math.min(dimensions.texture, base?.fluidFabric ? 0.45 : 0.6);
    evidence.push('leisure_tailoring_and_denim_with_dress_footwear');
  }
  if (bottom?.subtype === 'jeans' && bottom.volume === 'wide'
    && (dressShoe || (shoe?.subtype === 'loafer' && !shoe.casualSole))) {
    dimensions.proportion = Math.min(dimensions.proportion, dressShoe ? 0.45 : 0.6);
    dimensions.footwear = Math.min(dimensions.footwear, dressShoe ? 0.55 : 0.65);
    evidence.push('wide_denim_hem_with_dress_footwear');
  }
  const refinedOffice = context.dressCode === 'smart'
    && /\b(work|office|business)\b/.test(context.occasion)
    && ['old_money', 'classic', 'semi_classic', 'business_casual'].includes(context.style);
  const splitOfficeRegister = refinedOffice && base?.fluidFabric && outer?.croppedZip
    && bottom?.structured && dressShoe;
  if (splitOfficeRegister) {
    dimensions.layer = Math.min(dimensions.layer, 0.45);
    dimensions.footwear = Math.min(dimensions.footwear, 0.5);
    evidence.push('fluid_base_and_short_zip_layer_split_office_register');
  }
  const disconnectedCasualTailoring = context.dressCode === 'casual'
    && outer?.subtype === 'blazer' && !outer.relaxedTailoring
    && base?.volume === 'wide' && base.loudBranding
    && shoe?.vividFootwear && ['sneaker', 'minimal-sneaker'].includes(shoe.subtype);
  if (disconnectedCasualTailoring) {
    dimensions.layer = Math.min(dimensions.layer, 0.4);
    dimensions.footwear = Math.min(dimensions.footwear, 0.45);
    evidence.push('graphic_volume_tailoring_and_vivid_shoes_lack_shared_register');
  }
  const travel = /\b(travel|trip|sightseeing)\b/.test(`${context.occasion} ${context.request}`);
  const activeIntent = /\b(gym|workout|training|fitness|cross[ -]?train(?:ing)?|hik(?:e|ing)|trek(?:king)?|trail|sport)\b/.test(`${context.occasion} ${context.request}`);
  if (context.dressCode === 'casual' && travel && !activeIntent && shoe?.trainingFootwear
    && bottom && !bottom.athletic && base && !base.athletic) {
    dimensions.footwear = Math.min(dimensions.footwear, 0.4);
    evidence.push('performance_training_shoe_without_active_travel_intent');
  }
  // Whole-outfit color relationships (tonal, contrast, accent balance).
  // Deliberately SOFT: the harmony score moves ranking through this
  // dimension, but palette reasons never fail the look — hard rejections
  // stay with the occasion/dress-code judges.
  const harmony = scoreColorHarmony(
    items.map((item, index) => ({
      colors: colorTokensOf(item),
      colorFamily: colorFamilyOf(item),
      pattern: descriptors[index] ? descriptors[index].pattern : 'unknown',
    }))
  );
  dimensions.palette = harmony.score;
  for (const reason of harmony.reasons) evidence.push(`palette_${reason}`);
  evidence.push(...harmony.evidence);
  const intent = outfitStyleIntent(descriptors, context);
  dimensions.style = intent.score * 0.75 + 0.25 *
    items.reduce((sum, item) => sum + scoreGarmentForContext(item, context), 0) / Math.max(1, items.length);
  evidence.push(`style_intent:${intent.id}`);
  const officePenalty = items.reduce((total, item, i) => total + officeRefinementPenalty(item, context, descriptors[i]), 0);
  if (officePenalty > 0) {
    dimensions.style = Math.min(dimensions.style, Math.max(0.3, 1 - officePenalty));
    evidence.push('casual_construction_weakens_office_refinement');
  }
  if (context.dressCode === 'smart' && /\b(work|office|business)\b/.test(context.occasion)
    && top && layer && ['stripe', 'check', 'print'].includes(bySlot('top')!.pattern)
    && ['stripe', 'check', 'print'].includes(outer!.pattern)) {
    dimensions.palette = Math.min(dimensions.palette, 0.55);
    evidence.push('competing_office_upper_patterns');
  }
  // Evaluate the outfit's pattern hierarchy and dress register as a whole.
  // One accent and smart-casual bridges remain supported; uncertainty is neutral.
  const motifs=descriptors.filter(d=>['stripe','check','print','graphic'].includes(d.pattern));
  if(motifs.length>1 && new Set(motifs.map(d=>d.pattern)).size>1
    && motifs.some(d=>d.patternProminence==='prominent')) {
    dimensions.palette=Math.min(dimensions.palette,0.45);
    evidence.push('competing_outfit_motifs');
  }
  const upper=descriptors.filter(d=>['top','outerwear'].includes(d.slot));
  if(context.dressCode!=='athletic' && upper.length===2 && upper.every(d=>d.formality==='casual')
    && bottom?.formality==='formal' && shoe?.formality==='formal') {
    dimensions.footwear=Math.min(dimensions.footwear,0.45);
    evidence.push('upper_and_lower_registers_lack_styling_bridge');
  }
  const registerClash = wholeOutfitRegisterClash(items, context);
  if (registerClash) reasons.push(registerClash);
  const valid = !reasons.length;
  return {
    valid,
    score: valid ? Object.values(dimensions).reduce((s, n) => s + n, 0) / 7 : 0,
    reasons: [...new Set(reasons)],
    evidence: [...new Set(evidence)],
    dimensions,
  };
}
