/**
 * Cross-slot editorial admission before visual inference.
 *
 * This is deliberately narrow: reject only combinations with *explicit*
 * catalog evidence of a formality collision. Never infer fit, gender, or
 * product availability. Unknown names/subtypes abstain from this gate.
 * A pinned item cannot be quietly swapped: if it causes a collision,
 * return a short deck rather than violating the user's hard constraints.
 */
export const EDITORIAL_REGISTER_VERSION = 's6-cross-slot-register-v1';

const norm = (value) => String(value || '').toLowerCase().replace(/[_-]+/g, ' ').trim();
const slot = (look, name) => (look?.items || []).find(item => item?.macroCategory === name);
const described = (item) => norm([item?.subtype, item?.name].filter(Boolean).join(' '));
const exact = (value, regex) => regex.test(value);
const elevatedStyle = /^(classic|old money|semi classic|smart casual|business casual)$/;
const elevatedOccasion = /\b(work|office|business|meeting|date|dinner|romantic|cocktail|theater|opera|wedding)\b/;

export function editorialRegisterVeto(look, { style, occasion, brief = '', request = '', weather = null, sourceMode = null } = {}) {
  const isElevated = elevatedStyle.test(norm(style)) &&
    (elevatedOccasion.test(norm(occasion)) || elevatedOccasion.test(norm(brief)));
  if (!isElevated) return null;
  // Actual full-catalog Classic / Work at +9C: "Navy Rib Vest" was chosen
  // as the ONLY outer layer. A vest or gilet alone cannot protect in cool
  // weather. The user can still explicitly request their own waistcoat.
  const temperature = Number.isFinite(weather?.temp) ? weather.temp : null;
  if (sourceMode === 'catalog' && temperature !== null && temperature < 12 &&
      !/\b(?:vest|gilet|waistcoat|sweater vest)\b/i.test(`${brief} ${request}`)) {
    const layerItem = slot(look, 'outerwear');
    if (layerItem && /\b(?:vests?|gilets?|waistcoats?)\b/i.test(described(layerItem)))
      return 'cool_weather_sleeveless_single_layer';
  }
  const top = described(slot(look, 'top'));
  const layer = described(slot(look, 'outerwear'));
  const bottom = described(slot(look, 'bottom'));
  const shoes = described(slot(look, 'shoes'));
  if (!top || !bottom || !shoes) return null;
  // Generalizable formality floor for a *polished Classic Work* request.
  // A coordinated knit + zip cardigan + chinos + leather loafers can look
  // attractive on a flat lay, yet is too casual when the user asks for
  // professional tailored office styling. Judge the WHOLE outfit structure,
  // not a named SKU or a particular brand. Explicit casual-Friday requests
  // and personally pinned owned clothing are not blocked.
  const classicPolishedOffice = sourceMode === 'catalog' && norm(style) === 'classic' &&
    /\b(work|office|business|meeting|conference)\b/.test(norm(occasion)) &&
    !/\b(casual|relaxed|informal|dress-down)\b/i.test(`${brief} ${request}`);
  if (classicPolishedOffice) {
    const outer = slot(look, 'outerwear');
    const lower = slot(look, 'bottom');
    const structuralLayer = /^(blazer|suit-jacket|coat|overcoat|topcoat|peacoat|trench|tailored-jacket)$/.test(
      String(outer?.subtype || '').toLowerCase()) ||
      /\b(suit jacket|blazer|overcoat|wool coat|topcoat|peacoat|trench coat)\b/i.test(layer);
    const structuralBottom = /^(tailored-trouser|dress-trouser|suit-trouser|wool-trouser|pleated-trouser)$/.test(
      String(lower?.subtype || '').toLowerCase()) ||
      /\b(suit trousers?|wool trousers?|tailored trousers?|dress trousers?|pleated trousers?|wool flannel pants)\b/i.test(bottom);
    const relaxedTop = /\b(knit|knitted|sweater|turtleneck|polo|tee|jersey|half[- ]?zip|quarter[- ]?zip)\b/i.test(top) ||
      /^(fine-knit|sweater|polo|t-shirt|henley)$/.test(String(slot(look, 'top')?.subtype || '').toLowerCase());
    if (!structuralLayer && !structuralBottom && relaxedTop)
      return 'polished_classic_office_requires_tailored_structure';
  }
  const tailoring = exact(bottom, /\b(tailored|suit|dress trouser|pleated|smart stretch|wool trouser|slacks)\b/);
  const blazer = exact(layer, /\b(blazer|suit jacket|sartorial)\b/);
  const formalShoes = exact(shoes, /\b(oxford|derby|brogue|formal|dress shoe|dress loafers?|leather loafers?)\b/);
  const casualTop = exact(top, /\b(t shirt|tee|oversized t shirt|graphic t shirt|jersey tee)\b/);
  const loudPrintTop = exact(top, /\b(leopard|cheetah|zebra|snakeskin|animal print|hawaiian|tropical|camo)\b/);
  const flannel = exact(top, /\b(flannel shirt|flannel)\b/);
  const henley = exact(top, /\bhenleys?\b/);
  const casualShoe = exact(shoes, /\b(sneaker|trainer|running shoe|hybrid|s lite)\b/);
  const shoeItem = slot(look, 'shoes');
  const comfortLoafer = exact(shoes, /\b(spherica|flextride plus|moner w 2fit)\b/)
    || /^(?:shop_)?geox-(?:u66mac|u44q6d|u56q6a|u65hyb000(?:22|46)|u66cjc00047)/i.test(String(shoeItem?.id || ''));
  const suedeShoe = exact(shoes, /\bsuede\b/);
  const workOrFormalSuit = /\b(work|office|business|formal)\b/.test(norm(occasion))
    && /^(classic|old money|business casual)$/.test(norm(style));
  // An athletic-soled sneaker or comfort loafer with a full business suit is
  // not a coherent direction in Classic/Old Money office styling.
  if (blazer && tailoring && casualShoe) return 'athletic_shoe_with_formal_tailoring';
  if (blazer && tailoring && comfortLoafer) return 'comfort_sole_with_formal_tailoring';
  if (blazer && tailoring && suedeShoe && workOrFormalSuit) return 'suede_footwear_with_formal_suit';
  if (loudPrintTop && (tailoring || formalShoes)) return 'loud_print_with_dress_tailoring';
  if (casualTop && formalShoes && (tailoring || !layer)) return 'casual_tee_with_dress_footwear';
  // Real Classic / Work replay: a "RIBBED LONG SLEEVED HENLEY SHIRT"
  // was falsely classified as a dress shirt and paired with pleated suit
  // trousers and Derby shoes. The missing dress collar is source-evident.
  // Allow casual/date outfits and a user's explicit Henley request.
  const polishedWork = /\b(work|office|business|meeting)\b/.test(norm(occasion)) &&
    /^(classic|old money|semi classic|business casual)$/.test(norm(style));
  if (henley && tailoring && formalShoes && polishedWork &&
      !/\bhenleys?\b/i.test(`${brief} ${request}`)) return 'henley_with_office_tailoring';
  // Flannel and a business-tailored outfit should not displace a proper
  // shirt on a formal work brief. The rule is NOT applied to leisure outfits.
  if (flannel && tailoring && formalShoes && /\b(work|office|business|meeting)\b/.test(norm(occasion)))
    return 'flannel_with_office_tailoring';
  return null;
}
