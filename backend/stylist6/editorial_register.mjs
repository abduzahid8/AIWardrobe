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

export function editorialRegisterVeto(look, { style, occasion, brief = '' } = {}) {
  const isElevated = elevatedStyle.test(norm(style)) &&
    (elevatedOccasion.test(norm(occasion)) || elevatedOccasion.test(norm(brief)));
  if (!isElevated) return null;
  const top = described(slot(look, 'top'));
  const layer = described(slot(look, 'outerwear'));
  const bottom = described(slot(look, 'bottom'));
  const shoes = described(slot(look, 'shoes'));
  if (!top || !bottom || !shoes) return null;
  const tailoring = exact(bottom, /\b(tailored|suit|dress trouser|pleated|smart stretch|wool trouser|slacks)\b/);
  const blazer = exact(layer, /\b(blazer|suit jacket|sartorial)\b/);
  const formalShoes = exact(shoes, /\b(oxford|derby|brogue|formal|dress shoe|dress loafers?|leather loafers?)\b/);
  const casualTop = exact(top, /\b(t shirt|tee|oversized t shirt|graphic t shirt|jersey tee)\b/);
  const loudPrintTop = exact(top, /\b(leopard|cheetah|zebra|snakeskin|animal print|hawaiian|tropical|camo)\b/);
  const flannel = exact(top, /\b(flannel shirt|flannel)\b/);
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
  // Flannel and a business-tailored outfit should not displace a proper
  // shirt on a formal work brief. The rule is NOT applied to leisure outfits.
  if (flannel && tailoring && formalShoes && /\b(work|office|business|meeting)\b/.test(norm(occasion)))
    return 'flannel_with_office_tailoring';
  return null;
}
