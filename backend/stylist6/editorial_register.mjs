/**
 * Cross-slot editorial admission before visual inference. Reject only
 * combinations with explicit catalog evidence of incompatible formality.
 * Unknown names/subtypes abstain; never infer fit, rights or availability.
 */
export const EDITORIAL_REGISTER_VERSION = 's6-cross-slot-register-v1';
const norm = value => String(value || '').toLowerCase().replace(/[_-]+/g, ' ').trim();
const slot = (look,name) => (look?.items || []).find(item => item?.macroCategory === name);
const described = item => norm([item?.subtype,item?.name].filter(Boolean).join(' '));
const elevatedStyle = /^(classic|old money|semi classic|smart casual|business casual)$/;
const elevatedOccasion = /\b(work|office|business|meeting|date|dinner|romantic|cocktail|theater|opera|wedding)\b/;
export function editorialRegisterVeto(look,{style,occasion,brief=''}={}) {
  const elevated = elevatedStyle.test(norm(style)) &&
      (elevatedOccasion.test(norm(occasion)) || elevatedOccasion.test(norm(brief)));
  if(!elevated)return null;
  const top=described(slot(look,'top')), layer=described(slot(look,'outerwear'));
  const bottom=described(slot(look,'bottom')), shoes=described(slot(look,'shoes'));
  if(!top||!bottom||!shoes)return null;
  const tailored=/\b(tailored|suit|dress trouser|pleated|smart stretch|wool trouser|slacks)\b/.test(bottom);
  const blazer=/\b(blazer|suit jacket|sartorial)\b/.test(layer);
  const dressShoes=/\b(oxford|derby|brogue|formal|dress shoe|dress loafers?|leather loafers?)\b/.test(shoes);
  const tee=/\b(t shirt|tee|oversized t shirt|graphic t shirt|jersey tee)\b/.test(top);
  const flannel=/\b(flannel shirt|flannel)\b/.test(top);
  const athletic=/\b(sneaker|trainer|running shoe|hybrid|s lite)\b/.test(shoes);
  if(blazer&&tailored&&athletic)return 'athletic_shoe_with_formal_tailoring';
  if(tee&&dressShoes&&(tailored||!layer))return 'casual_tee_with_dress_footwear';
  if(flannel&&tailored&&dressShoes&&/\b(work|office|business|meeting)\b/.test(norm(occasion)))
    return 'flannel_with_office_tailoring';
  return null;
}
