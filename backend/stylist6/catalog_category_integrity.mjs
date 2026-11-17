/**
 * Trust boundary for merchant category metadata.
 *
 * Source categories are normally authoritative. However a small subset of
 * legacy feeds have unambiguous contradictions (e.g. "Oxford Shirt" filed as
 * bottoms, or "Tapered Jeans" filed as tops). Never silently move those SKUs
 * into another role: quarantine only if BOTH the product title and the
 * independent merchant URL path agree on the opposite garment role.
 *
 * This is category integrity, not a fashion taste rule or a stock assertion.
 * A single keyword is deliberately insufficient (Oxford shirts, bootcut
 * jeans, brand-name "Tommy Jeans" and multi-piece sets are ambiguous).
 */
const TOP = /\b(?:t[- ]?shirts?|shirts?|polos?|poloshirts?|sweaters?|cardigans?|hoodies?|blazers?|jackets?|overshirts?|overcoats?|coats?|pullovers?|crewnecks?|jumpers?)\b/i;
const BOTTOM = /\b(?:jeans|trousers?|pants|chinos?|joggers?|shorts|slacks)\b/i;
const SHOES = /\b(?:sneakers?|loafers?|boots?|trainers?|sandals?|brogues?|moccasins?|oxford shoes?|derby shoes?)\b/i;

const CATEGORY_ROLE = { tops: 'upper_body', outerwear: 'upper_body', bottoms: 'lower_body', shoes: 'shoes' };
const GARMENT_ROLE = { upper_body: 'upper_body', lower_body: 'lower_body', shoes: 'shoes', footwear: 'shoes' };
const ROLE_TARGETS = {
  upper_body: { category: 'tops', garment_type: 'upper_body' },
  lower_body: { category: 'bottoms', garment_type: 'lower_body' },
  shoes: { category: 'shoes', garment_type: 'shoes' },
};

export function explicitGarmentRole(text) {
  const value = String(text || '');
  const found = [TOP.test(value) && 'upper_body', BOTTOM.test(value) && 'lower_body', SHOES.test(value) && 'shoes'].filter(Boolean);
  return found.length === 1 ? found[0] : null;
}

export function merchantPathRole(sourceUrl) {
  try {
    const url = new URL(String(sourceUrl || ''));
    if (url.protocol !== 'https:') return null;
    const path = decodeURIComponent(url.pathname).replace(/[-_/]+/g, ' ');
    return explicitGarmentRole(path);
  } catch {
    return null;
  }
}

export function assessCatalogCategory(row) {
  if (!row || typeof row !== 'object') return { status: 'unreadable_row' };
  const category = CATEGORY_ROLE[String(row.category || '').toLowerCase()];
  const garment = GARMENT_ROLE[String(row.garment_type || '').toLowerCase()];
  if (category && garment && category !== garment) {
    return { status: 'native_role_conflict', categoryRole: category, garmentRole: garment };
  }
  const nativeRole = category || garment;
  if (!nativeRole) return { status: 'outside_supported_roles' };
  const titleRole = explicitGarmentRole(row.name);
  if (!titleRole || titleRole === nativeRole) return { status: 'consistent_or_ambiguous' };
  const pathRole = merchantPathRole(row.source_url);
  if (pathRole !== titleRole) {
    return { status: 'needs_manual_review', nativeRole, titleRole, pathRole };
  }
  return {
    status: 'confirmed_category_conflict',
    nativeRole, titleRole, pathRole,
    suggested: ROLE_TARGETS[titleRole],
  };
}
